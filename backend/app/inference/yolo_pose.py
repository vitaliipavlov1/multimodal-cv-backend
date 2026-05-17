"""
YOLO26 pose estimation inference module.

Model  : YOLO26s-pose TensorRT FP16 — NMS-free end-to-end
Input  : FP16 [B, 3, 640, 640]
Output : FP32 [B, 300, 57]
         Columns: [x1, y1, x2, y2, conf, cls, kpt0_x, kpt0_y, kpt0_c, ..., kpt16_x, kpt16_y, kpt16_c]
         4 + 1 + 1 + 17*3 = 57 (note: Triton config uses dims=[300, 57])
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tritonclient.grpc.aio as grpcclient

from .triton_client import get_triton_client

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, size: int) -> Tuple[float, int, int, int, int]:
    """Resize image preserving aspect ratio with symmetric padding."""
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    pad_l, pad_t = (size - nw) // 2, (size - nh) // 2
    return r, pad_l, pad_t, nw, nh


def _unpad_boxes(boxes: np.ndarray, r: float, pad_l: int, pad_t: int) -> np.ndarray:
    """Remove letterbox padding and scale boxes back to original frame coords."""
    if boxes.size == 0:
        return boxes
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / r
    return boxes


# ─────────────────────────────────────────────────────────────────
#  YOLO26 Pose Triton module
# ─────────────────────────────────────────────────────────────────

class YOLO26PoseModule:
    """
    Async YOLO26s-pose detection via Triton Inference Server.

    Triton model config:
        input  → name="images"  dtype=FP16  dims=[3, 640, 640]
        output → name="output0" dtype=FP32  dims=[300, 57]

    Output column layout (57 values per detection):
        [x1, y1, x2, y2, conf, cls, kpt0_x, kpt0_y, kpt0_c, ..., kpt16_x, kpt16_y, kpt16_c]
    Boxes are already xyxy decoded — no xywh conversion or NMS needed.
    """

    def __init__(
        self,
        model_name:  str   = "yolo_pose",
        url:         str   = "127.0.0.1:8001",
        target_size: int   = 640,
        conf_thres:  float = 0.25,
        max_det:     int   = 300,
    ):
        self.model_name  = model_name
        self.url         = url
        self.target_size = target_size
        self.conf_thres  = conf_thres
        self.max_det     = max_det

        self.client: Optional[grpcclient.InferenceServerClient] = None
        self._input_name:  Optional[str] = None
        self._output_name: Optional[str] = None
        self._dtype:       Optional[str] = None
        self._use_fp16:    bool          = False
        self._setup_lock   = asyncio.Lock()

        self._fps_ema:   Dict[str, float] = {}
        self._fps_alpha: float            = 0.1

    async def setup(self) -> None:
        async with self._setup_lock:
            if self.client is None:
                self.client = await get_triton_client(self.url)
            if self._input_name is None:
                meta = await self.client.get_model_metadata(self.model_name)
                self._input_name  = meta.inputs[0].name
                self._output_name = meta.outputs[0].name
                self._dtype       = meta.inputs[0].datatype
                self._use_fp16    = self._dtype in ("FP16", "FLOAT16")
                logger.info(
                    "YOLO26 pose ready | model=%s  dtype=%s  fp16=%s",
                    self.model_name, self._dtype, self._use_fp16,
                )

    def _preprocess(
        self, frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple]]:
        dtype = np.float16 if self._use_fp16 else np.float32
        batch = np.zeros(
            (len(frames), 3, self.target_size, self.target_size), dtype=dtype
        )
        metas: List[Tuple] = []
        for i, frame in enumerate(frames):
            h0, w0 = frame.shape[:2]
            r, pad_l, pad_t, nw, nh = _letterbox(frame, self.target_size)
            canvas = np.zeros((self.target_size, self.target_size, 3), np.uint8)
            canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = cv2.resize(frame, (nw, nh))
            img = canvas[:, :, ::-1].astype(dtype) / dtype(255.0)
            batch[i] = img.transpose(2, 0, 1)
            metas.append((r, pad_l, pad_t, w0, h0))
        return batch, metas

    def _postprocess(
        self, pred: np.ndarray, meta: Tuple
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decode YOLO26-pose NMS-free output.
        Boxes are already xyxy — no conversion needed.
        Keypoints: columns [6:] reshaped to [N, 17, 3] (x, y, conf).
        """
        if pred is None:
            return None
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if pred.ndim != 2 or pred.shape[1] < 6:
            return None

        r, pad_l, pad_t, w0, h0 = meta

        boxes  = pred[:, :4].astype(np.float32)  # xyxy, already decoded
        scores = pred[:, 4].astype(np.float32)
        kpts   = pred[:, 6:].astype(np.float32)  # col 5 = cls, col 6+ = keypoints

        mask = scores > self.conf_thres
        if not mask.any():
            return None

        boxes, scores, kpts = boxes[mask], scores[mask], kpts[mask]

        # Scale boxes back to original frame — no xywh2xyxy, already decoded
        boxes = _unpad_boxes(boxes, r, pad_l, pad_t)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

        # Cap detections (NMS already applied inside model)
        if boxes.shape[0] > self.max_det:
            idx = scores.argsort()[::-1][:self.max_det]
            boxes, scores, kpts = boxes[idx], scores[idx], kpts[idx]

        # Reshape and unpad keypoints: (N, 51) → (N, 17, 3)
        if kpts.shape[1] % 3 != 0:
            return None
        kpts = kpts.reshape(-1, kpts.shape[1] // 3, 3)
        kpts[..., 0] = np.clip((kpts[..., 0] - pad_l) / r, 0, w0)
        kpts[..., 1] = np.clip((kpts[..., 1] - pad_t) / r, 0, h0)

        return boxes, scores, kpts

    async def infer_batch(
        self, frames: List[np.ndarray], stream_id: str
    ) -> Dict[str, Any]:
        if not frames:
            return {"results": [], "fps": 0.0, "batch_size": 0, "stream_id": stream_id}

        if self.client is None:
            await self.setup()

        imgs, metas = self._preprocess(frames)
        t0 = time.perf_counter()

        inp = grpcclient.InferInput(self._input_name, imgs.shape, self._dtype)
        inp.set_data_from_numpy(imgs)
        out = grpcclient.InferRequestedOutput(self._output_name)

        try:
            res = await asyncio.wait_for(
                self.client.infer(
                    model_name=self.model_name,
                    inputs=[inp],
                    outputs=[out],
                    model_version="1",
                ),
                timeout=5.0,
            )
            preds = res.as_numpy(self._output_name)
        except asyncio.TimeoutError:
            raise RuntimeError("YOLO26 pose Triton inference timeout")
        except asyncio.CancelledError:
            raise

        if preds is None or not isinstance(preds, np.ndarray):
            return {
                "results":    [None] * len(frames),
                "fps":        0.0,
                "batch_size": len(frames),
                "stream_id":  stream_id,
            }

        if preds.ndim == 2:
            preds = preds[None, ...]   # (N, 57) → (1, N, 57)

        results: List[Optional[Dict]] = []
        for i in range(len(frames)):
            det = self._postprocess(preds[i], metas[i])
            if det is None:
                results.append(None)
            else:
                boxes, scores, kpts = det
                n = len(boxes)
                results.append({
                    "boxes":     boxes.tolist(),
                    "scores":    [float(s) for s in scores],
                    "classes":   [0] * n,
                    "keypoints": kpts.tolist(),
                })

        fps = len(frames) / max(1e-9, time.perf_counter() - t0)
        prev = self._fps_ema.get(stream_id)
        self._fps_ema[stream_id] = fps if prev is None else (
            self._fps_alpha * fps + (1.0 - self._fps_alpha) * prev
        )

        return {
            "results":    results,
            "fps":        self._fps_ema[stream_id],
            "batch_size": len(frames),
            "stream_id":  stream_id,
        }
