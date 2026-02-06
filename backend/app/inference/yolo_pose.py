# seayolo_pose_triton_batch_module.py
import asyncio
import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import tritonclient.grpc.aio as grpcclient
from .triton_client import get_triton_client


# -------------------- Utility Functions --------------------
def letterbox_resize(img: np.ndarray, size: int) -> Tuple[float, int, int, int, int]:
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    pad_l, pad_t = (size - nw) // 2, (size - nh) // 2
    return r, pad_l, pad_t, nw, nh


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    return np.column_stack([
        boxes[:, 0] - boxes[:, 2] / 2,
        boxes[:, 1] - boxes[:, 3] / 2,
        boxes[:, 0] + boxes[:, 2] / 2,
        boxes[:, 1] + boxes[:, 3] / 2
    ])


def scale_coords_unpad(boxes: np.ndarray, r: float, pad_l: int, pad_t: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / r
    return boxes


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_thres]

    return np.array(keep, dtype=int)


# -------------------- YOLOv8 Pose Triton Module --------------------
class YOLOv8lPoseTritonModule:
    def __init__(
        self,
        model_name: str = "yolo_pose",
        url: str = "127.0.0.1:8001",
        target_size: int = 640,
        warmup_runs: int = 2,
        conf_thres: float = 0.25,
        use_nms: bool = True,
        max_candidates: int = 1000,
        iou_thres: float = 0.45
    ):
        self.model_name = model_name
        self.url = url
        self.target_size = target_size
        self.warmup_runs = warmup_runs
        self.conf_thres = conf_thres
        self.use_nms = use_nms
        self.max_candidates = max_candidates
        self.iou_thres = iou_thres
        self.max_det: int = 300

        self.client: Optional[grpcclient.InferenceServerClient] = None
        self._input_name = None
        self._output_name = None
        self._dtype = None
        self._use_fp16 = (self._dtype == "FP16")
        self._setup_lock = asyncio.Lock()

        # FPS EMA per stream
        self._fps_ema: Dict[str, float] = {}
        self._fps_alpha = 0.1

    # -------------------- Setup --------------------
    async def setup(self):
        async with self._setup_lock:
            if self.client is None:
                self.client = await get_triton_client(self.url)

            if self._input_name is None:
                meta = await self.client.get_model_metadata(self.model_name)
                self._input_name = meta.inputs[0].name
                self._output_name = meta.outputs[0].name
                self._dtype = meta.inputs[0].datatype
                self._use_fp16 = False
                # await self._warmup()

    async def _warmup(self):
        dummy = np.zeros(
            (1, 3, self.target_size, self.target_size),
            np.float16 if self._use_fp16 else np.float32
        )
        inp = grpcclient.InferInput(self._input_name, dummy.shape, self._dtype)
        inp.set_data_from_numpy(dummy)
        out = grpcclient.InferRequestedOutput(self._output_name)
        for _ in range(self.warmup_runs):
            try:
                await self.client.infer(self.model_name, [inp], [out])
            except Exception:
                break

    # -------------------- Preprocess --------------------
    def _preprocess_batch(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[float, int, int, int, int]]]:

        #print(">>> ENTER preprocess", flush=True)

        batch = np.zeros(
            (len(frames), 3, self.target_size, self.target_size),
            np.float16 if self._use_fp16 else np.float32
        )
        metas = []

        for i, f in enumerate(frames):
            h0, w0 = f.shape[:2]
            r, pad_l, pad_t, nw, nh = letterbox_resize(f, self.target_size)

            canvas = np.zeros((self.target_size, self.target_size, 3), np.uint8)
            canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = cv2.resize(f, (nw, nh))

            img = canvas[:, :, ::-1] / 255.0
            batch[i] = img.transpose(2, 0, 1)
            metas.append((r, pad_l, pad_t, w0, h0))

        #print(">>> EXIT preprocess", flush=True)

        return batch, metas

    # -------------------- Postprocess --------------------
    def _postprocess_single(self, pred: np.ndarray, meta):

        #print(
            #"[POSE RAW PRED]",
            #"shape=", None if pred is None else pred.shape,
            #flush=True
        #)

        if pred is None:
            return None

        # (1, N, D) → (N, D)
        if pred.ndim == 3:
            pred = pred[0]

        # (D, N) → (N, D)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        #print("[POSE FIX] pred shape:", pred.shape, flush=True)

        #print(
            #"[POSE PRED FORMAT]",
            #"pred_shape=", pred.shape,
            #"sample_row=", pred[0, :10].tolist() if pred.shape[0] > 0 else None,
            #flush=True
        #)

        if pred.ndim != 2 or pred.shape[1] < 7:
            return None

        #print(
            #"[POSE RAW PRED NORMALIZED]",
            #"shape=", pred.shape,
            #flush=True
        #)

        r, pad_l, pad_t, w0, h0 = meta

        boxes = pred[:, :4].astype(np.float32)

        scores = pred[:, 4]  # ← ТОЛЬКО objectness
        cls_ids = np.zeros(len(scores), dtype=int)  # pose = один класс

        kpts = pred[:, 5:]  # ← ВСЕ ОСТАЛЬНОЕ — keypoints

        mask = scores > self.conf_thres

        #print(
            #"[POSE CONF]",
            #"conf_thres=", self.conf_thres,
            #"candidates=", int(mask.sum()),
            #flush=True
        #)

        if not mask.any():
            return None


        boxes, scores, cls_ids, kpts = (
            boxes[mask],
            scores[mask],
            cls_ids[mask],
            kpts[mask]
        )

        boxes = scale_coords_unpad(xywh2xyxy(boxes), r, pad_l, pad_t)

        # --- performance guard before NMS ---
        if (
                self.use_nms
                and self.max_candidates
                and scores.shape[0] > self.max_candidates
        ):
            topk = scores.argsort()[-self.max_candidates:]
            boxes = boxes[topk]
            scores = scores[topk]
            cls_ids = cls_ids[topk]
            kpts = kpts[topk]


        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

        if self.use_nms:
            keep = nms_numpy(boxes, scores, self.iou_thres)
            boxes, scores, cls_ids, kpts = (
                boxes[keep],
                scores[keep],
                cls_ids[keep],
                kpts[keep]
            )


        if boxes.shape[0] > self.max_det:
            boxes = boxes[:self.max_det]
            scores = scores[:self.max_det]
            cls_ids = cls_ids[:self.max_det]
            kpts = kpts[:self.max_det]



        if kpts.shape[1] % 3 != 0:
            return None

        num_kpts = kpts.shape[1] // 3
        kpts = kpts.reshape(-1, num_kpts, 3)

        #print(
            #"[POSE KPTS SAMPLE]",
            #"raw_kpt_xy=",
            #kpts[0, :2] if kpts.shape[0] > 0 else None,
            #"pad_l=", pad_l,
            #"pad_t=", pad_t,
            #"r=", r,
            #"w0=", w0,
            #"h0=", h0,
            #flush=True
        #)

        # ---- UNPAD KEYPOINTS ----
        kpts[..., 0] = (kpts[..., 0] - pad_l) / r
        kpts[..., 1] = (kpts[..., 1] - pad_t) / r

        # ---- CLIP KEYPOINTS (FIX) ----
        kpts[..., 0] = np.clip(kpts[..., 0], 0, w0)
        kpts[..., 1] = np.clip(kpts[..., 1], 0, h0)

        return boxes, scores, cls_ids, kpts

    # -------------------- Inference (Multi-stream) --------------------
    async def infer_batch(
        self,
        frames: List[np.ndarray],
        stream_id: str
    ) -> Dict[str, Any]:

        #print(">>> ENTER infer_batch", flush=True)

        if not frames:
            return {
                "results": [],
                "fps": 0.0,
                "batch_size": 0,
                "stream_id": stream_id
            }

        if self.client is None:
            await self.setup()

        #print(">>> AFTER setup guard", flush=True)

        imgs, metas = self._preprocess_batch(frames)

        #print(">>> AFTER imgs, metas = self._preprocess_batch(frames)", flush=True)

        start = time.perf_counter()

        inp = grpcclient.InferInput(self._input_name, imgs.shape, self._dtype)

        #print(">>> AFTER inp = grpcclient.InferInput(self._input_name, imgs.shape, self._dtype)", flush=True)

        inp.set_data_from_numpy(imgs)

        #print(">>> AFTER inp.set_data_from_numpy(imgs)", flush=True)

        out = grpcclient.InferRequestedOutput(self._output_name)

        #print(">>> AFTER out = grpcclient.InferRequestedOutput(self._output_name)", flush=True)


        try:
            #print(">>> BEFORE TRITON INFER", flush=True)

            res = await asyncio.wait_for(
                self.client.infer(model_name=self.model_name, inputs=[inp], outputs=[out], model_version="1"),
                timeout=5.0
            )

            #print(">>> AFTER TRITON INFER", flush=True)

            preds = res.as_numpy(self._output_name)

            #print(
                #"[POSE AFTER INFER]",
                #"preds_type=", type(preds),
                #"preds_shape=", None if preds is None else preds.shape,
                #flush=True
            #)

            if preds is None:
                #print(
                    #"[POSE FATAL] Triton returned None output",
                    #"model=", self.model_name,
                    #"output_name=", self._output_name,
                    #flush=True
                #)
                return {
                    "results": [None] * len(frames),
                    "fps": 0.0,
                    "batch_size": len(frames),
                    "stream_id": stream_id
                }


            if preds.ndim == 2:
                preds = preds[None, ...]  # (N, D) → (1, N, D)

            #print(
                #"[POSE AFTER INFER]",
                #"preds_type=", type(preds),
                #"preds_shape=", None if preds is None else preds.shape,
                #flush=True
            #)

        except asyncio.TimeoutError:
            raise RuntimeError("YOLO Pose Triton inference timeout")

        except asyncio.CancelledError:
            raise


        results = []
        for i in range(len(frames)):
            out_i = self._postprocess_single(preds[i], metas[i])
            if out_i is None:

                #print("[POSE RESULT] out_i=None", flush=True)

                results.append(None)
            else:

                #print("[POSE DEBUG] boxes:", len(out_i[0]), flush=True)

                results.append({
                    "boxes": out_i[0].tolist(),
                    "scores": out_i[1].tolist(),
                    "classes": out_i[2].tolist(),
                    "keypoints": out_i[3].tolist(),
                })

        fps = len(frames) / max(1e-6, (time.perf_counter() - start))
        prev = self._fps_ema.get(stream_id)
        self._fps_ema[stream_id] = fps if prev is None else (
            self._fps_alpha * fps + (1 - self._fps_alpha) * prev
        )

        return {
            "results": results,
            "fps": self._fps_ema[stream_id],
            "batch_size": len(frames),
            "stream_id": stream_id,
        }
