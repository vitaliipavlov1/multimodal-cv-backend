"""
YOLO26 detection inference module with per-stream ByteTrack tracking.

Model  : YOLO26s TensorRT FP16 — NMS-free end-to-end
Input  : FP16 [B, 3, 640, 640]
Output : FP32 [B, 300, 6]  →  [x1, y1, x2, y2, conf, cls]
Tracker: Two-pass ByteTrack with lapjv optimal assignment (lapx)
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

try:
    import lap
    _HAS_LAP = True
except ImportError:
    _HAS_LAP = False
    logger.warning("lapx not available — falling back to greedy matching")


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


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between every pair. a: [N,4], b: [M,4] → [N,M]."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _assign(cost: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimal assignment via lapjv (lapx) with greedy fallback.
    Returns matched (row_indices, col_indices) where cost < threshold.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty(0, int), np.empty(0, int)

    if _HAS_LAP:
        padded = cost.copy()
        padded[padded > threshold] = threshold + 1e-4
        size = max(n, m)
        mat = np.full((size, size), threshold + 1e-4, dtype=np.float64)
        mat[:n, :m] = padded
        _, row_to_col, _ = lap.lapjv(mat, extend_cost=True)
        matched = [
            (r, row_to_col[r])
            for r in range(n)
            if row_to_col[r] < m and cost[r, row_to_col[r]] <= threshold
        ]
        if not matched:
            return np.empty(0, int), np.empty(0, int)
        rows, cols = zip(*matched)
        return np.array(rows, int), np.array(cols, int)

    # Greedy fallback — O(n*m), sufficient for n <= 20
    col_used = np.zeros(m, bool)
    rows, cols = [], []
    for r in np.argsort(cost.min(axis=1)):
        best_c, best_v = -1, threshold
        for c in range(m):
            if not col_used[c] and cost[r, c] < best_v:
                best_v, best_c = cost[r, c], c
        if best_c >= 0:
            rows.append(r)
            cols.append(best_c)
            col_used[best_c] = True
    return np.array(rows, int), np.array(cols, int)


# ─────────────────────────────────────────────────────────────────
#  ByteTrack — two-pass per-stream tracker
# ─────────────────────────────────────────────────────────────────

class _Track:
    """Internal track state. Uses __slots__ for minimal memory footprint."""
    __slots__ = ("box", "score", "cls", "track_id", "age", "hits")

    def __init__(self, box: np.ndarray, score: float, cls: int, track_id: int):
        self.box      = np.asarray(box, dtype=np.float32)
        self.score    = float(score)
        self.cls      = int(cls)
        self.track_id = int(track_id)
        self.age      = 0   # frames since last successful match
        self.hits     = 1   # total matched frames


class ByteTracker:
    """
    Lightweight ByteTrack implementation for real-time single-GPU inference.

    Pass 1: high-confidence detections ↔ all active tracks   (IoU ≥ iou_high)
    Pass 2: low-confidence  detections ↔ unmatched tracks    (IoU ≥ iou_low)
    Tracks are pruned after max_age consecutive missed frames.
    """

    def __init__(
        self,
        conf_high: float = 0.5,
        conf_low:  float = 0.1,
        iou_high:  float = 0.4,
        iou_low:   float = 0.25,
        max_age:   int   = 5,
    ):
        self.conf_high = conf_high
        self.conf_low  = conf_low
        self.iou_high  = iou_high
        self.iou_low   = iou_low
        self.max_age   = max_age
        self._next_id  = 1
        self.tracks: List[_Track] = []

    def update(
        self,
        boxes:   np.ndarray,  # [N, 4] float32  xyxy
        scores:  np.ndarray,  # [N]    float32
        classes: np.ndarray,  # [N]    int32
    ) -> Tuple[List[List[int]], List[float], List[int], List[int]]:
        """
        Update tracker with new detections.
        Returns (boxes, scores, classes, track_ids) for active detections.
        All values are native Python types — JSON-serialisable.
        """
        n = len(boxes)

        if n == 0:
            for tr in self.tracks:
                tr.age += 1
            self.tracks = [tr for tr in self.tracks if tr.age <= self.max_age]
            return [], [], [], []

        hi_mask = scores >= self.conf_high
        hi_idx  = np.where(hi_mask)[0]
        lo_idx  = np.where(~hi_mask)[0]

        track_boxes = np.array([tr.box for tr in self.tracks], dtype=np.float32) \
            if self.tracks else np.empty((0, 4), dtype=np.float32)

        matched_det: set = set()
        matched_trk: set = set()

        # Pass 1: high-confidence detections ↔ all tracks
        if len(hi_idx) > 0 and len(self.tracks) > 0:
            r_idx, c_idx = _assign(
                1.0 - _iou_matrix(boxes[hi_idx], track_boxes),
                1.0 - self.iou_high,
            )
            for r, c in zip(r_idx, c_idx):
                det_i = int(hi_idx[r])
                tr = self.tracks[c]
                tr.box = boxes[det_i]
                tr.score = float(scores[det_i])
                tr.cls = int(classes[det_i])
                tr.age = 0
                tr.hits += 1
                matched_det.add(det_i)
                matched_trk.add(c)

        # Pass 2: low-confidence detections ↔ unmatched tracks
        unmatched_trk = [i for i in range(len(self.tracks)) if i not in matched_trk]
        if len(lo_idx) > 0 and unmatched_trk:
            utm_boxes = track_boxes[unmatched_trk]
            r_idx, c_idx = _assign(
                1.0 - _iou_matrix(boxes[lo_idx], utm_boxes),
                1.0 - self.iou_low,
            )
            for r, c in zip(r_idx, c_idx):
                det_i = int(lo_idx[r])
                tr = self.tracks[unmatched_trk[c]]
                tr.box = boxes[det_i]
                tr.score = float(scores[det_i])
                tr.cls = int(classes[det_i])
                tr.age = 0
                tr.hits += 1
                matched_det.add(det_i)
                matched_trk.add(unmatched_trk[c])

        # Age unmatched tracks
        for i, tr in enumerate(self.tracks):
            if i not in matched_trk:
                tr.age += 1

        # Spawn new tracks for unmatched high-confidence detections
        for i in range(n):
            if i not in matched_det and scores[i] >= self.conf_high:
                self.tracks.append(
                    _Track(boxes[i], float(scores[i]), int(classes[i]), self._next_id)
                )
                self._next_id += 1

        # Prune dead tracks
        self.tracks = [tr for tr in self.tracks if tr.age <= self.max_age]

        # Collect output — active tracks only (age == 0)
        out_boxes:   List[List[int]] = []
        out_scores:  List[float]     = []
        out_classes: List[int]       = []
        out_ids:     List[int]       = []

        for tr in self.tracks:
            if tr.age == 0:
                out_boxes.append([int(v) for v in tr.box])
                out_scores.append(float(tr.score))
                out_classes.append(int(tr.cls))
                out_ids.append(int(tr.track_id))

        return out_boxes, out_scores, out_classes, out_ids


# ─────────────────────────────────────────────────────────────────
#  Tracker registry
# ─────────────────────────────────────────────────────────────────

class _TrackerEntry:
    __slots__ = ("tracker", "last_seen")

    def __init__(self, tracker: ByteTracker):
        self.tracker   = tracker
        self.last_seen = time.time()


# ─────────────────────────────────────────────────────────────────
#  YOLO26 Triton detection module
# ─────────────────────────────────────────────────────────────────

class YOLO26DetectionModule:
    """
    Async YOLO26s detection via Triton Inference Server + ByteTrack.

    Triton model config:
        input  → name="images"  dtype=FP16  dims=[3, 640, 640]
        output → name="output0" dtype=FP32  dims=[300, 6]
    """

    TRACKER_TTL: int = 300  # seconds before an idle tracker is pruned

    def __init__(
        self,
        model_name:      str   = "yolo",
        url:             str   = "127.0.0.1:8001",
        target_size:     int   = 640,
        conf_thres:      float = 0.2,
        max_det:         int   = 300,
        return_track_id: bool  = True,
    ):
        self.model_name      = model_name
        self.url             = url
        self.target_size     = target_size
        self.conf_thres      = conf_thres
        self.max_det         = max_det
        self.return_track_id = return_track_id

        self.client: Optional[grpcclient.InferenceServerClient] = None
        self._input_name:  Optional[str] = None
        self._output_name: Optional[str] = None
        self._dtype:       Optional[str] = None
        self._use_fp16:    bool          = False
        self._setup_lock   = asyncio.Lock()

        self._trackers:     Dict[str, _TrackerEntry] = {}
        self._trackers_lock = asyncio.Lock()

        self._fps_ema:   Optional[float] = None
        self._fps_alpha: float           = 0.1

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
                    "YOLO26 detection ready | model=%s  dtype=%s  fp16=%s",
                    self.model_name, self._dtype, self._use_fp16,
                )

    async def _get_tracker(self, stream_id: str) -> ByteTracker:
        async with self._trackers_lock:
            now = time.time()
            expired = [
                k for k, v in self._trackers.items()
                if now - v.last_seen > self.TRACKER_TTL
            ]
            for k in expired:
                del self._trackers[k]
            if stream_id not in self._trackers:
                self._trackers[stream_id] = _TrackerEntry(
                    ByteTracker(
                        conf_high=max(self.conf_thres, 0.4),
                        conf_low=self.conf_thres,
                    )
                )
            entry = self._trackers[stream_id]
            entry.last_seen = now
            return entry.tracker

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
        Decode YOLO26 NMS-free output.
        Boxes are already in xyxy format — no decode or NMS required.
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
        boxes   = pred[:, :4].astype(np.float32)
        scores  = pred[:, 4].astype(np.float32)
        cls_ids = pred[:, 5].astype(np.int32)

        mask = scores > self.conf_thres
        if not mask.any():
            return None

        boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
        boxes = _unpad_boxes(boxes, r, pad_l, pad_t)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

        if len(boxes) > self.max_det:
            idx = scores.argsort()[::-1][:self.max_det]
            boxes, scores, cls_ids = boxes[idx], scores[idx], cls_ids[idx]

        return boxes, scores, cls_ids

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
            raise RuntimeError("YOLO26 Triton inference timeout")
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
            preds = preds[None, ...]   # (N, 6) → (1, N, 6)

        tracker = await self._get_tracker(stream_id)
        results: List[Optional[Dict]] = []
        _empty = np.zeros((0, 4), np.float32)

        for i, pred in enumerate(preds):
            det = self._postprocess(pred, metas[i])

            if det is None:
                tracker.update(_empty, np.empty(0, np.float32), np.empty(0, np.int32))
                results.append(None)
                continue

            boxes, scores, cls_ids = det
            t_boxes, t_scores, t_classes, t_ids = tracker.update(boxes, scores, cls_ids)

            results.append({
                "boxes":     t_boxes,
                "classes":   t_classes,
                "scores":    t_scores,
                "track_ids": t_ids,
            } if t_boxes else None)

        elapsed = time.perf_counter() - t0
        fps = len(frames) / elapsed if elapsed > 0 else 0.0
        self._fps_ema = fps if self._fps_ema is None else (
            self._fps_alpha * fps + (1.0 - self._fps_alpha) * self._fps_ema
        )

        return {
            "results":    results,
            "fps":        self._fps_ema,
            "batch_size": len(frames),
            "stream_id":  stream_id,
        }
