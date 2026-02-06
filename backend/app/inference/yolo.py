import asyncio
from typing import List, Dict, Optional, Tuple, Any
import time
import numpy as np
import cv2
import tritonclient.grpc.aio as grpcclient
from .triton_client import get_triton_client
from ultralytics.trackers.byte_tracker import BYTETracker

# from ultralytics.trackers import BOTSORT

print("=== YOLO MODULE LOADED ===", flush=True)

# -------------------- Utility Functions --------------------

class TrackerEntry:
    def __init__(self, tracker):
        self.tracker = tracker
        self.last_seen = time.time()


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


def scale_coords_unpad(
    boxes: np.ndarray, r: float, pad_l: int, pad_t: int
) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / r
    return boxes


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thres: float = 0.45,
    max_det: int = 300
) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], int)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size and len(keep) < max_det:
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
        order = order[np.where(iou <= iou_thres)[0] + 1]

    return np.array(keep, int)

# ==========================================================
#                   Tracker (per-stream)
# ==========================================================
class ByteTrackWrapper:
    def __init__(self, conf_thres: float, track_buffer: int = 30):
        self.tracker = BYTETracker(
            track_thresh=conf_thres,
            track_buffer=track_buffer,
            match_thresh=0.8
        )

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray
    ) -> List[Dict[str, Any]]:

        if boxes is None or len(boxes) == 0:
            return []

        # ByteTrack ожидает tlbr
        online_targets = self.tracker.update(
            scores,
            boxes,
            classes
        )

        results = []
        for t in online_targets:
            results.append({
                "box": t.tlbr.astype(int).tolist(),
                "class": int(t.cls),
                "score": float(t.score),
                "track_id": int(t.track_id)
            })

        return results












class StreamTracker:
    def __init__(
        self,
        conf_thres: float,
        iou_thres: float,
        return_track_id: bool = True,
        track_buffer: int = 30
    ):
        self.return_track_id = return_track_id
        self.tracker = BOTSORT(
            track_buffer=track_buffer,
            conf_thres=conf_thres,
            match_thresh=iou_thres
        )

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray
    ) -> List[Dict[str, Any]]:
        if boxes is None or len(boxes) == 0:
            return []

        boxes = boxes.astype(np.float32, copy=False)
        scores = scores.astype(np.float32, copy=False)
        classes = classes.astype(np.int32, copy=False)

        tracks = self.tracker.update(
            boxes=boxes,
            scores=scores,
            classes=classes
        )

        results = []
        for t in tracks:
            item = {
                "box": t.tlbr.astype(int).tolist(),
                "class": int(t.cls),
                "score": float(t.score)
            }
            if self.return_track_id:
                item["track_id"] = int(t.track_id)
            results.append(item)

        return results

# ==========================================================
#                YOLOv8 Triton Async Pipeline
# ==========================================================
class YOLOv8lTritonModule:
    def __init__(
        self,
        model_name: str = "yolo",
        url: str = "127.0.0.1:8001",
        target_size: int = 640,
        warmup_runs: int = 2,
        conf_thres: float = 0.2,
        iou_thres: float = 0.4,
        max_det: int = 300,
        max_candidates: int = 1000,
        return_track_id: bool = True
    ):
        self.model_name = model_name
        self.url = url
        self.target_size = target_size
        self.warmup_runs = warmup_runs
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.max_candidates = max_candidates
        self.return_track_id = return_track_id

        self.client: Optional[grpcclient.InferenceServerClient] = None
        self._input_name = None
        self._output_name = None
        self._dtype = None
        self._use_fp16 = (self._dtype == "FP16")
        self._setup_lock = asyncio.Lock()

        self.trackers: Dict[str, TrackerEntry] = {}
        self.TRACKER_TTL = 300  # seconds
        self._trackers_lock = asyncio.Lock()

        self._fps_ema: Optional[float] = None
        self._fps_alpha = 0.1

    # -------------------- Setup --------------------
    async def setup(self):
        async with self._setup_lock:
            if self.client is None:
                self.client = await get_triton_client(self.url)

            if self._input_name is None:
                meta = await self.client.get_model_metadata(self.model_name)

                #print(
                   # "[YOLO META]",
                   # "inputs=", [i.name for i in meta.inputs],
                   # "outputs=", [o.name for o in meta.outputs],
                   # flush=True
                #)


                self._input_name = meta.inputs[0].name
                self._output_name = meta.outputs[0].name
                self._dtype = meta.inputs[0].datatype
                self._use_fp16 = False
#                self._bytetrack_template = ByteTrackWrapper(
#                    conf_thres=self.conf_thres
#                )
#                print("[YOLO SETUP] ByteTrack template READY", flush=True)
     
                               # self._dtype.upper() in ("FP16", "FLOAT16")
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
        self, frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[float, int, int, int, int]]]:

        #print(">>> ENTER preprocess", flush=True)

        batch = np.zeros(
            (len(frames), 3, self.target_size, self.target_size),
            dtype=np.float32
        )
        metas = []

        for i, f in enumerate(frames):
            h0, w0 = f.shape[:2]
            r, pad_l, pad_t, nw, nh = letterbox_resize(f, self.target_size)
            canvas = np.zeros((self.target_size, self.target_size, 3), np.uint8)
            canvas[pad_t:pad_t+nh, pad_l:pad_l+nw] = cv2.resize(f, (nw, nh))
            img = canvas[:, :, ::-1] / 255.0
            batch[i] = img.transpose(2, 0, 1)
            metas.append((r, pad_l, pad_t, w0, h0))

        #print(">>> EXIT preprocess", flush=True)

        return batch, metas

    # -------------------- Postprocess --------------------
    # Expected YOLO output format:
    # [cx, cy, w, h, obj_conf, class1, class2, ...]

    def _postprocess_single(
        self,
        pred: np.ndarray,
        meta: Tuple[float, int, int, int, int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:


        #print(
            #"[YOLO RAW PRED]",
            #"shape=", None if pred is None else pred.shape,
            #flush=True
        #)


        # ================= YOLOv8 TensorRT output normalize =================
        # Expected final shape: (N, 84) where N=8400

        if pred is None:
            return None

        # Triton may return:
        # (1, 8400, 84) OR (8400, 84) OR (84, 8400)

        if pred.ndim == 3:
            pred = pred[0]  # (1, 8400, 84) → (8400, 84)


        # Handle transposed case
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        #print("[YOLO FIX] pred shape:", pred.shape, flush=True) 

       
        #print(
            #"[YOLO RAW PRED NORMALIZED]",
            #"shape=", pred.shape,
            #flush=True
        #)
        # ====================================================================


        r, pad_l, pad_t, w0, h0 = meta

        boxes = pred[:, :4]
        cls_scores = pred[:, 4:]          # ← ВСЕ классы
        cls_ids = cls_scores.argmax(1)
        scores = cls_scores.max(1)        # ← БЕЗ objectness

        mask = scores > self.conf_thres

        #print(
            #"[YOLO CONF]",
            #"conf_thres=", self.conf_thres,
            #"candidates=", int(mask.sum()),
            #flush=True
        #)


        if not mask.any():
            return None

        boxes = scale_coords_unpad(
            xywh2xyxy(boxes[mask]),
            r, pad_l, pad_t
        )
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

        scores = scores[mask]
        cls_ids = cls_ids[mask]

        # --- performance guard before NMS ---
        # Limit number of candidates before NMS to protect CPU in worst-case scenarios
       # if self.max_candidates and scores.shape[0] > self.max_candidates:
        #    topk = scores.argsort()[-self.max_candidates:]
         #   boxes = boxes[topk]
          #  scores = scores[topk]
           # cls_ids = cls_ids[topk]

        keep = nms_numpy(boxes, scores, self.iou_thres, self.max_det)
        return boxes[keep], scores[keep], cls_ids[keep]

    # -------------------- Async Inference + Tracking --------------------
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

        # --- tracker lifecycle ---
        #  async with self._trackers_lock:
        # --- tracker lifecycle ---
#        now = time.time()

        # cleanup TTL
#        expired = [
#            sid for sid, entry in self.trackers.items()
#            if now - entry.last_seen > self.TRACKER_TTL
#        ]

#        print(">>> AFTER TTL cleanup")

#        for sid in expired:
#            del self.trackers[sid]

#        entry = self.trackers.get(stream_id)

#        print(">>> AFTER self.trackers.get(stream_d)")

#        if entry is None:

#            print(">>> AFTER f entry s None:")

#            entry = TrackerEntry(self._bytetrack_template)

#            print(">>> AFTER entry = TrackerEntry(self.bytetrack_template")

#            self.trackers[stream_id] = entry

#            print(">>> AFTER self.trackers[stream_d = entry")

#        entry.last_seen = now

        #print(">>> BEFORE imgs, metas = self._preprocess_batch(frames)", flush=True)

        imgs, metas = self._preprocess_batch(frames)

        #print(">>> AFTER imgs, metas = self._preprocess_batch(frames)", flush=True)


        start = time.perf_counter()

        inp = grpcclient.InferInput(self._input_name, imgs.shape, self._dtype)
        inp.set_data_from_numpy(imgs)
        out = grpcclient.InferRequestedOutput(self._output_name)


        try:

            #print(">>> BEFORE TRITON INFER", flush=True)

            res = await asyncio.wait_for(
                self.client.infer(model_name=self.model_name, inputs=[inp], outputs=[out], model_version="1"),
                timeout=5.0
            )

            #print(">>> AFTER TRITON INFER", flush=True)

            preds = res.as_numpy(self._output_name)

            #print(
                #"[YOLO AFTER INFER]",
                #"preds_type=", type(preds),
                #"preds_shape=", None if preds is None else preds.shape,
                #flush=True
            #)


            if preds is None:
                #print(
                    #"[YOLO FATAL] Triton returned None output",
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


            # 🔴 FIX: ensure batch dimension
            if preds is not None and preds.ndim == 2:
                preds = preds[None, ...]

            #print(
                #"[YOLO TRITON OUTPUT]",
                #"type=", type(preds),
                #"shape=", None if preds is None else preds.shape,
                #flush=True
            #)


        except asyncio.TimeoutError:
            raise RuntimeError("YOLO Triton inference timeout")

        except asyncio.CancelledError:
            # ВАЖНО: не глотаем cancel, иначе зависнет GPU semaphore
            raise

        if not isinstance(preds, np.ndarray):
            #print(
                #"[YOLO ERROR] Output is not numpy array:",
                #type(preds),
                #flush=True
            #)
            return {
                "results": [None] * len(frames),
                "fps": 0.0,
                "batch_size": len(frames),
                "stream_id": stream_id
            }

 

        results = []
        for i, pred in enumerate(preds):
            out_i = self._postprocess_single(pred, metas[i])

            if out_i is not None:
                pass
                #print("[YOLO DEBUG] boxes:", len(out_i[0]), flush=True)


            if out_i is None:
                #print("[YOLO RESULT] out_i=None", flush=True)
                results.append(None)
                continue
            else:
                pass    
                #print("[YOLO RESULT] boxes=", out_i[0].shape[0], flush=True)

            boxes, scores, classes = out_i

            tracked = None        # entry.tracker.update(boxes, scores, classes)

            if tracked:
                results.append({
                    "boxes": [t["box"] for t in tracked],
                    "classes": [t["class"] for t in tracked],
                    "scores": [t["score"] for t in tracked],
                    "track_ids": [t.get("track_id") for t in tracked]
                })
            else:
                # FALLBACK — ЧИСТЫЙ YOLO
                results.append({
                    "boxes": boxes.astype(int).tolist(),
                    "classes": classes.tolist(),
                    "scores": scores.tolist(),
                    "track_ids": None
                })

        elapsed = time.perf_counter() - start
        fps = len(frames) / elapsed if elapsed > 0 else 0.0
        self._fps_ema = fps if self._fps_ema is None else (
            self._fps_alpha * fps + (1 - self._fps_alpha) * self._fps_ema
        )

        return {
            "results": results,
            "fps": self._fps_ema,
            "batch_size": len(frames),
            "stream_id": stream_id
        }
