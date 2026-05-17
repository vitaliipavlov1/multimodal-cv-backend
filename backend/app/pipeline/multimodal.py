"""
Real-time multimodal inference pipeline.

Runs YOLO26 detection, YOLO26 pose, PaddleOCR, and Farneback optical flow
concurrently per frame using asyncio.gather. Stream-only — no batching.
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Dict, Optional

import numpy as np

from app.inference.motion import FarnebackMotion
from app.inference.ocr import PaddleOCRInferenceModule
from app.inference.yolo import YOLO26DetectionModule
from app.inference.yolo_pose import YOLO26PoseModule

logger = logging.getLogger(__name__)


def _pack_flow(flow: Optional[np.ndarray], step: int = 16) -> Optional[Dict]:
    """Downsample optical flow field for JSON transport."""
    if flow is None:
        return None
    h, w = flow.shape[:2]
    return {
        "fx":     flow[::step, ::step, 0].tolist(),
        "fy":     flow[::step, ::step, 1].tolist(),
        "step":   step,
        "orig_w": w,
        "orig_h": h,
    }


async def run_inference_all(
    frames_provider: Optional[Callable[[], Any]] = None,
    use_ocr:         bool = False,
    use_yolo:        bool = False,
    use_yolo_pose:   bool = False,
    use_motion:      bool = False,
    yolo_model_name: str  = "yolo",
    pose_model_name: str  = "yolo_pose",
    yolo_triton_url: str  = "triton:8001",
    stream_id:       str  = "default",
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async generator that yields inference results for each frame.

    All enabled models run concurrently via asyncio.gather.
    Frame queue polling uses a tight sleep (5ms) tuned for YOLO26 ~3.5ms latency.
    """
    logger.info(
        "Pipeline started | yolo=%s  pose=%s  ocr=%s  motion=%s  stream=%s",
        use_yolo, use_yolo_pose, use_ocr, use_motion, stream_id,
    )

    detection_module: Optional[YOLO26DetectionModule] = None
    pose_module:      Optional[YOLO26PoseModule]      = None
    ocr_module:       Optional[PaddleOCRInferenceModule] = None
    motion:           Optional[FarnebackMotion]       = None

    if use_yolo:
        detection_module = YOLO26DetectionModule(
            model_name=yolo_model_name, url=yolo_triton_url
        )
        await detection_module.setup()

    if use_yolo_pose:
        pose_module = YOLO26PoseModule(
            model_name=pose_model_name, url=yolo_triton_url
        )
        await pose_module.setup()

    if use_ocr:
        ocr_module = PaddleOCRInferenceModule()
        await ocr_module.setup()

    if use_motion:
        motion = FarnebackMotion()

    frame_idx = 0

    try:
        while True:
            # Fetch next frame
            if frames_provider is not None:
                try:
                    frame = await frames_provider()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await asyncio.sleep(0.001)
                    continue

                if frame is None:
                    # Tight poll — YOLO26 latency ~3.5ms, avoid adding delay
                    await asyncio.sleep(0.005)
                    continue

            # Optical flow is synchronous CPU — compute before dispatching async tasks
            flow = motion.step(frame) if motion else None

            # Dispatch all inference tasks concurrently
            tasks: Dict[str, asyncio.Task] = {}
            if detection_module:
                tasks["yolo"] = asyncio.create_task(
                    detection_module.infer_batch([frame], stream_id=stream_id)
                )
            if pose_module:
                tasks["pose"] = asyncio.create_task(
                    pose_module.infer_batch([frame], stream_id=stream_id)
                )
            if ocr_module:
                tasks["ocr"] = asyncio.create_task(
                    ocr_module.infer_batch([frame])
                )

            results: Dict[str, Any] = {}
            if tasks:
                done = await asyncio.gather(*tasks.values(), return_exceptions=True)
                results = dict(zip(tasks.keys(), done))

            # Unpack results — gracefully handle exceptions from individual models
            yolo_result = _extract_first(results.get("yolo"))
            pose_result = _extract_first(results.get("pose"))
            ocr_result  = _extract_ocr(results.get("ocr"))

            yield {
                "batch": [{
                    "frame_idx": frame_idx,
                    "yolo":      yolo_result,
                    "pose":      pose_result,
                    "ocr":       ocr_result,
                    "flow":      _pack_flow(flow),
                }]
            }

            frame_idx += 1
            await asyncio.sleep(0)  # yield control to event loop

    finally:
        for module in (detection_module, pose_module, ocr_module):
            if module and hasattr(module, "close"):
                await module.close()


def _extract_first(res: Any) -> Optional[Dict]:
    """Extract first result from an infer_batch response dict."""
    if not isinstance(res, dict):
        return None
    results = res.get("results")
    if results and results[0] is not None:
        return results[0]
    return None


def _extract_ocr(res: Any) -> Optional[Any]:
    """Extract OCR result, handling both dict and list response formats."""
    if isinstance(res, Exception) or res is None:
        return None
    if isinstance(res, dict):
        items = res.get("results", [None])
        return items[0] if items else None
    if isinstance(res, list):
        return res[0] if res else None
    return None
