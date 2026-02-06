# NOTE:
# This pipeline is intentionally stream-only.


import asyncio
import numpy as np
from typing import Optional, Dict, Any, AsyncIterator, List, Callable, Awaitable

# ------------------------- Modules -------------------------
from .yolo_triton_module import YOLOv8lTritonModule
from .yolo_pose_triton_module import YOLOv8lPoseTritonModule
from .ocr_triton_module import PaddleOCRInferenceModule
from .motion_farneback import FarnebackMotion


def pack_flow(flow, step=16):
    if flow is None:
        return None

    h, w = flow.shape[:2]

    fx = flow[::step, ::step, 0].tolist()
    fy = flow[::step, ::step, 1].tolist()

    return {
        "fx": fx,
        "fy": fy,
        "step": step,
        "orig_w": w,
        "orig_h": h,
    }



# ------------------------- Async -------------------------

async def run_inference_all(
    frames_provider: Optional[Callable[[], Awaitable[Optional[np.ndarray]]]] = None,
    use_ocr: bool = False,
    use_yolo: bool = False,
    use_yolo_pose: bool = False,
    use_motion: bool = False,
    yolo_model_name: str = "yolo",
    pose_model_name: str = "yolo_pose",
    yolo_triton_url: str = "triton:8001",
    stream_id: str = "default",
) -> AsyncIterator[Dict[str, Any]]:


    print(
        "[PIPELINE] run_inference_all STARTED",
        "use_yolo=", use_yolo,
        "use_pose=", use_yolo_pose,
        "use_ocr=", use_ocr,
        flush=True
    )


    # ---- setup modules ----

    if use_yolo:
        yolo_module = YOLOv8lTritonModule(
            model_name=yolo_model_name,
            url=yolo_triton_url
        )

        await yolo_module.setup()

    if use_yolo_pose:
        pose_module = YOLOv8lPoseTritonModule(
            model_name=pose_model_name,
            url=yolo_triton_url
        )
        await pose_module.setup()

    if use_ocr:
        ocr_module = PaddleOCRInferenceModule()
        await ocr_module.setup()

    if use_motion:
        motion = FarnebackMotion(stride=1)


    frame_idx = 0


    try:
        while True:

            if frames_provider is not None:
                try:
                    frame = await frames_provider()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await asyncio.sleep(0.001)
                    continue

                if frame is None:
                    await asyncio.sleep(0.005)
                    continue


            frame_flow = None
            ocr_results = [None]
            yolo_results = [None]
            pose_results = [None]

            #  with PIPELINE_INFER_TIME.time():

            if use_motion:
                frame_flow = motion.step(frame)

            if use_yolo:
                yolo_out = await yolo_module.infer_batch([frame], stream_id=stream_id)
                yolo_results[0] = yolo_out["results"][0]

            if use_yolo_pose:
                pose_out = await pose_module.infer_batch([frame], stream_id=stream_id)
                pose_results[0] = pose_out["results"][0]

            if use_ocr:
                ocr_results[0] = (await ocr_module.infer_batch([frame]))[0]

            yield {
                "batch": [{
                    "frame_idx": frame_idx,
                    "yolo": yolo_results[0],
                    "pose": pose_results[0],
                    "ocr": ocr_results[0],
                    "flow": pack_flow(frame_flow) if use_motion else None,
                }]
            }

            frame_idx += 1
            await asyncio.sleep(0)

    
    finally:
        for m in [
            locals().get("yolo_module"),
            locals().get("pose_module"),
             locals().get("ocr_module"),
        ]:
            if m and hasattr(m, "close"):
                await m.close()
