from typing import Dict, AsyncIterator, Optional, List, Any, Callable
import asyncio
import json
import cv2
import time
import logging
import numpy as np
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessageChunk

from app.pipeline.multimodal import run_inference_all
from app.inference.triton_client import close_triton_client

# ---------------- PROMETHEUS ----------------
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.monitoring.metrics import *

logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)
logger = logging.getLogger("inference-api")

DEBUG_MODE = False

# ============================================================
# GLOBAL STATE & CONFIGB
# ============================================================

# ---- inference / gpu ----
inference_semaphore = asyncio.Semaphore(3)

# ---- LLM backlog ----
MAX_LLM_BACKLOG = 200  # reasonable limit

LLM_INTERVAL_MIN = 1.0  # Shouldn't be faster
LLM_INTERVAL_MAX = 5.0  # Shouldn't be slower

BACKLOG_MAX_WAIT_SEC = 15.0  # Max backlog waiting
BACKLOG_CHECK_INTERVAL = 0.2

# ---- active sessions (REAL, NOT prometheus) ----
active_sessions_counter = 0
active_sessions_lock = asyncio.Lock()


# ---------------------------------------------------------------------
# Observation buffer (RAM + spill-to-disk)
# ---------------------------------------------------------------------

class ObservationBuffer:
    def __init__(self, session_id: str, ram_max: int = 100, max_pending: int = 500):
        self.session_id = session_id
        self.queue = asyncio.Queue(maxsize=ram_max)
        self.file = Path(f"llm_backlog_{session_id}.jsonl")
        self.file_lock = asyncio.Lock()
        self.max_pending = max_pending
        self.file_count = 0

    async def pending_count(self) -> int:
        return self.queue.qsize() + self.file_count

    async def put(self, summary: Dict[str, Any]):
        item = {
            "ts": time.time(),
            "session_id": self.session_id,
            "summary": summary,
        }

        # --- BACKPRESSURE: wait if the backlog is full ---
        if await self.pending_count() >= self.max_pending:
            LLM_BACKLOG_SIZE.labels(session_id=self.session_id).set(self.max_pending)
            return

        try:
            self.queue.put_nowait(item)

            LLM_BACKLOG_SIZE.labels(
                session_id=self.session_id
            ).set(self.queue.qsize())

            return


        except asyncio.QueueFull:
            async with self.file_lock:
                with self.file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.file_count += 1

        LLM_BACKLOG_SIZE.labels(
            session_id=self.session_id
        ).set(self.queue.qsize() + self.file_count)

    async def get(self) -> Optional[Dict[str, Any]]:
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        async with self.file_lock:
            if not self.file.exists():
                return None

            with self.file.open("r+", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    return None

                first = lines[0]
                f.seek(0)
                f.writelines(lines[1:])
                f.truncate()

            self.file_count -= 1

        LLM_BACKLOG_SIZE.labels(
            session_id=self.session_id
        ).set(self.queue.qsize() + self.file_count)

        return json.loads(first)

    async def requeue(self, item: Dict[str, Any]):
        async with self.file_lock:
            with self.file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.file_count += 1

        LLM_BACKLOG_SIZE.labels(
            session_id=self.session_id
        ).set(self.queue.qsize() + self.file_count)

    async def cleanup(self):

        self.file_count = 0

        LLM_BACKLOG_SIZE.labels(
            session_id=self.session_id
        ).set(0)

        if self.file.exists():
            try:
                self.file.unlink()

            except Exception:
                logger.exception(
                    "Failed to delete backlog file for session %s",
                    self.session_id
                )


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------
class InferenceParams(BaseModel):
    source: Optional[str | int] = 0
    batch_size: int = 1
    use_ocr: bool = False
    use_yolo: bool = False
    use_yolo_pose: bool = False
    use_motion: bool = False
    use_langchain: bool = False
    system_prompt: Optional[str] = "Interpret inference results."

    # <<< CONFIGURABLE HISTORY LIMIT
    history_max_messages: Optional[int] = 20


# ---------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------


def create_llm_scene_interpreter(system_prompt: str):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        streaming=True,
        request_timeout=5.0,
        max_tokens=120
    )

    system_prompt = """
    You are a real-time vision analysis assistant.

    STRICT OUTPUT RULES (MANDATORY):

    - NEVER use plural forms like "persons" or "people".
    - ALWAYS use exactly this sentence for person:
      "A person was detected at least X times."
      where X = min_events.person_events

    - ALWAYS use exactly this sentence for pose:
      "A pose was detected at least X times."
      where X = min_events.pose_events

    - ALWAYS use exactly this sentence for objects:
      "Objects were detected at least X times."
      where X = min_events.object_events
      If X == 0, say exactly:
      "No objects were detected."

    - ALWAYS use exactly this sentence for text:
      "Text was detected at least X times, text: "text1", "text2", "text3"."
      where X = min_events.ocr_events
      Use texts exactly as provided, without translation or modification.


    - Motion is binary:
      If motion_detected is true, say exactly:
      "Motion was detected."

    - DO NOT infer identity or uniqueness.
    - DO NOT change sentence structure.
    - DO NOT add extra wording.

    Output plain text only.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}\nFrames count: {frames_count}"),
    ])

    chain: Runnable = prompt | llm

    return chain


def summarize_batch(batch):
    summary = {
        "frames": len(batch),

        "min_events": {
            "person_events": 0,
            "object_events": 0,
            "pose_events": 0,
            "ocr_events": 0,
        },

        "motion_detected": False,
        "texts": [],
    }

    for f in batch:
        # -------- YOLO --------
        yolo = f.get("yolo") or {}
        classes = yolo.get("classes") or []

        for cls_id in classes:
            if cls_id == 0:
                # COCO class 0 = person
                summary["min_events"]["person_events"] += 1
            else:
                # NOT human
                summary["min_events"]["object_events"] += 1

        # -------- POSE --------
        pose = f.get("pose") or {}
        boxes = pose.get("boxes")
        if boxes:
            summary["min_events"]["pose_events"] += len(boxes)

        # -------- OCR --------
        ocr = f.get("ocr") or []
        if isinstance(ocr, list):
            for item in ocr:
                text = item.get("text") if isinstance(item, dict) else None
                if text:
                    summary["min_events"]["ocr_events"] += 1
                    summary["texts"].append(text.strip())

        # -------- MOTION --------
        if f.get("flow") is not None:
            summary["motion_detected"] = True

    summary["texts"] = summary["texts"][:10]
    return summary


async def scene_narrator_loop(llm_interpreter, ws, observation_buffer, session_id, params, ws_send_lock):
    while True:
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            # log_event(
            #     "llm_worker_stopped",
            #     session_id=session_id
            # )
            break

        if ws.client_state.name != "CONNECTED":
            break

        try:
            item = await asyncio.wait_for(observation_buffer.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        if item is None:
            await asyncio.sleep(0.1)
            continue


        buffer = []

        try:

            def build_summary_text(summary: dict) -> str:
                parts = []

                if summary.get("motion_detected"):
                    parts.append("Motion was detected.")

                pe = summary["min_events"]["person_events"]
                if pe > 0:
                    parts.append(f"A person was detected at least {pe} times.")

                oe = summary["min_events"]["object_events"]
                if oe > 0:
                    parts.append(f"Objects were detected at least {oe} times.")
                else:
                    parts.append("No objects were detected.")

                po = summary["min_events"]["pose_events"]
                if po > 0:
                    parts.append(f"A pose was detected at least {po} times.")

                ocr = summary["min_events"].get("ocr_events", 0)
                texts = summary.get("texts") or []
                if ocr > 0 and texts:
                    unique_texts = list(dict.fromkeys(texts))[:10]  # remove duplicates, limit
                    quoted = ", ".join(f"\"{t}\"" for t in unique_texts)
                    parts.append(
                        f"Text was detected at least {ocr} times, text: {quoted}."
                    )

                return " ".join(parts)

            async for chunk in llm_interpreter.astream(
                    {
                        "input": build_summary_text(item["summary"]),
                        "frames_count": item["summary"]["frames"]
                    }
            ):

                if isinstance(chunk, AIMessageChunk):
                    if chunk.content:
                        buffer.append(chunk.content)

            final_text = "".join(buffer).strip()

            logger.info(
                "[LLM] FINAL TEXT session_id=%s text=%r",
                session_id,
                final_text
            )

            if final_text:
                try:
                    async with ws_send_lock:
                        await ws.send_json({
                            "type": "llm_summary",
                            "data": final_text,
                        })
                except RuntimeError:
                    break


        except Exception as e:
            logger.exception(
                "[LLM] STREAM FAILED session_id=%s",
                session_id
            )
            await asyncio.sleep(1.0)
            continue


# ---------------------------------------------------------------------
# Inference tool
# ---------------------------------------------------------------------
async def inference_tool(
        params: Dict,
        llm_interpreter: Optional[Runnable],
        session_id: str,
        observation_buffer: ObservationBuffer,
        frames_provider: Callable[[], Optional[np.ndarray]],
) -> AsyncIterator[Dict]:
    llm_batch: List[Dict[str, Any]] = []
    llm_batch_size = 10

    frame_idx = 0

    FPS_INTERVAL = 0.5  # seconds

    fps_window_start = time.time()
    fps_window_frames = 0
    current_fps = 0.0

    last_llm_ts = time.time()
    current_llm_interval = LLM_INTERVAL_MIN

    INFERENCE_REQUESTS.labels(status="started").inc()

    with INFERENCE_DURATION.time():

        async with inference_semaphore:

            GPU_SEMAPHORE_IN_USE.inc()
            try:
                async for result in run_inference_all(
                        batch_size=params["batch_size"],
                        use_ocr=params["use_ocr"],
                        use_yolo=params["use_yolo"],
                        use_yolo_pose=params["use_yolo_pose"],
                        use_motion=params["use_motion"],
                        frames_provider=frames_provider,
                        stream_id=session_id,
                ):

                    batch = result.get("batch", [])
                    batch_size = len(batch)

                    cached_backlog_size = await observation_buffer.pending_count()

                    if batch_size > 0:
                        fps_window_frames += batch_size

                    now = time.time()

                    if now - fps_window_start >= FPS_INTERVAL:
                        current_fps = fps_window_frames / (now - fps_window_start)
                        PIPELINE_FPS.set(current_fps)

                        fps_window_start = now
                        fps_window_frames = 0

                    if batch:
                        SESSION_FRAMES.labels(session_id=session_id).inc(len(batch))

                    for frame_data in batch:

                        frame_data["frame_idx"] = frame_idx
                        frame_idx += 1

                        # -------- LLM LOGIC (Not blocking inference) --------
                        if llm_interpreter is not None and params.get("use_langchain"):

                            backlog_size = cached_backlog_size

                            # 🔵 adaptive LLM interval
                            if backlog_size > MAX_LLM_BACKLOG * 0.7:
                                current_llm_interval = min(
                                    current_llm_interval * 1.5,
                                    LLM_INTERVAL_MAX,
                                )
                            elif backlog_size < MAX_LLM_BACKLOG * 0.3:
                                current_llm_interval = max(
                                    current_llm_interval * 0.8,
                                    LLM_INTERVAL_MIN,
                                )

                            if time.time() - last_llm_ts >= current_llm_interval:
                                last_llm_ts = time.time()
                                llm_batch.append(frame_data)

                                # summary
                                if len(llm_batch) >= llm_batch_size:
                                    summary = summarize_batch(llm_batch)
                                    summary["frame_idx_range"] = {
                                        "from": llm_batch[0].get("frame_idx"),
                                        "to": llm_batch[-1].get("frame_idx"),
                                    }

                                    if DEBUG_MODE:
                                        pass

                                    wait_started = time.time()
                                    if await observation_buffer.pending_count() >= MAX_LLM_BACKLOG:
                                        if (
                                                time.time() - wait_started
                                                > BACKLOG_MAX_WAIT_SEC
                                        ):
                                            logger.error(
                                                f"LLM backlog stuck for session {session_id}"
                                            )
                                            break

                                        await asyncio.sleep(BACKLOG_CHECK_INTERVAL)
                                        continue

                                    await observation_buffer.put(summary)
                                    cached_backlog_size += 1

                                    llm_batch.clear()
                                    last_llm_ts = time.time()

                        # -------- INFERENCE RESULT (ALWAYS) --------
                        yield {
                            "type": "inference",
                            "data": {
                                "frame_idx": frame_data.get("frame_idx"),
                                "yolo": frame_data.get("yolo"),
                                "pose": frame_data.get("pose"),
                                "ocr": frame_data.get("ocr"),
                                "flow": frame_data.get("flow"),
                                "fps": current_fps,
                            },
                        }

            except asyncio.CancelledError:
                logger.info("Client disconnected, cancelling inference")
                raise

            finally:

                GPU_SEMAPHORE_IN_USE.dec()

                INFERENCE_REQUESTS.labels(status="finished").inc()

                # Sending the last incomplete batch:
                if llm_interpreter and params.get("use_langchain") and llm_batch:
                    summary = summarize_batch(llm_batch)
                    summary["frame_idx_range"] = {
                        "from": llm_batch[0].get("frame_idx"),
                        "to": llm_batch[-1].get("frame_idx"),
                    }

                    wait_started = time.time()

                    while await observation_buffer.pending_count() >= MAX_LLM_BACKLOG:
                        if time.time() - wait_started > BACKLOG_MAX_WAIT_SEC:
                            logger.error(
                                f"LLM backlog stuck on final batch for session {session_id}"
                            )
                            break

                        await asyncio.sleep(BACKLOG_CHECK_INTERVAL)

                    await observation_buffer.put(summary)
                    llm_batch.clear()

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------


app = FastAPI(title="Multimodal Triton Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_triton():
    await close_triton_client()


@app.websocket("/ws/inference")
async def ws_inference(ws: WebSocket):
    global active_sessions_counter

    origin = ws.headers.get("origin")
    logger.info("WS CONNECT ATTEMPT, Origin=%s", origin)

    await ws.accept()

    inference_started = False

    ws_send_lock = asyncio.Lock()

    async def ws_heartbeat():
        try:
            while True:
                await asyncio.sleep(5)
                async with ws_send_lock:
                    await ws.send_json({"type": "ping"})
        except Exception:
            pass

    heartbeat_task = asyncio.create_task(ws_heartbeat())

    # --- receive init params from client ---
    init_msg = await ws.receive_json()

    if init_msg.get("type") != "init":
        await ws.close(code=1003)
        return

    ACTIVE_SESSIONS.inc()

    params = InferenceParams(**init_msg.get("params", {}))

    async with active_sessions_lock:
        active_sessions_counter += 1
        current_active = active_sessions_counter

    async with ws_send_lock:
        await ws.send_json({
            "type": "queue_status",
            "state": "waiting",
            "active_sessions": current_active,
        })

    llm_task: Optional[asyncio.Task] = None

    # --- session / params ---
    session_id = uuid4().hex
    observation_buffer = ObservationBuffer(session_id)

    llm_interpreter: Optional[Runnable] = None
    if params.use_langchain:
        llm_interpreter = create_llm_scene_interpreter(
            system_prompt=params.system_prompt,
        )

    if params.use_langchain and llm_interpreter is not None:
        llm_task = asyncio.create_task(
            scene_narrator_loop(
                llm_interpreter=llm_interpreter,
                ws=ws,
                observation_buffer=observation_buffer,
                session_id=session_id,
                params=params,
                ws_send_lock=ws_send_lock,
            )
        )

    frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=3)

    inference_task: asyncio.Task | None = None

    async def frames_provider():
        try:
            return await asyncio.wait_for(frame_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def inference_loop():

        # ---- ENTER GPU QUEUE ----
        try:
            async with ws_send_lock:
                await ws.send_json({
                    "type": "queue_status",
                    "state": "running",
                    "active_sessions": active_sessions_counter,
                })
        except Exception:
            return

        try:
            async for result in inference_tool(
                    params=params.dict(),
                    llm_interpreter=llm_interpreter,
                    session_id=session_id,
                    observation_buffer=observation_buffer,
                    frames_provider=frames_provider
            ):
                async with ws_send_lock:
                    try:
                        await ws.send_json(result)
                    except RuntimeError:
                        return



        except asyncio.CancelledError:
            # NORMAL client termination
            raise


        except Exception:
            # real error
            logger.exception("Inference loop crashed")
            return

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                break

            # ===== 1️⃣ BINARY FRAME =====
            if msg["type"] == "websocket.receive" and "bytes" in msg:
                jpg_bytes = msg["bytes"]

                def decode():
                    arr = np.frombuffer(jpg_bytes, np.uint8)
                    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

                loop = asyncio.get_running_loop()

                img = await loop.run_in_executor(None, decode)

                if img is None:
                    continue

                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                await frame_queue.put(img)
                continue

            # ===== 2️⃣ JSON CONTROL MESSAGES =====
            if "text" in msg:
                data = json.loads(msg["text"])

                if data["type"] == "start" and not inference_started:
                    inference_started = True

                    async def inference_entry():
                        nonlocal inference_started
                        try:
                            await inference_loop()
                        finally:
                            inference_started = False

                    inference_task = asyncio.create_task(inference_entry())


                elif data["type"] == "stop":
                    if inference_task:
                        inference_task.cancel()
                        inference_task = None


    except WebSocketDisconnect:
        if inference_task:
            inference_task.cancel()

    finally:
        async with active_sessions_lock:
            active_sessions_counter = max(0, active_sessions_counter - 1)
            current_active = active_sessions_counter

        try:
            async with ws_send_lock:
                await ws.send_json({
                    "type": "queue_status",
                    "state": "finished",
                    "active_sessions": current_active,
                })
        except Exception:
            pass

        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except:
                pass

        # 1️⃣ Stop inference

        if inference_task:
            inference_task.cancel()
            try:
                await inference_task
            except asyncio.CancelledError:
                pass

        await asyncio.sleep(0.1)

        # 2️⃣ Stop LLM worker
        if llm_task:
            llm_task.cancel()
            try:
                await llm_task
            except asyncio.CancelledError:
                logger.info("LLM task cancelled cleanly")
                pass

        # 3️⃣ Delete observation buffer (RAM + file)

        await observation_buffer.cleanup()

        logger.info(
            "[SESSION CLEANUP DONE] session_id=%s",
            session_id
        )

        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        ACTIVE_SESSIONS.dec()


# ---------------------------------------------------------------------
# Prometheus endpoint
# ---------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
def health():
    return {"status": "ok"}
