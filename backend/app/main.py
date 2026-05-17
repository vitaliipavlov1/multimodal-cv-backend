"""
FastAPI WebSocket server for real-time multimodal inference.

Handles:
- Binary JPEG frames from browser clients
- JSON control messages (init / start / stop)
- Concurrent inference sessions guarded by an asyncio Semaphore
- LangChain scene narration running in a background task per session
- Prometheus metrics export
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app.inference.triton_client import close_triton_client
from app.monitoring.metrics import (
    ACTIVE_SESSIONS,
    GPU_SEMAPHORE_IN_USE,
    INFERENCE_DURATION,
    INFERENCE_REQUESTS,
    LLM_BACKLOG_SIZE,
    PIPELINE_FPS,
    SESSION_FRAMES,
)
from app.pipeline.multimodal import run_inference_all

logging.basicConfig(level=logging.INFO)
for noisy in ("langchain", "langchain_core", "langchain_openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("inference-api")

# ── Shared resources ─────────────────────────────────────────────
_decode_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jpeg-decode")

# YOLO26 latency ~3.5ms — 2 concurrent GPU sessions avoids memory contention
inference_semaphore = asyncio.Semaphore(2)

# ── LLM config ───────────────────────────────────────────────────
MAX_LLM_BACKLOG        = 200
LLM_INTERVAL_MIN       = 0.5
LLM_INTERVAL_MAX       = 5.0
BACKLOG_MAX_WAIT_SEC   = 15.0
BACKLOG_CHECK_INTERVAL = 0.2

# ── Session counter ──────────────────────────────────────────────
_active_sessions       = 0
_active_sessions_lock  = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────
#  Observation buffer — RAM queue with file spill for LLM backlog
# ─────────────────────────────────────────────────────────────────

class ObservationBuffer:
    """
    Bounded async queue with overflow spill to a per-session JSONL file.
    Ensures the LLM worker always has data without blocking inference.
    """

    def __init__(self, session_id: str, ram_max: int = 100, max_pending: int = 500):
        self.session_id  = session_id
        self._queue      = asyncio.Queue(maxsize=ram_max)
        self._file       = Path(f"llm_backlog_{session_id}.jsonl")
        self._file_lock  = asyncio.Lock()
        self._max        = max_pending
        self._file_count = 0

    async def pending(self) -> int:
        return self._queue.qsize() + self._file_count

    async def put(self, summary: Dict[str, Any]) -> None:
        if await self.pending() >= self._max:
            LLM_BACKLOG_SIZE.labels(session_id=self.session_id).set(self._max)
            return

        item = {"ts": time.time(), "session_id": self.session_id, "summary": summary}
        try:
            self._queue.put_nowait(item)
            LLM_BACKLOG_SIZE.labels(session_id=self.session_id).set(self._queue.qsize())
            return
        except asyncio.QueueFull:
            pass

        async with self._file_lock:
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._file_count += 1
        LLM_BACKLOG_SIZE.labels(session_id=self.session_id).set(await self.pending())

    async def get(self) -> Optional[Dict[str, Any]]:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        async with self._file_lock:
            if not self._file.exists():
                return None
            with self._file.open("r+", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    return None
                item_str = lines[0]
                f.seek(0)
                f.writelines(lines[1:])
                f.truncate()
            self._file_count -= 1

        LLM_BACKLOG_SIZE.labels(session_id=self.session_id).set(await self.pending())
        return json.loads(item_str)

    async def cleanup(self) -> None:
        self._file_count = 0
        LLM_BACKLOG_SIZE.labels(session_id=self.session_id).set(0)
        if self._file.exists():
            try:
                self._file.unlink()
            except OSError:
                logger.exception("Failed to delete backlog file for session %s", self.session_id)


# ─────────────────────────────────────────────────────────────────
#  Request schema
# ─────────────────────────────────────────────────────────────────

class InferenceParams(BaseModel):
    batch_size:    int  = 1
    use_ocr:       bool = False
    use_yolo:      bool = False
    use_yolo_pose: bool = False
    use_motion:    bool = False
    use_langchain: bool = False
    system_prompt: Optional[str] = "Interpret inference results."


# ─────────────────────────────────────────────────────────────────
#  LLM scene narration
# ─────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """
You are a real-time vision analysis assistant.

STRICT OUTPUT RULES (MANDATORY):
- NEVER use plural forms like "persons" or "people".
- ALWAYS use exactly this sentence for person:
  "A person was detected at least X times."
- ALWAYS use exactly this sentence for pose:
  "A pose was detected at least X times."
- ALWAYS use exactly this sentence for objects:
  "Objects were detected at least X times."
  If X == 0: "No objects were detected."
- ALWAYS use exactly this sentence for text:
  "Text was detected at least X times, text: "t1", "t2"."
  Use texts verbatim — no translation or modification.
- Motion is binary. If detected: "Motion was detected."
- DO NOT infer identity, uniqueness, or add extra wording.

Output plain text only.
"""


def _create_llm_chain(system_prompt: Optional[str] = None) -> Runnable:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        streaming=True,
        request_timeout=5.0,
        max_tokens=120,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or _LLM_SYSTEM_PROMPT),
        ("human", "{input}\nFrames count: {frames_count}"),
    ])
    return prompt | llm


def _summarise(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate detection counts across a batch of frames."""
    summary: Dict[str, Any] = {
        "frames":     len(batch),
        "min_events": {"person_events": 0, "object_events": 0,
                       "pose_events": 0, "ocr_events": 0},
        "motion_detected": False,
        "texts": [],
    }
    for frame in batch:
        for cls_id in (frame.get("yolo") or {}).get("classes") or []:
            key = "person_events" if cls_id == 0 else "object_events"
            summary["min_events"][key] += 1

        boxes = (frame.get("pose") or {}).get("boxes")
        if boxes:
            summary["min_events"]["pose_events"] += len(boxes)

        for item in (frame.get("ocr") or []):
            text = (item.get("text") if isinstance(item, dict) else None)
            if text:
                summary["min_events"]["ocr_events"] += 1
                summary["texts"].append(text.strip())

        if frame.get("flow") is not None:
            summary["motion_detected"] = True

    summary["texts"] = summary["texts"][:10]
    return summary


def _build_prompt(summary: Dict) -> str:
    parts: List[str] = []
    ev = summary["min_events"]

    if summary.get("motion_detected"):
        parts.append("Motion was detected.")
    if ev["person_events"] > 0:
        parts.append(f"A person was detected at least {ev['person_events']} times.")
    parts.append(
        f"Objects were detected at least {ev['object_events']} times."
        if ev["object_events"] > 0 else "No objects were detected."
    )
    if ev["pose_events"] > 0:
        parts.append(f"A pose was detected at least {ev['pose_events']} times.")
    if ev["ocr_events"] > 0 and summary.get("texts"):
        unique = list(dict.fromkeys(summary["texts"]))[:10]
        quoted = ", ".join(f'"{t}"' for t in unique)
        parts.append(f"Text was detected at least {ev['ocr_events']} times, text: {quoted}.")

    return " ".join(parts)


async def _narrator_loop(
    chain:    Runnable,
    ws:       WebSocket,
    buf:      ObservationBuffer,
    sid:      str,
    lock:     asyncio.Lock,
) -> None:
    """Background task: pull observations from buffer, stream LLM response to client."""
    while True:
        await asyncio.sleep(0)
        if ws.client_state.name != "CONNECTED":
            break

        try:
            item = await asyncio.wait_for(buf.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        if item is None:
            await asyncio.sleep(0.1)
            continue

        tokens: List[str] = []
        try:
            async for chunk in chain.astream({
                "input":        _build_prompt(item["summary"]),
                "frames_count": item["summary"]["frames"],
            }):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    tokens.append(chunk.content)

            text = "".join(tokens).strip()
            logger.info("[LLM] session=%s  text=%r", sid, text)
            if text:
                async with lock:
                    await ws.send_json({"type": "llm_summary", "data": text})
        except RuntimeError:
            break
        except Exception:
            logger.exception("[LLM] stream failed  session=%s", sid)
            await asyncio.sleep(1.0)


# ─────────────────────────────────────────────────────────────────
#  Inference pipeline wrapper
# ─────────────────────────────────────────────────────────────────

async def _run_inference(
    params:    Dict[str, Any],
    chain:     Optional[Runnable],
    session_id: str,
    buf:       ObservationBuffer,
    provider:  Callable,
) -> AsyncIterator[Dict]:
    llm_batch:  List[Dict] = []
    frame_idx   = 0
    fps_start   = time.time()
    fps_frames  = 0
    current_fps = 0.0
    last_llm_ts = time.time()
    llm_interval = LLM_INTERVAL_MIN
    LLM_BATCH_SIZE = 10
    FPS_WINDOW = 0.5

    INFERENCE_REQUESTS.labels(status="started").inc()

    with INFERENCE_DURATION.time():
        async with inference_semaphore:
            GPU_SEMAPHORE_IN_USE.inc()
            try:
                async for result in run_inference_all(
                    use_ocr=params["use_ocr"],
                    use_yolo=params["use_yolo"],
                    use_yolo_pose=params["use_yolo_pose"],
                    use_motion=params["use_motion"],
                    frames_provider=provider,
                    stream_id=session_id,
                ):
                    batch = result.get("batch", [])

                    now = time.time()
                    fps_frames += len(batch)
                    if now - fps_start >= FPS_WINDOW:
                        current_fps = fps_frames / (now - fps_start)
                        PIPELINE_FPS.set(current_fps)
                        fps_start, fps_frames = now, 0

                    if batch:
                        SESSION_FRAMES.labels(session_id=session_id).inc(len(batch))

                    for frame_data in batch:
                        frame_data["frame_idx"] = frame_idx
                        frame_idx += 1

                        # LLM batching — adaptive interval based on backlog pressure
                        if chain and params.get("use_langchain"):
                            backlog = await buf.pending()
                            if backlog > MAX_LLM_BACKLOG * 0.7:
                                llm_interval = min(llm_interval * 1.5, LLM_INTERVAL_MAX)
                            elif backlog < MAX_LLM_BACKLOG * 0.3:
                                llm_interval = max(llm_interval * 0.8, LLM_INTERVAL_MIN)

                            if time.time() - last_llm_ts >= llm_interval:
                                last_llm_ts = time.time()
                                llm_batch.append(frame_data)
                                if len(llm_batch) >= LLM_BATCH_SIZE:
                                    summary = _summarise(llm_batch)
                                    summary["frame_idx_range"] = {
                                        "from": llm_batch[0].get("frame_idx"),
                                        "to":   llm_batch[-1].get("frame_idx"),
                                    }
                                    if await buf.pending() < MAX_LLM_BACKLOG:
                                        await buf.put(summary)
                                    llm_batch.clear()
                                    last_llm_ts = time.time()

                        yield {
                            "type": "inference",
                            "data": {
                                "frame_idx": frame_data.get("frame_idx"),
                                "yolo":      frame_data.get("yolo"),
                                "pose":      frame_data.get("pose"),
                                "ocr":       frame_data.get("ocr"),
                                "flow":      frame_data.get("flow"),
                                "fps":       current_fps,
                            },
                        }

            except asyncio.CancelledError:
                logger.info("Inference cancelled  session=%s", session_id)
                raise
            finally:
                GPU_SEMAPHORE_IN_USE.dec()
                INFERENCE_REQUESTS.labels(status="finished").inc()

                # Flush remaining LLM batch
                if chain and params.get("use_langchain") and llm_batch:
                    summary = _summarise(llm_batch)
                    summary["frame_idx_range"] = {
                        "from": llm_batch[0].get("frame_idx"),
                        "to":   llm_batch[-1].get("frame_idx"),
                    }
                    wait = time.time()
                    while await buf.pending() >= MAX_LLM_BACKLOG:
                        if time.time() - wait > BACKLOG_MAX_WAIT_SEC:
                            break
                        await asyncio.sleep(BACKLOG_CHECK_INTERVAL)
                    await buf.put(summary)


# ─────────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    yield
    await close_triton_client()
    _decode_executor.shutdown(wait=False)


app = FastAPI(title="Multimodal Triton Inference API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/inference")
async def ws_inference(ws: WebSocket) -> None:
    global _active_sessions

    logger.info("WS connect  origin=%s", ws.headers.get("origin"))
    await ws.accept()

    ws_lock          = asyncio.Lock()
    inference_active = False

    # Heartbeat keeps the connection alive through proxies / load balancers
    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(5)
                async with ws_lock:
                    await ws.send_json({"type": "ping"})
        except Exception:
            pass

    heartbeat = asyncio.create_task(_heartbeat())

    raw = await ws.receive()
    if "text" not in raw:
        await ws.close(code=1003)
        heartbeat.cancel()
        return
    init_msg = json.loads(raw["text"])
    if init_msg.get("type") != "init":
        await ws.close(code=1003)
        heartbeat.cancel()
        return

    ACTIVE_SESSIONS.inc()
    params = InferenceParams(**init_msg.get("params", {}))

    async with _active_sessions_lock:
        _active_sessions += 1
        active_count = _active_sessions

    async with ws_lock:
        await ws.send_json({
            "type": "queue_status",
            "state": "waiting",
            "active_sessions": active_count,
        })

    session_id = uuid4().hex
    buf        = ObservationBuffer(session_id)
    chain:     Optional[Runnable]       = None
    llm_task:  Optional[asyncio.Task]   = None

    if params.use_langchain:
        chain    = _create_llm_chain(params.system_prompt)
        llm_task = asyncio.create_task(
            _narrator_loop(chain, ws, buf, session_id, ws_lock)
        )

    # Frame queue: maxsize=2 keeps pipeline latency minimal
    frame_queue:    asyncio.Queue = asyncio.Queue(maxsize=2)
    inference_task: Optional[asyncio.Task] = None

    async def _frames_provider():
        try:
            return await asyncio.wait_for(frame_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def _inference_loop() -> None:
        nonlocal inference_active
        try:
            async with ws_lock:
                await ws.send_json({
                    "type": "queue_status",
                    "state": "running",
                    "active_sessions": _active_sessions,
                })
        except Exception:
            return

        try:
            async for result in _run_inference(
                params.model_dump(), chain, session_id, buf, _frames_provider
            ):
                async with ws_lock:
                    try:
                        await ws.send_json(result)
                    except RuntimeError:
                        return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Inference loop crashed  session=%s", session_id)
        finally:
            inference_active = False

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                break

            # Binary JPEG frame
            if msg.get("type") == "websocket.receive" and "bytes" in msg:
                jpg = msg["bytes"]
                loop = asyncio.get_running_loop()
                img  = await loop.run_in_executor(
                    _decode_executor,
                    lambda: cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR),
                )
                if img is None:
                    continue
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await frame_queue.put(img)
                continue

            # JSON control messages
            if "text" in msg:
                data = json.loads(msg["text"])

                if data["type"] == "start" and not inference_active:
                    inference_active = True
                    inference_task   = asyncio.create_task(_inference_loop())

                elif data["type"] == "stop" and inference_task:
                    inference_task.cancel()
                    inference_task = None

    except WebSocketDisconnect:
        if inference_task:
            inference_task.cancel()

    finally:
        async with _active_sessions_lock:
            _active_sessions = max(0, _active_sessions - 1)
            active_count     = _active_sessions

        try:
            async with ws_lock:
                await ws.send_json({
                    "type": "queue_status",
                    "state": "finished",
                    "active_sessions": active_count,
                })
        except Exception:
            pass

        heartbeat.cancel()
        try:
            await heartbeat
        except (Exception, asyncio.CancelledError):
            pass

        if inference_task:
            inference_task.cancel()
            try:
                await inference_task
            except asyncio.CancelledError:
                pass

        await asyncio.sleep(0.1)

        if llm_task:
            llm_task.cancel()
            try:
                await llm_task
            except asyncio.CancelledError:
                pass

        await buf.cleanup()
        logger.info("Session closed  session_id=%s", session_id)

        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        ACTIVE_SESSIONS.dec()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
