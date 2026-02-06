# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# ---------------- API ----------------
INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["status"]
)

ACTIVE_SESSIONS = Gauge(
    "inference_active_sessions",
    "Currently active inference sessions"
)

REJECTED_SESSIONS = Counter(
    "inference_rejected_sessions_total",
    "Rejected inference sessions"
)

INFERENCE_DURATION = Histogram(
    "inference_duration_seconds",
    "End-to-end inference duration"
)

GPU_SEMAPHORE_IN_USE = Gauge(
    "gpu_semaphore_in_use",
    "GPU semaphore currently in use"
)

# ---------------- PIPELINE ----------------
PIPELINE_BATCH_SIZE = Histogram(
    "pipeline_batch_size",
    "Batch size per inference"
)

PIPELINE_FPS = Gauge(
    "pipeline_fps",
    "Pipeline FPS"
)

PIPELINE_INFER_TIME = Histogram(
    "pipeline_infer_seconds",
    "Pipeline inference time"
)

PIPELINE_MODEL_FRAMES = Counter(
    "pipeline_model_frames_total",
    "Frames processed per model",
    ["model"]
)

PIPELINE_FRAMES_TOTAL = Counter(
    "pipeline_frames_total",
    "Total processed frames"
)

SESSION_FRAMES = Counter(
    "session_frames_total",
    "Frames processed per session",
    ["session_id"]
)

LLM_BACKLOG_SIZE = Gauge(
    "llm_backlog_size",
    "LLM backlog size per session",
    ["session_id"]
)


# ---------------- MODELS ----------------
MODEL_ERRORS = Counter(
    "model_errors_total",
    "Model errors",
    ["model"]
)
