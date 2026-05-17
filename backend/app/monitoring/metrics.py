"""
Prometheus metrics for the multimodal inference API.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Request lifecycle ────────────────────────────────────────────
INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["status"],
)

ACTIVE_SESSIONS = Gauge(
    "inference_active_sessions",
    "Currently active inference sessions",
)

INFERENCE_DURATION = Histogram(
    "inference_duration_seconds",
    "End-to-end inference session duration",
)

# ── GPU / concurrency ────────────────────────────────────────────
GPU_SEMAPHORE_IN_USE = Gauge(
    "gpu_semaphore_in_use",
    "Number of inference sessions currently holding the GPU semaphore",
)

# ── Pipeline throughput ──────────────────────────────────────────
PIPELINE_FPS = Gauge(
    "pipeline_fps",
    "Rolling pipeline frames-per-second",
)

SESSION_FRAMES = Counter(
    "session_frames_total",
    "Frames processed per session",
    ["session_id"],
)

# ── LLM backlog ──────────────────────────────────────────────────
LLM_BACKLOG_SIZE = Gauge(
    "llm_backlog_size",
    "Pending LLM observation items per session",
    ["session_id"],
)
