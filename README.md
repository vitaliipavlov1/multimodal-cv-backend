# Multimodal CV Inference Backend

Production-ready real-time multimodal computer vision backend built on **FastAPI**, **NVIDIA Triton Inference Server**, and **TensorRT FP16**.

➔ Live demo: https://vitaliipavlov.website

> GPU instance starts on demand — cold start ~30–60s

---

## Stack

| Component | Technology |
|---|---|
| Detection | YOLO26s TensorRT FP16 — NMS-free |
| Tracking | ByteTrack with lapjv optimal assignment |
| Pose estimation | YOLO26s-pose TensorRT FP16 |
| OCR | PaddleOCR (CPU, throttled) |
| Motion | Farneback dense optical flow |
| Inference server | Triton 24.08 (TensorRT 10.3) |
| API | FastAPI async WebSocket |
| LLM | LangChain + GPT-4o-mini (optional) |
| Metrics | Prometheus |
| Proxy | Nginx (HTTPS + WebSocket) |
| Infrastructure | AWS EC2 g4dn.xlarge (T4 16GB) |

---

## Architecture

```
Browser (30+ FPS)
    │ HTTPS / WSS
    ▼
Nginx (TLS termination, WS proxy)
    │
    ▼
FastAPI Backend (async)
    │ gRPC
    ▼
Triton Inference Server 24.08
    │ TensorRT FP16
    ▼
NVIDIA T4 GPU
```

Frontend and backend FPS are intentionally decoupled — the frontend always targets ≥30 FPS regardless of inference latency.

---

## Inference Pipeline

Each frame goes through an async multimodal pipeline:

1. Binary JPEG received via WebSocket
2. JPEG decoded off the event loop (ThreadPoolExecutor)
3. Concurrent inference via `asyncio.gather`:
   - YOLO26s detection → ByteTrack tracking
   - YOLO26s-pose → 17 keypoints per person
   - PaddleOCR (throttled, event-based)
   - Farneback optical flow (CPU)
4. Results streamed back as JSON
5. Batched summaries sent to LLM asynchronously

---

## Models

### YOLO26s Detection
- Input: FP16 `[1, 3, 640, 640]`
- Output: FP32 `[1, 300, 6]` — `[x1, y1, x2, y2, conf, cls]`
- NMS-free end-to-end — no post-processing NMS needed
- Built with `trtexec --fp16 --inputIOFormats=fp16:chw`

### YOLO26s Pose
- Input: FP16 `[1, 3, 640, 640]`
- Output: FP32 `[1, 300, 57]` — boxes + 17 keypoints × 3
- NMS-free, single class (person)

### ByteTrack
- Two-pass IoU association (high-conf → all tracks, low-conf → unmatched)
- Optimal assignment via lapjv (lapx)
- Per-stream tracker registry with TTL pruning
- Stable `track_id` per detection across frames

---

## Project Structure

```
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py                    # FastAPI app, WebSocket handler
│       ├── inference/
│       │   ├── yolo.py                # YOLO26 detection + ByteTracker
│       │   ├── yolo_pose.py           # YOLO26 pose estimation
│       │   ├── triton_client.py       # Shared async gRPC client
│       │   ├── motion.py              # Farneback optical flow
│       │   └── ocr.py                 # PaddleOCR (throttled)
│       ├── pipeline/
│       │   └── multimodal.py          # Concurrent inference pipeline
│       └── monitoring/
│           └── metrics.py             # Prometheus metrics
├── triton/
│   └── model_repository/
│       ├── yolo/                      # Detection model (TRT FP16)
│       └── yolo_pose/                 # Pose model (TRT FP16)
├── nginx/
│   ├── backend.conf                   # HTTPS + WebSocket proxy
│   └── http_redirect.conf
├── prometheus/
│   └── prometheus.yml
├── infra/
│   └── systemd/
│       └── inference-compose.service.example
└── docker-compose.yml
```

---

## Deployment

### Requirements
- AWS EC2 g4dn.xlarge (NVIDIA T4, 16GB VRAM)
- Ubuntu 22.04, NVIDIA Driver 580+, Docker + NVIDIA Container Toolkit

### Build TensorRT engines
```bash
# Detection
docker run --rm --gpus all -v ~/project:/workspace \
  nvcr.io/nvidia/tensorrt:24.08-py3 trtexec \
  --onnx=/workspace/onnx/yolo26s.onnx \
  --saveEngine=/workspace/triton/model_repository/yolo/1/model.plan \
  --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp32:chw

# Pose
docker run --rm --gpus all -v ~/project:/workspace \
  nvcr.io/nvidia/tensorrt:24.08-py3 trtexec \
  --onnx=/workspace/onnx/yolo26s-pose.onnx \
  --saveEngine=/workspace/triton/model_repository/yolo_pose/1/model.plan \
  --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp32:chw
```

### Start the stack
```bash
cp .env.example .env   # add OPENAI_API_KEY
docker compose up -d
```

### Systemd autostart
```bash
sudo cp infra/systemd/inference-compose.service.example \
  /etc/systemd/system/inference-compose.service
sudo systemctl daemon-reload
sudo systemctl enable inference-compose
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required for LLM summarization |

Copy `.env.example` to `.env` and fill in the values. The `.env` file is gitignored.

---

## API

### WebSocket `/ws/inference`

**Connect → send `init` → send `start` → stream binary JPEG frames**

```json
// init message
{
  "type": "init",
  "params": {
    "use_yolo": true,
    "use_yolo_pose": true,
    "use_ocr": false,
    "use_motion": false,
    "use_langchain": false
  }
}
```

```json
// inference result (per frame)
{
  "type": "inference",
  "data": {
    "yolo": {
      "boxes": [[x1, y1, x2, y2], ...],
      "classes": [0, ...],
      "scores": [0.95, ...],
      "track_ids": [1, 2, ...]
    },
    "pose": {
      "boxes": [...],
      "keypoints": [[[x, y, conf], ...], ...]
    },
    "flow": {"fx": [...], "fy": [...], "step": 16},
    "fps": 28.4
  }
}
```

### REST
- `GET /health` — health check
- `GET /metrics` — Prometheus metrics

---

## EC2 Lifecycle

The GPU instance starts and stops automatically via AWS Lambda:

1. Frontend → Lambda `start-inference-server` → EC2 Start
2. systemd boots → Docker Compose up (Triton → Backend → Prometheus)
3. Scheduled Lambda stops EC2 after inactivity

No manual SSH required for normal operation.

---

## License

MIT
