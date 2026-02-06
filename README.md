# Multimodal CV Inference Backend

Production-ready **real-time multimodal computer vision backend** built around
**FastAPI**, **Triton Inference Server**, and **GPU acceleration**.

➔ Live demo: https://vitaliipavlov.website

⚠️ GPU instance is auto-started on demand (cold start ~30–60s)

The system is designed for **low-latency streaming inference** with strict FPS
requirements on the frontend, independent of backend inference speed.

---

## Key Features

- 🔥 Real-time **WebSocket-based inference**
- ⚡ High-performance **GPU inference via Triton**
- 🧠 Multimodal pipeline:
  - Object detection (YOLOv8)
  - Human pose estimation (YOLOv8 Pose)
  - OCR (PaddleOCR)
  - Motion estimation (Farneback optical flow)
- 📦 Dynamic batching with backpressure control
- 🔒 HTTPS + WebSocket proxy via Nginx
B- 📊 Prometheus metrics (backend + Triton)
- 🤖 Optional LLM-based semantic summarization (LangChain)

---

## System Architecture

Client (Browser / Frontend, 30+ FPS)
│
│ HTTPS + WebSocket
▼
Nginx
│
│ WS / HTTP
▼
FastAPI Backend (Async)
│
│ gRPC
▼
Triton Inference Server
│
│ TensorRT
▼
GPU


**Key principle**  
> Backend inference FPS and frontend rendering FPS are intentionally decoupled.  
> The frontend always targets **≥30 FPS**, regardless of inference speed.

---

## Inference Pipeline

Each frame passes through an **asynchronous multimodal pipeline**:

1. Frame received via WebSocket
2. Optional motion estimation (CPU)
3. Optional inference stages:
   - YOLO (objects / persons)
   - YOLO Pose (keypoints)
   - OCR (event-based, throttled)
4. Postprocessing & normalization
5. Results streamed back to client
6. (Optional) Batched summaries sent to LLM asynchronously

Pipeline entry point:
app/pipeline/multimodal.py


---

## Triton Inference Server

All heavy models run **exclusively via Triton**:

- TensorRT engines (`.plan`)
- Dynamic batching enabled
- gRPC async client
- Single shared client instance (singleton)

Model repository:
triton/model_repository/
├── yolo/
└── yolo_pose/


### Dynamic Batching (Triton)

Example configuration:

preferred_batch_size: [1, 2, 4, 8, 16]
max_queue_delay_microseconds: 2000


---

## Backend Design Highlights

### Async & Concurrency

- Fully asynchronous FastAPI backend
- Shared Triton gRPC client (singleton)
- GPU access limited via `asyncio.Semaphore`
- No blocking operations inside the event loop

### Backpressure & Stability

- Frame queues with drop strategy
- LLM backlog with RAM + disk spillover
- Adaptive LLM rate based on backlog size
- Safe cancellation on client disconnect

### WebSocket-First API

- Binary JPEG frames as input
- JSON inference results as output
- Long-lived sessions supported (hours)

---

## Monitoring & Metrics

Prometheus metrics are exposed at:

GET /metrics


Collected metrics:

- Active inference sessions
- Per-model frame counters
- Batch sizes
- End-to-end inference latency
- Pipeline FPS
- Triton GPU metrics (latency, batching)

Prometheus configuration:

prometheus/prometheus.yml


---

## Docker & Deployment

### Services

- **backend** — FastAPI inference API
- **triton** — Triton Inference Server (GPU)
- **prometheus** — metrics collection
- **nginx** — host-based reverse proxy

### Deployment

Start the full inference stack:

```bash
docker compose up --build
```

The backend source code is mounted as a Docker volume, allowing rapid iteration without rebuilding the container image.

---

## Nginx Reverse Proxy (HTTPS & WebSocket)

Nginx runs on the **host machine (not inside Docker)** and acts as the secure network perimeter of the system.

### Responsibilities

- TLS/HTTPS termination
- HTTP → HTTPS redirection
- WebSocket proxy (`/ws/inference`)
- Health endpoint proxying

### Reference configuration

```
nginx/
backend.conf
http_redirect.conf
README.md
```

---

## OCR Design Strategy

OCR is intentionally **not real-time**.

It operates using an **event-based model**:

- Triggered only when needed
- Throttled by minimum interval
- CPU-safe
- Non-blocking to the main inference pipeline

This prevents OCR from degrading overall system latency.

---

## LLM Integration (Optional)

When enabled:

- Vision outputs are summarized asynchronously
- The inference pipeline is never blocked
- Strict output formatting is enforced
- Request rate adapts to backlog pressure

The LLM is treated as a **best-effort enhancement**, not part of the critical path.

---

## Reliability Guarantees

### Design principles

- Stability over peak FPS
- Predictable latency
- Graceful degradation
- Clear separation of concerns
- Production-first architecture

The system is designed to tolerate:

- Slow models
- Bursty clients
- Long-lived sessions
- GPU contention

---

## EC2 Lifecycle & Cost Optimization

The backend runs on a GPU EC2 instance (e.g., `g4dn.xlarge`) that is started and stopped on demand.

### Startup Flow

1. Frontend calls AWS Lambda `start-server`
2. Lambda starts EC2 if it is stopped
3. On boot, `systemd` automatically launches Docker Compose
4. Services start in order:
   - Triton Inference Server
   - FastAPI Backend
   - Prometheus
5. Backend exposes the WebSocket inference API

### Shutdown Flow

- A scheduled AWS Lambda automatically stops EC2 after inactivity
- The backend is fully stateless and safe to terminate at any time

### Design goals

- No manual SSH required
- Safe cold starts
- GPU resources used only when needed
- systemd-managed Docker Compose

### Reference

```
infra/systemd/inference-compose.service.example
```

---

## Status

- Production-ready
- GPU-accelerated
- Scalable
- Observable
- Portfolio-grade system

---

## License

MIT
