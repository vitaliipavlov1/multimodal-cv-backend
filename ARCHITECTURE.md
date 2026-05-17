# System Architecture — Real-Time Multimodal CV Platform

This document explains the complete architecture of the real-time multimodal computer vision inference system.

The goal of the system is to provide low-latency, production-grade GPU inference through a web browser while guaranteeing a stable frontend framerate independent of backend inference speed.

---

## High-Level Overview

The system is a distributed, event-driven inference platform composed of four major layers:

1) Browser client (UI + Engine)
2) AWS orchestration layer
3) GPU inference server (EC2)
4) Inference runtime (FastAPI + Triton)

The key architectural principle:

**Frontend FPS is decoupled from inference FPS.**

The backend may run at 5–20 FPS depending on GPU load, while the browser consistently renders ≥30 FPS.

---

## Complete Data Flow

User Browser
→ HTTPS
→ CloudFront + Static Hosting
→ AWS Lambda (start request)
→ EC2 GPU Instance
→ Nginx Reverse Proxy
→ FastAPI WebSocket Backend
→ Triton Inference Server (gRPC)
→ TensorRT Models on GPU

Results flow back:

GPU → Triton → FastAPI → WebSocket → Browser Engine → Canvas Rendering

---

## Components

### 1. Frontend UI
The UI is a control interface:
- session start
- model selection
- status monitoring
- stopping inference

The UI does NOT send video frames.

It only orchestrates sessions.

### 2. Engine (Browser Runtime)
The engine is embedded via an iframe and is responsible for:

• camera capture  
• frame encoding  
• WebSocket communication  
• rendering inference overlays  

Frames are captured using `getUserMedia`, resized to 320×240, JPEG encoded and streamed to backend via WebSocket.

The browser renders video continuously at 30–60 FPS, while inference results are interpolated between backend responses.

---

## Why an iframe is used

The engine runs inside an isolated runtime:

Benefits:
- independent lifecycle
- crash isolation
- safe restart
- UI remains responsive
- secure communication via postMessage

The UI communicates with the engine using:
`window.postMessage`

---

## AWS Orchestration Layer

The backend is not always running.

Instead, the system uses **on-demand GPU allocation**.

### Start sequence

1. UI calls AWS API Gateway
2. Lambda checks EC2 state
3. If stopped → EC2 is started
4. Lambda schedules auto-shutdown
5. UI waits for backend readiness

### Stop sequence

A scheduled EventBridge task triggers another Lambda that stops EC2 automatically after inactivity.

This reduces GPU cost dramatically.

---

## EC2 Runtime

When EC2 boots:

systemd automatically starts the inference stack.

No manual SSH interaction is required.

Boot chain:

EC2 boot → systemd → Docker Compose → Triton → Backend

---

## Reverse Proxy (Nginx)

Nginx runs on the host (not in Docker).

Responsibilities:

• TLS termination  
• HTTPS support  
• WebSocket proxy  
• health endpoint proxy  

WebSocket endpoint:

`wss://api.<domain>/ws/inference`

---

## Backend (FastAPI)

The backend is a fully asynchronous WebSocket server.

Responsibilities:

• receive frames (binary JPEG)  
• queue management  
• backpressure control  
• inference orchestration  
• result streaming  

The backend never renders images and never stores video.

It is completely stateless.

---

## Inference Pipeline

Each received frame goes through an asynchronous pipeline:

1. Frame decode
2. Optional motion estimation (Farneback optical flow)
3. YOLO object detection
4. YOLO Pose estimation
5. OCR (event-triggered)
6. Packaging results
7. Streaming to client

The pipeline supports parallel model execution.

---

## Triton Inference Server

All heavy ML models run inside NVIDIA Triton.

Reasons for using Triton:
- GPU scheduling
- dynamic batching
- TensorRT acceleration
- gRPC async inference

The backend communicates with Triton via asynchronous gRPC calls.

---

## Why dynamic batching matters

Multiple users may send frames simultaneously.

Triton groups frames into a single GPU batch:

Instead of:
1 frame → 1 GPU call

It performs:
N frames → 1 GPU call

This dramatically increases GPU utilization and throughput.

---

## Backpressure Control

The system must protect the GPU.

The backend uses a bounded frame queue.

If the client sends frames faster than the GPU can process:

older frames are dropped.

This guarantees:
- low latency
- stable UI
- no memory explosion

The system is latency-optimized, not throughput-optimized.

---

## Monitoring

Prometheus collects metrics from:

• FastAPI backend
• Triton inference server

Collected metrics:
- active sessions
- inference latency
- batch sizes
- pipeline FPS
- model errors

---

## System Design Goals

The architecture prioritizes:

• stability  
• predictable latency  
• safe resource usage  
• graceful degradation  

Not raw FPS.

The system is designed to survive:

- slow GPU inference
- multiple users
- long sessions
- network instability

---

## Why this is production architecture

Key production characteristics:

- automatic infrastructure lifecycle
- GPU cost control
- stateless backend
- fault isolation
- monitoring
- reverse proxy
- TLS
- async communication

This is not a demo ML server.

It is a real-time distributed inference platform.

