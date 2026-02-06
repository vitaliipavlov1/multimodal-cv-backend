FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# --------------------
# System dependencies
# --------------------
RUN apt-get update && apt-get install -y \
    tzdata \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
 && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
 && python3 -m pip install --upgrade pip setuptools wheel \
 && rm -rf /var/lib/apt/lists/*

# --------------------
# Python dependencies
# --------------------
# requirements.txt копируем отдельно — хороший cache layer
COPY requirements.txt /app/requirements.txt

# Torch (CUDA 11.7 — СОВМЕСТИМО)
RUN pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html

# mmcv-full под torch 1.13 + cu117
RUN pip install --no-cache-dir \
    mmcv-full==1.7.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Paddle (CPU) + OCR
RUN pip install --no-cache-dir \
    paddlepaddle==2.6.2 \
    paddleocr==2.7.3

# Остальные зависимости проекта
RUN pip install --no-cache-dir -r /app/requirements.txt

# --------------------
# IMPORTANT
# --------------------
# ❗ Код НЕ копируем намертво
# ❗ Он монтируется volume'ом из docker-compose
# ❗ Поэтому любые изменения подхватываются после restart

# --------------------
# Run FastAPI
# --------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

