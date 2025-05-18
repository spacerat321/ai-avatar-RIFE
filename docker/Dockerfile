# -------------------------
# Base: CUDA 11.8 + cuDNN8 + Ubuntu 22.04
# -------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# -------------------------
# System setup
# -------------------------
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Ensure python and pip default to Python 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# -------------------------
# Copy source code
# -------------------------
COPY . /workspace

# -------------------------
# Install Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Default command
# -------------------------
ENTRYPOINT ["python", "inference_video.py"]
