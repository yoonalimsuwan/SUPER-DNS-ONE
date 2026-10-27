# ============================================================================
# SESI TRAINING IMAGE
# Multi-GPU PyTorch training environment with biomedical dependencies
# ============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License      : MIT
# Year         : 2026
#
# Build:
#   docker build -t sesi/training:latest .
#
# Run (single GPU):
#   docker run --gpus all -v ./data:/data sesi/training:latest
#
# Run (multi-GPU DDP):
#   torchrun --nproc_per_node=4 train.py
# ============================================================================

# ── Base image: PyTorch 2.1 with CUDA 12.1 ─────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# ── Metadata ─────────────────────────────────────────────────────────────
LABEL maintainer="PAI & Yoon A Limsuwan <msps4u@gmail.com>"
LABEL org.mspsnetwork.project="SESI"
LABEL org.mspsnetwork.version="1.0.0"
LABEL description="SESI biomedical training environment"

# ── System dependencies ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────
# Core ML
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0

# Biomedical data processing
RUN pip install --no-cache-dir \
    biotite==0.40.0 \
    biopython==1.83 \
    requests \
    numpy \
    scipy \
    pandas \
    scikit-learn

# Training utilities
RUN pip install --no-cache-dir \
    tensorboard \
    tqdm \
    pyyaml \
    omegaconf

# Optional: molecular dynamics (uncomment if needed)
# RUN conda install -y -c conda-forge openmm openmm-ml

# ── Working directory ────────────────────────────────────────────────────
WORKDIR /workspace

# ── Copy source code ─────────────────────────────────────────────────────
COPY . /workspace

# ── Create data directories ──────────────────────────────────────────────
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# ── Environment variables ────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=4
ENV NCCL_DEBUG=INFO

# ── Default entrypoint ───────────────────────────────────────────────────
# Override with: docker run sesi/training:latest python train.py
ENTRYPOINT ["python"]
CMD ["train.py", "--config", "configs/training.yaml"]
