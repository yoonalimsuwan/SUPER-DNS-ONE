#=============================================================================
# SESI CFD PRODUCTION VALIDATION — DOCKERFILE
# Multi-stage build for GPU-accelerated CFD with MPI + Triton + SDA-ONE
#=============================================================================
# Developer    : Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# License      : MIT
# Year         : 2026
# Version      : 1.0.0
#=============================================================================
# BUILD:
#   docker build -t sesi-cfd-prod:v6.0.0 .
#   docker build --build-arg CUDA_VERSION=12.2.0 -t sesi-cfd-prod:v6.0.0 .
#
# RUN (single-GPU):
#   docker run --gpus all -it sesi-cfd-prod:v6.0.0 python validate_env.py
#
# RUN (multi-GPU MPI):
#   docker run --gpus all --ipc=host --net=host -it sesi-cfd-prod:v6.0.0 \
#     mpirun -np 4 python sesi_cfd_production_validation.py
#
# RUN (interactive):
#   docker run --gpus all -v $(pwd):/workspace -it sesi-cfd-prod:v6.0.0 bash
#=============================================================================
# CARPET-19 DISCIPLINE:
#   • Build stages are DISCRETE (no continuous dependency graph)
#   • Each layer is FIRST-ORDER (install/check, no gradient-based decisions)
#   • Final image is minimal (no unnecessary packages)
#=============================================================================

# ============================================================================
# STAGE 1: Base CUDA + MPI
# ============================================================================
ARG CUDA_VERSION=12.2.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

LABEL maintainer="Yoon A Limsuwan <yoonalimsuwan@msps.network>"
LABEL org.opencontainers.image.title="SESI CFD Production Validation"
LABEL org.opencontainers.image.description="GPU-accelerated CFD with MPI + Triton + SDA-ONE"
LABEL org.opencontainers.image.version="6.0.0"
LABEL org.opencontainers.image.vendor="MSPS NETWORK"
LABEL org.opencontainers.image.licenses="MIT"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # MPI
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    # Scientific libraries
    libhdf5-dev \
    libfftw3-dev \
    liblapack-dev \
    libblas-dev \
    # Utilities
    vim \
    tmux \
    htop \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================================
# STAGE 2: Python Dependencies
# ============================================================================
FROM base AS python-deps

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Triton (GPU kernel compiler)
RUN pip install --no-cache-dir triton==2.2.0

# Install scientific computing stack
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.12.0 \
    scikit-image==0.22.0 \
    scikit-learn==1.4.0 \
    pandas==2.2.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.1

# Install MPI Python bindings
RUN pip install --no-cache-dir \
    mpi4py==3.1.5 \
    pymetis==2023.1

# Install additional dependencies
RUN pip install --no-cache-dir \
    h5py==3.10.0 \
    biopython==1.83 \
    mdtraj==1.9.9 \
    nglview==3.0.8 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    requests==2.31.0 \
    pillow==10.2.0

# ============================================================================
# STAGE 3: Application Code
# ============================================================================
FROM python-deps AS app

# Create non-root user for security
RUN groupadd -r sesi && useradd -r -g sesi -m -s /bin/bash sesi

# Create workspace directory
WORKDIR /workspace

# Copy application code
COPY sesi_cfd_production_validation.py .
COPY structural_domain_assembly_one.py .
COPY validate_env.py .
COPY submit_sesi_cfd.slurm .
COPY README.md .

# Set ownership
RUN chown -R sesi:sesi /workspace

# Switch to non-root user
USER sesi

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command: run validation
CMD ["python", "validate_env.py", "--phase", "full"]

# ============================================================================
# STAGE 4: Production (minimal runtime image)
# ============================================================================
FROM app AS production

# Remove development files to reduce image size
USER root
RUN rm -rf /var/cache/apt/archives/* /tmp/* /var/tmp/*

# Switch back to non-root user
USER sesi

# Final health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import mpi4py; assert torch.cuda.is_available()" || exit 1

# Entrypoint for production
ENTRYPOINT ["python"]
CMD ["sesi_cfd_production_validation.py"]
