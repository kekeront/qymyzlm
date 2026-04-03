# KazLLM training container for Vertex AI
# Base: PyTorch with CUDA support for A100
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps (install before copying code for layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    torch>=2.3.0 \
    transformers>=4.44.0 \
    datasets>=2.20.0 \
    accelerate>=0.32.0 \
    tokenizers>=0.19.0 \
    sentencepiece>=0.2.0 \
    hydra-core>=1.3.2 \
    omegaconf>=2.3.0 \
    wandb>=0.17.0 \
    rich>=13.0.0 \
    numpy>=1.26.0 \
    google-cloud-storage

# Copy project code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Default: run training
ENTRYPOINT ["python", "scripts/train_vertex.py"]
