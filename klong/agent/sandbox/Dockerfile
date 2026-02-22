FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy pandas scipy matplotlib seaborn scikit-learn \
    einops transformers datasets tokenizers \
    pytest black flake8 tqdm requests pillow \
    jax jaxlib

WORKDIR /workspace
