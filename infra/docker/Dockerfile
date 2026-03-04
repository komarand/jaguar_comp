FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git build-essential \
    ca-certificates \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.docker.txt /app/requirements.docker.txt

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision \
 && python3 -m pip install -r /app/requirements.docker.txt

COPY . /app

CMD ["bash"]