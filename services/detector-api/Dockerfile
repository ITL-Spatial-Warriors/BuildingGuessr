FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY ./src/detector_api/app.py .

ENV MODEL_ID="IDEA-Research/grounding-dino-base"
ENV DETECTOR_CLASS_NAMES="building"
ENV DETECT_CONF=0.3
ENV DETECT_TEXT_THRESH=0.3

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:$PATH"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
