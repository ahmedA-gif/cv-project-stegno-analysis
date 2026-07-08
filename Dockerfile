FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir opencv-python-headless>=4.8 --force-reinstall

# ── Pre-download YOLO models ────────────────────────────────────────────────
# YOLOv8x (130 MB) — largest v8, highest accuracy
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
# YOLOv8s (22 MB) — small v8, fastest
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
# YOLO11x (130 MB) — latest architecture
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"

# ── DeepSORT re-ID model (5 MB) ────────────────────────────────────────────
RUN python3 -c "from boxmot import DeepSort; DeepSort()" || true

# ── EasyOCR model (EN, ~185 MB) ────────────────────────────────────────────
RUN python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False)" || true

# ── GAN detection model (ViT, ~340 MB) ─────────────────────────────────────
RUN python3 -c "from transformers import AutoImageProcessor, AutoModelForImageClassification; AutoImageProcessor.from_pretrained('dima806/deepfake_vs_real_image_detection'); AutoModelForImageClassification.from_pretrained('dima806/deepfake_vs_real_image_detection')" || true

COPY . .

EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "--threads", "2"]
