import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = "cpu"
YOLO_ACTIVE_MODEL = "yolov8x"

YOLO_MODEL_REGISTRY = {
    "yolov8x": {
        "path": os.path.join(MODELS_DIR, "yolov8x.pt"),
        "description": "YOLOv8x — largest v8, 68M params, highest accuracy",
        "size_mb": 130,
    },
    "yolov8s": {
        "path": os.path.join(MODELS_DIR, "yolov8s.pt"),
        "description": "YOLOv8s — small v8, 11M params, fastest inference",
        "size_mb": 22,
    },
    "yolo11x": {
        "path": os.path.join(MODELS_DIR, "yolo11x.pt"),
        "description": "YOLO11x — latest architecture, ~70M params, state of the art",
        "size_mb": 130,
    },
}

DEFAULT_TRACKER = "bytetrack"

HEATMAP_WINDOW_SIZE = 32
HEATMAP_STRIDE = 16
SRM_SENSITIVITY = 1.5

EXPORT_FORMAT = "onnx"
EXPORT_DIR = MODELS_DIR

MAX_IMAGE_SIZE = 1024
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".webp", ".bmp")

# ── Forensic config ──────────────────────────────────────────────────────────
ELA_QUALITY = 85
ELA_SCALE = 15

OCR_LANGUAGES = ("en",)
OCR_GPU = False

GAN_MODEL_REPO = "dima806/deepfake_vs_real_image_detection"
GAN_CONFIDENCE_THRESHOLD = 0.5
