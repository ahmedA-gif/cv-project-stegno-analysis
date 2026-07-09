---
title: STEGANOSCAN
emoji: 🕵️
colorFrom: green
colorTo: gray
sdk: docker
python_version: "3.12"
app_file: app.py
pinned: false
---

# ⦿ STEGANOSCAN — Neural Forensics Terminal v4.2

A **Flask-based neural forensics web application** for image steganalysis, tamper detection, object detection/tracking, and AI-generated image identification. Trained on **16,505 images** and validated on **4,127 images** with a **256-dimensional feature vector** extracted via SRM (Spatial Rich Model) filters and related steganalysis techniques.

> **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/ahmedg12104/Stegno-image-analysis)

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Modules](#modules)
   - [Config (`modules/config.py`)](#config-modulesconfigpy)
   - [YOLO Detector (`modules/yolo_detector.py`)](#yolo-detector-modulesyolo_detectorpy)
   - [Stego Analyzer (`modules/stego_analyzer.py`)](#stego-analyzer-modulesstego_analyzerpy)
   - [Residual Heatmap (`modules/residual_heatmap.py`)](#residual-heatmap-modulesresidual_heatmappy)
   - [ELA Forensics (`modules/ela_forensics.py`)](#ela-forensics-modulesela_forensicspy)
   - [Text OCR (`modules/text_ocr.py`)](#text-ocr-modulestext_ocrpy)
   - [GAN Detector (`modules/gan_detector.py`)](#gan-detector-modulesgan_detectorpy)
   - [Utils (`modules/utils.py`)](#utils-modulesutilspy)
   - [Trackers (`modules/trackers/`)](#trackers-modulestrackers)
4. [Feature Extraction Pipeline (256-dim)](#feature-extraction-pipeline-256-dim)
5. [ML Models](#ml-models)
   - [Steganalysis Models](#steganalysis-models)
   - [YOLO Models](#yolo-models)
   - [GAN Detection Model](#gan-detection-model)
6. [API Endpoints](#api-endpoints)
7. [UI / Frontend](#ui--frontend)
8. [Installation & Setup](#installation--setup)
9. [Docker & HuggingFace Spaces](#docker--huggingface-spaces)
10. [Project Structure](#project-structure)
11. [Dependencies](#dependencies)

---

## Features

- **Steganalysis Detection** — SVM + URD ensemble models detect hidden data in images using 18 feature families (256-dim)
- **YOLO Object Detection** — YOLOv8x/YOLOv8s/YOLO11x with SRM residual heatmap overlay
- **Multi-Object Tracking** — 5 trackers: ByteTrack, Bot-SORT, Deep SORT, Strong SORT, OC-SORT
- **Error Level Analysis (ELA)** — JPEG tamper detection without ML models
- **OCR Text Detection** — EasyOCR-based text region extraction
- **GAN/AI Image Detection** — Vision Transformer (ViT) + frequency heuristic hybrid
- **All-in-One Forensics Pipeline** — Run all 5 analyses in a single request
- **Cyberpunk Terminal UI** — Matrix rain, CRT scanlines, glitch animations, glass-morphism panels
- **Interactive Charts** — Radar chart, per-dataset bar chart, animated GIF metrics, probability gauge
- **Drag-and-Drop** — Image upload via click or drag-and-drop on all analysis tabs

---

## Architecture Overview

```
User Browser (HTML/JS/CSS)
        │
        ▼ HTTP
  Flask Web Server (app.py)
        │
        ├── /api/predict  ──► Feature Extraction (256-dim) ──► SVM / URD
        ├── /api/yolo/*   ──► YOLODetector + ResidualHeatmap
        ├── /api/forensics/*
        │   ├── ela       ──► ELAAnalyzer
        │   ├── ocr       ──► TextDetector (EasyOCR)
        │   ├── gan       ──► GANDetector (ViT + FFT heuristics)
        │   └── full      ──► All of the above + steganalysis + YOLO
        ├── /api/tracker/* ──► Tracker Factory (5 trackers)
        ├── /api/metrics   ──► Validation metrics + charts
        └── /              ──► index.html (SPA frontend)
```

---

## Modules

### Config (`modules/config.py`)

Central configuration constants for all modules:

| Setting | Value | Description |
|---------|-------|-------------|
| `YOLO_CONF_THRESHOLD` | `0.25` | Detection confidence threshold |
| `YOLO_IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `YOLO_DEVICE` | `cpu` | Inference device |
| `YOLO_DEFAULT` | `yolov8x` | Default YOLO model |
| `YOLO_MODEL_REGISTRY` | `yolov8x`, `yolov8s`, `yolo11x` | Available models |
| `HEATMAP_WINDOW` | `32` | SRM sliding window size |
| `HEATMAP_STRIDE` | `16` | SRM sliding window stride |
| `HEATMAP_SENSITIVITY` | `1.5` | Heatmap sensitivity multiplier |
| `ELA_QUALITY` | `85` | JPEG re-compression quality |
| `ELA_SCALE` | `15` | ELA error amplification scale |
| `OCR_LANGUAGES` | `("en",)` | EasyOCR language list |
| `OCR_GPU` | `False` | Use GPU for OCR |
| `GAN_MODEL_REPO` | `dima806/deepfake_vs_real_image_detection` | HF model for GAN detection |
| `MAX_IMAGE_SIZE` | `1024` | Max dimension for analysis |
| `PREDICT_MAX_SIZE` | `512` | Max dimension for steganalysis |
| `ALLOWED_EXTENSIONS` | `.png,.jpg,.jpeg,.tiff,.webp,.bmp` | Supported image formats |
| `DEFAULT_TRACKER` | `bytetrack` | Default tracking algorithm |

---

### YOLO Detector (`modules/yolo_detector.py`)

#### `YOLODetector`
Wraps Ultralytics YOLO for object detection.

| Method | Description |
|--------|-------------|
| `detect(image)` | Run inference → list of detections with `box`, `confidence`, `class_id`, `label` |
| `detect_stego_regions(image, srm_energy_map)` | Detect + enrich each detection with mean SRM energy |
| `export_onnx()` | Export model to ONNX format |
| `get_model_info()` | Model metadata: task, classes, param count |

#### `YOLOPool`
Multi-model manager with lazy loading.

| Method | Description |
|--------|-------------|
| `get_model(name)` | Get or create a detector instance |
| `switch(name)` | Swap active model |
| `list_models()` | Return registry descriptions |

**Models:**

| Name | Params | Size | Description |
|------|--------|------|-------------|
| `yolov8x` | 68M | 130MB | Highest accuracy (default) |
| `yolov8s` | 11M | 22MB | Fastest inference |
| `yolo11x` | ~70M | 130MB | Latest architecture |

---

### Stego Analyzer (`modules/stego_analyzer.py`)

#### `StegoAnalyzer`
Orchestrates combined steganalysis analysis:

1. Runs YOLO detection on the image
2. Computes SRM residual heatmap
3. For each detected region, extracts 256-dim features and runs SVM + URD prediction
4. Returns per-detection probabilities, hotspot contours, and aggregate scores

---

### Residual Heatmap (`modules/residual_heatmap.py`)

#### `ResidualHeatmap`
Computes SRM (Spatial Rich Model) energy maps:

- Applies 3 SRM filters: `f1h` (horizontal), `f1v` (vertical), `sq3` (square 3×3)
- Sliding-window aggregation (32×32 window, 16 stride)
- Generates COLORMAP_JET overlay
- Extracts stego hotspot contours (regions with energy > mean + sensitivity × std)
- Returns heatmap image (base64) and hotspot polygons

---

### ELA Forensics (`modules/ela_forensics.py`)

#### `ELAAnalyzer`
JPEG tamper detection without ML models:

1. Re-saves image at JPEG quality `ELA_QUALITY` (default 85)
2. Computes pixel-wise absolute difference between original and re-encoded
3. Scales error by `ELA_SCALE` (default 15×)
4. Thresholds to find suspicious regions (error > 95th percentile)
5. Returns:
   - `ela_score` — mean error
   - `mean_error`, `max_error`, `p95_error`
   - `suspicious_ratio` — fraction of pixels above threshold
   - `suspicious_regions` — bounding contours > 50 px²
   - Overlay heatmap (COLORMAP_HOT)
   - Verdict: `"MANIPULATED"` (score > 0.3) or `"ORIGINAL"`

---

### Text OCR (`modules/text_ocr.py`)

#### `TextDetector`
Wraps EasyOCR (default: English, CPU).

| Method | Description |
|--------|-------------|
| `analyze(image)` | Detect text → returns `num_text_regions`, `total_characters`, `avg_confidence`, `avg_region_area`, per-region details (`box`, `text`, `confidence`, `length`) |
| `draw_text_regions(image)` | Annotate image with bounding boxes and recognized text |

---

### GAN Detector (`modules/gan_detector.py`)

#### `GANDetector`
Two-pronged AI-generated image detection:

**1. ViT Model (HuggingFace)**
- Model: `dima806/deepfake_vs_real_image_detection`
- Vision Transformer fine-tuned for real vs. fake classification
- Lazy-loaded on first use
- Gracefully falls back to heuristics if model unavailable

**2. Heuristic Analysis**
- **FFT Frequency Features:** radial power spectrum, high/mid/low frequency ratios, spectral entropy
- **Noise Correlation:** local noise std, spatial noise correlation

**Combined Score:** `0.7 × model_probability + 0.3 × heuristic_probability`

**Verdict:** `"AI_GENERATED"` (> 0.5) or `"NATURAL"`

---

### Utils (`modules/utils.py`)

| Function | Description |
|----------|-------------|
| `load_image(path, max_size)` | Read and resize image |
| `image_to_base64(img)` / `base64_to_image(b64)` | Base64 encode/decode |
| `rgb_to_gray(img)` | Color conversion |
| `xyxy_to_xywh()` / `xywh_to_xyxy()` / `clip_box()` | Bounding box operations |
| `iou(box_a, box_b)` | Intersection-over-Union |
| `draw_detections(img, detections)` | Annotate image with boxes (green < 0.5 stego prob, red ≥ 0.5) |
| `serialize_result(obj)` | Recursive numpy‑to‑native Python conversion for JSON |

---

### Trackers (`modules/trackers/`)

| File | Class | Backend | Description |
|------|-------|---------|-------------|
| `base_tracker.py` | `BaseTracker` (ABC) | — | Abstract interface: `track()`, `get_tracked_objects()`, `reset()` |
| `byte_tracker.py` | `ByteTracker` | Ultralytics `bytetrack.yaml` | Lightweight, fast |
| `bot_sort_tracker.py` | `BotSortTracker` | Ultralytics `botsort.yaml` | robust re-identification |
| `deep_sort_tracker.py` | `DeepSortTracker` | `boxmot.DeepSort` | Appearance-based |
| `strong_sort_tracker.py` | `StrongSortTracker` | `boxmot.StrongSort` | Robust with reID |
| `oc_sort_tracker.py` | `OCSortTracker` | `boxmot.OCSort` | Observation-centric |
| `tracker_factory.py` | Factory | — | `get_tracker(name, model, config)`, `list_trackers()` |

All trackers accept a YOLO model and return tracked objects with `track_id`, `box`, `confidence`, `class_id`, `label`.

---

## Feature Extraction Pipeline (256-dim)

The 256-dimensional feature vector is assembled from 18 feature families:

| # | Family | Dims | Description |
|---|--------|------|-------------|
| 1 | **SRM gray** | 50 | 10 SRM filters × 5 histogram bins (gray channel) |
| 2 | **SRM RGB** | 18 | 3 channels × 3 filters × 2 stats (mean, std) |
| 3 | **LSB entropy** | 12 | 3 channels × 4 bit-planes |
| 4 | **Chi-square** | 3 | Per-channel chi-square statistic |
| 5 | **Statistical moments** | 12 | 3 channels × 4 moments (mean, var, skew, kurt) |
| 6 | **Gradient features** | 2 | Gradient entropy + Laplacian variance |
| 7 | **FFT frequency** | 2 | Beta slope + high-frequency ratio |
| 8 | **Run-length encoding** | 1 | LSB run-length uniformity |
| 9 | **Color correlation** | 3 | RGB channel correlation |
| 10 | **Wavelet features** | 3 | LH, HL, HH subband std (Haar DWT) |
| 11 | **Markov transition** | 4 | LSB 2×2 transition probabilities |
| 12 | **Benford's law deviation** | 2 | Pixel + first-difference Benford fit |
| 13 | **PVD flatness** | 6 | Pixel-value differencing over 6 ranges |
| 14 | **GLCM features** | 8 | Contrast, energy, homogeneity, correlation (4 angles × 2 stats) |
| 15 | **Weighted stego (WS)** | 4 | WS statistics per channel |
| 16 | **RS analysis** | 6 | Regular/singular group statistics |
| 17 | **JPEG calibration** | 3 | JPEG compression fingerprint |
| 18 | **LSB 4-gram** | 16 | 4-bit LSB pattern frequencies |

---

## ML Models

### Steganalysis Models

| Model | File | Type | AUC | Accuracy | Description |
|-------|------|------|-----|----------|-------------|
| **Robust SVM** | `robust_svm.pkl` | SVM + PCA + StandardScaler | 0.8605 | 77.20% | Primary steganalysis model |
| **URD** | `urd.pkl` | Stacking ensemble (SVM + tree) + meta-model | 0.8738 | 77.44% | Unified Robust Detector |

Both models auto-download from GitHub Releases if missing:
```
https://github.com/ahmedA-gif/cv-project-stegno-analysis/releases/download/v1.0.0/robust_svm.pkl
https://github.com/ahmedA-gif/cv-project-stegno-analysis/releases/download/v1.0.0/urd.pkl
```

**Per-Dataset AUC** (from validation):

| Dataset | SVM AUC | URD AUC | SVM Acc | URD Acc |
|---------|---------|---------|---------|---------|
| ALASKA2 | 0.9305 | 0.9387 | — | — |
| BOSSBASE | 0.6565 | 0.6572 | — | — |
| IPHONE | 0.8641 | 0.9174 | — | — |
| STEGANAYIS | 0.7610 | 0.7780 | — | — |
| STEGO-PVD | 0.8530 | 0.8479 | — | — |
| STEGOIMAGES | 0.9856 | 0.9863 | — | — |
| UCID | 0.9549 | 0.9614 | — | — |

### YOLO Models

| Name | Params | Size | Task |
|------|--------|------|------|
| `yolov8x` | 68.3M | 130MB | Object detection (default) |
| `yolov8s` | 11.2M | 22MB | Object detection |
| `yolo11x` | ~70M | 130MB | Object detection |

### GAN Detection Model

| Model | Source | Type | Input |
|-------|--------|------|-------|
| `dima806/deepfake_vs_real_image_detection` | HuggingFace Hub | ViT (Vision Transformer) | 224×224 RGB |

---

## API Endpoints

### System & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the main HTML UI |
| `GET` | `/api/status` | Server status, model load state, version (4.2.0), feature dim |
| `GET` | `/api/metrics` | Validation metrics for SVM + URD (AUC, accuracy, per-dataset) |
| `GET` | `/api/metrics/animated` | Animated GIF — per-dataset bar chart (30 frames) |
| `GET` | `/api/metrics/histogram` | Static PNG — per-dataset AUC + Accuracy histogram |

### Steganalysis Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Upload image → extract 256-dim features → SVM + URD → probability + verdict |

**Request:** `multipart/form-data` with `image` field.
**Response:**
```json
{
  "success": true,
  "inference_time": 1.234,
  "results": {
    "urd": {
      "prediction": "STEGO",
      "probability": 0.8912,
      "threshold": 0.570,
      "confidence": 89.12
    },
    "robust_svm": {
      "prediction": "STEGO",
      "probability": 0.7234,
      "threshold": 0.490,
      "confidence": 72.34
    }
  }
}
```

### YOLO Detection

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/yolo/status` | YOLO model load status and metadata |
| `GET` | `/api/yolo/models` | List available models and active one |
| `POST` | `/api/yolo/switch` | Switch active model (`{"model": "yolov8s"}`) |
| `POST` | `/api/yolo/detect` | Upload image → YOLO detection + SRM heatmap → annotated + heatmap images |
| `POST` | `/api/yolo/analyze` | Full analysis: YOLO + per-region stego probabilities + hotspot contours |

### Tracker

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/tracker/list` | List available tracker algorithms and active one |
| `POST` | `/api/tracker/select` | Select active tracker (`{"tracker": "deepsort"}`) |
| `POST` | `/api/tracker/track` | Upload image → YOLO + tracker → annotated image with track IDs |

### Forensics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/forensics/ela` | Error Level Analysis → tamper score, suspicious regions, ELA heatmap |
| `POST` | `/api/forensics/ocr` | OCR text detection → text regions, character count, annotated image |
| `POST` | `/api/forensics/gan` | GAN/AI detection → ViT + heuristic probability, frequency/noise analysis |
| `POST` | `/api/forensics/full` | **All-in-one:** steganalysis + YOLO + ELA + OCR + GAN |

---

## UI / Frontend

**Single-page application** served from `templates/index.html` with a cyberpunk/hacker-terminal theme.

### Tabs

| Tab | Features |
|-----|----------|
| **Scan_Target** | Image upload → steganalysis gauge + verdict + stats (chi-square, RS, entropy, LSB) + spectral variance bars + terminal log |
| **YOLO_Detect** | Model selector + image upload → annotated image + SRM heatmap + detections table |
| **Forensics** | Sub-tabs: [ELA], [OCR], [GAN], [ALL-IN-1] → color-coded results with typing animation, raw JSON expand |
| **Tracker** | Algorithm selector + image upload → tracked objects with persistent IDs |
| **Metrics** | Model performance cards + radar chart + per-dataset bar chart + animated GIF + score list |

### Visual Theme

- **Colors:** Neon green (`#00ff41`), cyan (`#00eefc`), amber (`#ffb000`), dark background (`#050505`)
- **Effects:** CRT scanlines overlay, Matrix rain (katakana + ASCII), glass-morphism panels, glitch/flicker animations, animated border pulses
- **Typography:** JetBrains Mono (monospace)
- **Custom scrollbar** styled in neon green

---

## Installation & Setup

### Prerequisites

- Python 3.12+
- pip

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/ahmedA-gif/cv-project-stegno-analysis.git
cd cv-project-stegno-analysis

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

The app starts at `http://localhost:5050`.

> Models (`robust_svm.pkl`, `urd.pkl`) auto-download from GitHub Releases on first startup. YOLO weights download from Ultralytics on first use.

### Configuration

Edit `modules/config.py` to adjust:
- YOLO confidence/IoU thresholds
- Heatmap sensitivity
- ELA quality/scale
- OCR languages
- Max image dimensions

---

## Docker & HuggingFace Spaces

### Docker Build

```bash
docker build -t steganoscan .
docker run -p 7860:7860 steganoscan
```

### HuggingFace Spaces

The project is configured for Docker-based HuggingFace Spaces:

1. Fork/push the repo to a HF Space
2. Space SDK: **Docker**
3. Port: **7860**
4. The Space auto-builds on push

**Environment variables** (optional):
- `PORT` — server port (default: `5050` dev, `7860` Docker)

---

## Project Structure

```
cv-project/
├── app.py                      # Flask application (1007 lines)
├── Dockerfile                  # Docker image for HF Spaces
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── manifest.json               # Training metadata
├── robust_svm.pkl              # SVM model (auto-downloaded)
├── urd.pkl                     # URD ensemble model (auto-downloaded)
├── yolov8x.pt                  # YOLOv8x weights
├── yolov8s.pt                  # YOLOv8s weights
├── yolo11x.pt                  # YOLO11x weights
├── validation_metrics_animated.gif
│
├── templates/
│   └── index.html              # Single-page UI (718 lines)
│
├── static/                     # Static assets
│
├── uploads/                    # Temporary uploads (auto-cleaned)
│
├── modules/
│   ├── __init__.py
│   ├── config.py               # Global configuration
│   ├── yolo_detector.py        # YOLO detection + model pool
│   ├── stego_analyzer.py       # Combined stego region analyzer
│   ├── residual_heatmap.py     # SRM heatmap generation
│   ├── ela_forensics.py        # Error Level Analysis
│   ├── text_ocr.py             # EasyOCR text detection
│   ├── gan_detector.py         # GAN/AI image detection
│   ├── utils.py                # Helper functions
│   └── trackers/
│       ├── __init__.py
│       ├── base_tracker.py     # Abstract interface
│       ├── byte_tracker.py     # ByteTrack
│       ├── bot_sort_tracker.py # Bot-SORT
│       ├── deep_sort_tracker.py# Deep SORT
│       ├── strong_sort_tracker.py # Strong SORT
│       ├── oc_sort_tracker.py  # OC-SORT
│       └── tracker_factory.py  # Factory + registry
│
└── Phase 2/                    # Earlier version (identical structure)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| flask | ≥3.0 | Web framework |
| gunicorn | ≥22.0 | Production WSGI server |
| numpy | ≥1.24, <2.0 | Numerical computation |
| scipy | ≥1.10 | Signal processing (convolve2d, gaussian_filter) |
| scikit-learn | ≥1.3 | ML models (SVM, PCA, scalers, ensemble) |
| opencv-python-headless | ≥4.8 | Computer vision |
| Pillow | ≥10.0 | Image I/O |
| joblib | ≥1.3 | Model serialization |
| matplotlib | ≥3.7 | Charts and GIF generation |
| xgboost | ≥2.0 | URD base model |
| lightgbm | ≥4.0 | URD base model |
| ultralytics | ≥8.2 | YOLO models |
| boxmot | ≥10.0 | Multi-object trackers |
| easyocr | ≥1.7 | OCR text detection |
| transformers | ≥4.36, <5.0.0 | HuggingFace ViT for GAN detection |
| huggingface-hub | ≥0.20 | Model download from HF Hub |

---

## License

MIT
