# 🕵️ SteganoScan — Neural Steganalysis Pipeline

**Multi-dataset steganalysis using 256-dim rich model features + ensemble stacking.**

> **Authors:** Muhammad Ahmed Naeem, Sameed-ul-Hassan, Hamza Shahid

---

## 📋 Overview

SteganoScan detects hidden data embedded within images using steganography.

- **Robust SRM+SVM:** AUC **0.8605**
- **UltraRobustDetector (Ensemble):** AUC **0.8738**

---

## 🔬 Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ 9 Datasets   │──▶│ 256-dim      │──▶│  Scaling +   │──▶│ Ensemble Stacking│
│ (20K images) │   │  Features    │   │  PCA (99%)   │   │ SVM+RF+ET+XGB+LGB│
└──────────────┘   └──────────────┘   └──────────────┘   └────────┬─────────┘
                                                                    │
                                                    ┌───────────────▼──────────┐
                                                    │  LogisticRegression     │
                                                    │  Meta-Learner (5-fold)  │
                                                    └───────────┬─────────────┘
                                                                │
                                                    ┌───────────▼─────────────┐
                                                    │  Prediction: CLEAN /    │
                                                    │  STEGO + Confidence     │
                                                    └─────────────────────────┘
```

---

## 🗂️ Datasets

| Dataset | Clean | Stego | Stego Methods |
|---------|-------|-------|---------------|
| **ALASKA2** | 1,000 | 3,000 | JMiPOD, JUNIWARD, UERD |
| **BOSSBase+BOWS2** | 2,000 | 2,000 | GBRASNET |
| **ILSVRC2012** | 1,000 | 0 | — |
| **STEGANAYIS (S-UNIWARD)** | 1,000 | 2,000 | SUNI_02, SUNI_04 |
| **Stego-PVD** | 1,000 | 1,000 | PVD-based |
| **StegoImages** | 1,000 | 1,000 | Various |
| **UCID** | 1,032 | 0 | — |
| **iPhone (6s, 8, X)** | 400 | 2,400 | Multiple methods |
| **DIV2K** | 1,000 | 0 | — |
| **Total** | **9,432** | **11,400** | |

---

## 🧠 Feature Extraction (256-dim)

| Group | Dim | Purpose |
|-------|-----|---------|
| SRM Filters | 70 | 10 residual filter co-occurrences |
| SRM RGB | 18 | Cross-channel residual stats |
| LSB Entropy | 12 | Per-channel bit-plane entropy |
| Chi-Square | 3 | Pairs-of-values attack |
| Moments | 12 | Mean, std, skew, kurtosis |
| Gradient | 2 | Entropy + Laplacian variance |
| FFT Spectrum | 2 | Beta exponent + high-freq ratio |
| RLE | 1 | Run-length uniformity |
| Color Correlation | 3 | Histogram correlation (RGB) |
| Wavelet Haar | 3 | LH/HL/HH subband std |
| Markov LSB | 4 | 2×2 transition matrix |
| Benford's Law | 2 | First-digit deviation |
| PVD | 6 | Pixel difference flatness |
| GLCM | 8 | Texture features (2 directions) |
| WS (Weighted Stego) | 4 | Wu-Shi residual |
| RS Analysis | 6 | Regular/Singular asymmetry |
| Calibration Residual | 3 | JPEG Q75 recompress diff |
| 4-gram LSB | 16 | 4-bit co-occurrence |

---

## 🏆 Evaluation Metrics

### Robust SRM+SVM (threshold=0.490)

```
AUC:      0.8605
Accuracy: 77.20%

              precision    recall  f1-score
     clean      0.7700    0.6995    0.7330
     stego      0.7734    0.8307    0.8010
```

### UltraRobustDetector (threshold=0.570)

```
AUC:      0.8738
Accuracy: 77.44%

              precision    recall  f1-score
     clean      0.7508    0.7423    0.7465
     stego      0.7931    0.8004    0.7968
```

### Per-Dataset Breakdown (URD)

| Dataset | AUC | Accuracy |
|---------|-----|----------|
| ALASKA2 | 0.5304 | 66.71% |
| BOSSBase | 0.5779 | 56.58% |
| IPHONE | **1.0000** | **100.00%** |
| STEGANAYIS | 0.6701 | 62.18% |
| Stego-PVD | **0.9814** | **94.15%** |
| StegoImages | **0.9785** | **93.06%** |
| DIV2K | — | 100.00% |
| ILSVRC2012 | — | 88.48% |
| UCID | — | 98.50% |

---

## 🚀 Deployment

### Hugging Face Space
**https://huggingface.co/spaces/ahmedg12104/Stegno-image-analysis**

Drag & drop any image → real-time prediction + full metrics dashboard.

### API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Web UI |
| `/api/status` | GET | Model status |
| `/api/metrics` | GET | Validation metrics |
| `/api/metrics/histogram` | GET | Per-dataset chart (PNG) |
| `/api/predict` | POST | Upload image → prediction |

### Local Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 app.py
# → http://127.0.0.1:5050
```

---

## 📁 Structure

```
├── app.py                 # Flask backend + API
├── requirements.txt
├── Dockerfile
├── cv-project.ipynb       # Kaggle training notebook
├── robust_svm.pkl         # SVM model (8 MB)
├── urd.pkl                # Ensemble model (414 MB)
├── templates/index.html   # Web UI
└── README.md
```
