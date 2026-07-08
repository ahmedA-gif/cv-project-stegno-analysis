"""Error Level Analysis — JPEG tampering detection (no model needed)"""

import cv2
import numpy as np
from io import BytesIO
from PIL import Image


class ELAAnalyzer:
    def __init__(self, quality=85, scale=15):
        self.quality = quality
        self.scale = scale

    def compute_ela_map(self, image):
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        pil_in = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        buf = BytesIO()
        pil_in.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        pil_recomp = Image.open(buf).convert("RGB")
        recomp = np.array(pil_recomp, dtype=np.float32)

        diff = np.abs(image.astype(np.float32) - recomp)
        ela = np.clip(diff * self.scale, 0, 255).astype(np.uint8)
        return ela, diff

    def analyze(self, image):
        ela_map, diff = self.compute_ela_map(image)
        gray_ela = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY) if ela_map.ndim == 3 else ela_map

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))
        max_diff = float(np.max(diff))
        p95_diff = float(np.percentile(diff, 95))

        _, thresh = cv2.threshold(gray_ela, 30, 255, cv2.THRESH_BINARY)
        suspicious_pixels = float(np.sum(thresh > 0))
        total_pixels = thresh.size
        suspicious_ratio = suspicious_pixels / total_pixels

        # Regions of interest
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append({
                    "box": [int(x), int(y), int(x + w), int(y + h)],
                    "area": int(area),
                    "mean_ela": float(np.mean(gray_ela[y:y + h, x:x + w])),
                })

        regions.sort(key=lambda r: r["area"], reverse=True)
        regions = regions[:20]

        ela_score = min(1.0, suspicious_ratio * 5)
        prediction = "MANIPULATED" if ela_score > 0.3 else "ORIGINAL"

        return {
            "prediction": prediction,
            "ela_score": round(ela_score, 4),
            "suspicious_ratio": round(suspicious_ratio, 4),
            "mean_error": round(mean_diff, 4),
            "std_error": round(std_diff, 4),
            "max_error": round(max_diff, 4),
            "p95_error": round(p95_diff, 4),
            "num_suspicious_regions": len(regions),
            "suspicious_regions": regions,
        }

    def overlay_ela(self, image, alpha=0.6):
        ela_map, _ = self.compute_ela_map(image)
        ela_colored = cv2.applyColorMap(ela_map, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(image, alpha, ela_colored, 0.4, 0)
        return overlay
