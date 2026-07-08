"""Text detection in images using EasyOCR — text regions are common stego targets"""

import os
import numpy as np
import cv2


class TextDetector:
    def __init__(self, languages=("en",), gpu=False):
        self.languages = languages
        self.gpu = gpu
        self._reader = None

    def _lazy_load(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu)

    def detect(self, image):
        self._lazy_load()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        results = self._reader.readtext(image)
        detections = []
        for bbox, text, confidence in results:
            pts = np.array(bbox, dtype=np.int32)
            x1 = int(min(pts[:, 0]))
            y1 = int(min(pts[:, 1]))
            x2 = int(max(pts[:, 0]))
            y2 = int(max(pts[:, 1]))
            detections.append({
                "box": [x1, y1, x2, y2],
                "text": text,
                "confidence": round(float(confidence), 4),
                "length": len(text),
            })
        return detections

    def analyze(self, image):
        detections = self.detect(image)
        num_texts = len(detections)
        total_chars = sum(d["length"] for d in detections)
        avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0.0
        avg_area = np.mean([(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) for d in detections]) if detections else 0.0

        return {
            "num_text_regions": num_texts,
            "total_characters": total_chars,
            "avg_confidence": round(float(avg_conf), 4),
            "avg_region_area": round(float(avg_area), 1),
            "detections": detections,
        }

    def draw_text_regions(self, image, detections=None):
        if detections is None:
            detections = self.detect(image)
        out = image.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 165, 0), 2)
            label = f"{d['text'][:30]} {d['confidence']:.2f}"
            cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        return out
