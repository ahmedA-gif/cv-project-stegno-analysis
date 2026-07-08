import os, sys, numpy as np, cv2, math, io, json, warnings, time
warnings.filterwarnings('ignore')

from .yolo_detector import YOLODetector
from .residual_heatmap import ResidualHeatmap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.utils import draw_detections

class StegoAnalyzer:
    def __init__(self, model_wrapper=None, yolo_detector=None):
        self.model_wrapper = model_wrapper
        self.yolo = yolo_detector or YOLODetector()
        self.heatmapper = ResidualHeatmap()

    def set_model_wrapper(self, mw):
        self.model_wrapper = mw

    def analyze_image(self, image):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if max(image.shape[:2]) > 1024:
            scale = 1024 / max(image.shape[:2])
            new_w = int(image.shape[1] * scale)
            new_h = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_w, new_h))
            gray = cv2.resize(gray, (new_w, new_h))

        srm_energy = self.heatmapper.compute_srm_energy(gray)
        heatmap = self.heatmapper.generate_heatmap(image)

        detections = self.yolo.detect_stego_regions(image, srm_energy)

        if self.model_wrapper and self.model_wrapper.loaded:
            from app import extract_all_features
            for det in detections:
                x1, y1, x2, y2 = map(int, det["box"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    roi = image[y1:y2, x1:x2]
                    if max(roi.shape[:2]) > 512:
                        scale = 512 / max(roi.shape[:2])
                        roi = cv2.resize(roi, (int(roi.shape[1]*scale), int(roi.shape[0]*scale)))
                    fv = extract_all_features(roi)
                    svm_prob = self.model_wrapper.predict_svm(fv)[0]
                    urd_prob = self.model_wrapper.predict_urd(fv)[0]
                    det["stego_prob"] = round(max(svm_prob, urd_prob), 4)
                    det["svm_prob"] = round(svm_prob, 4)
                    det["urd_prob"] = round(urd_prob, 4)

        hotspots = self.heatmapper.get_stego_hotspots(heatmap)

        annotated = draw_detections(image, detections)
        heatmap_overlay = self.heatmapper.overlay_heatmap(image, heatmap)

        result = {
            "num_detections": len(detections),
            "detections": detections,
            "num_hotspots": len(hotspots),
            "hotspots": hotspots,
            "has_stego": any(d.get("stego_prob", 0) > 0.5 for d in detections) if detections else False,
        }

        if detections and any("stego_prob" in d for d in detections):
            probs = [d["stego_prob"] for d in detections if "stego_prob" in d]
            result["max_stego_prob"] = max(probs) if probs else 0
            result["avg_stego_prob"] = sum(probs) / len(probs) if probs else 0

        return result, annotated, heatmap_overlay
