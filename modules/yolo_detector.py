import os, warnings
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = '0'

import numpy as np
import cv2
from ultralytics import YOLO

from .config import (
    YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_DEVICE,
    YOLO_MODEL_REGISTRY, YOLO_ACTIVE_MODEL, MODELS_DIR,
)


class YOLODetector:
    def __init__(self, model_name=None, conf_threshold=None, iou_threshold=None, device=None):
        self.conf_threshold = conf_threshold or YOLO_CONF_THRESHOLD
        self.iou_threshold = iou_threshold or YOLO_IOU_THRESHOLD
        self.device = device or YOLO_DEVICE
        self.model_name = model_name or YOLO_ACTIVE_MODEL
        self.model_path = YOLO_MODEL_REGISTRY[self.model_name]["path"]
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(MODELS_DIR, exist_ok=True)
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Failed to load {self.model_path}: {e}")
            try:
                self.model = YOLO(f"{self.model_name}.pt")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                self.model = None

    def detect(self, image, conf=None, iou=None):
        if self.model is None:
            return []
        results = self.model(
            image,
            conf=conf or self.conf_threshold,
            iou=iou or self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf, cls in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy()
            ):
                detections.append({
                    "box": [float(x) for x in box],
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "label": r.names[int(cls)],
                })
        return detections

    def detect_stego_regions(self, image, srm_energy_map=None):
        dets = self.detect(image)
        if srm_energy_map is not None and len(dets) > 0:
            h, w = srm_energy_map.shape[:2]
            for d in dets:
                x1, y1, x2, y2 = map(int, d["box"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                region_energy = float(srm_energy_map[y1:y2, x1:x2].mean())
                d["srm_energy"] = round(region_energy, 4)
        return dets

    def export_onnx(self, path=None, imgsz=640):
        if self.model is None:
            return None
        out = path or os.path.join(MODELS_DIR, f"{self.model_name}.onnx")
        self.model.export(format="onnx", imgsz=imgsz, dynamic=True)
        return out

    def get_model_info(self):
        if self.model is None:
            return {"status": "not_loaded", "model_name": self.model_name}
        info = {
            "status": "loaded",
            "model_name": self.model_name,
            "task": self.model.task,
            "names": self.model.names,
            "model_path": self.model_path,
        }
        try:
            info["num_params"] = sum(p.numel() for p in self.model.model.parameters())
        except:
            info["num_params"] = "unknown"
        return info


class YOLOPool:
    def __init__(self):
        self._models = {}
        self._active_name = YOLO_ACTIVE_MODEL

    def get_model(self, model_name=None):
        name = model_name or self._active_name
        if name not in YOLO_MODEL_REGISTRY:
            valid = list(YOLO_MODEL_REGISTRY.keys())
            name = valid[0]
        if name not in self._models:
            self._models[name] = YOLODetector(model_name=name)
        self._active_name = name
        return self._models[name]

    def switch(self, model_name):
        if model_name not in YOLO_MODEL_REGISTRY:
            return False, f"Unknown model '{model_name}'. Available: {list(YOLO_MODEL_REGISTRY.keys())}"
        if model_name not in self._models:
            try:
                self._models[model_name] = YOLODetector(model_name=model_name)
            except Exception as e:
                return False, f"Failed to load {model_name}: {e}"
        self._active_name = model_name
        return True, f"Switched to {model_name}"

    @property
    def active_name(self):
        return self._active_name

    @property
    def active_model(self):
        return self.get_model()

    @classmethod
    def list_models(cls):
        return {k: v["description"] for k, v in YOLO_MODEL_REGISTRY.items()}
