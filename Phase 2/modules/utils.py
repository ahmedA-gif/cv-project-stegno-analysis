import cv2
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image

def load_image(path, max_size=1024):
    img = cv2.imread(path)
    if img is None:
        img = cv2.imdecode(np.frombuffer(open(path, "rb").read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def image_to_base64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

def base64_to_image(b64_str):
    buf = base64.b64decode(b64_str)
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def rgb_to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def clip_box(box, h, w):
    x1, y1, x2, y2 = box
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

def iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)

def draw_detections(img, detections, color=(0, 255, 0), thickness=2):
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = f"{det['label']} {det['confidence']:.2f}"
        if "stego_prob" in det:
            label += f" S:{det['stego_prob']:.2f}"
            sc = (0, 0, 255) if det["stego_prob"] > 0.5 else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), sc, thickness)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out

def serialize_result(result):
    if isinstance(result, np.ndarray):
        return result.tolist()
    if isinstance(result, np.generic):
        return result.item()
    if isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [serialize_result(v) for v in result]
    return result
