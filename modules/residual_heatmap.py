import numpy as np
import cv2
from scipy.signal import convolve2d

from .config import HEATMAP_WINDOW_SIZE, HEATMAP_STRIDE, SRM_SENSITIVITY

_SRM_FILTERS = {
    'f1h': np.array([[-1, 2, -1]], dtype=float) / 2.0,
    'f1v': np.array([[-1],[2],[-1]], dtype=float) / 2.0,
    'sq3': np.array([[-1,0,1],[0,0,0],[1,0,-1]], dtype=float) / 2.0,
}

class ResidualHeatmap:
    def __init__(self, window_size=None, stride=None, sensitivity=None):
        self.window_size = window_size or HEATMAP_WINDOW_SIZE
        self.stride = stride or HEATMAP_STRIDE
        self.sensitivity = sensitivity or SRM_SENSITIVITY

    def compute_srm_energy(self, gray_img):
        if gray_img.ndim == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
        gray = gray_img.astype(np.float64) / 255.0
        energy_map = np.zeros_like(gray)
        for k in _SRM_FILTERS.values():
            r = convolve2d(gray, k, mode='same', boundary='symm')
            energy_map += np.abs(r)
        return energy_map / len(_SRM_FILTERS)

    def generate_heatmap(self, image):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        gray = gray.astype(np.float64)
        h, w = gray.shape
        srm = self.compute_srm_energy(gray)
        heatmap = np.zeros((h, w), dtype=np.float32)
        ws = self.window_size
        st = self.stride
        for y in range(0, h - ws + 1, st):
            for x in range(0, w - ws + 1, st):
                region = srm[y:y+ws, x:x+ws]
                heatmap[y:y+ws, x:x+ws] += float(region.mean())
        heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
        heatmap = np.clip(heatmap * self.sensitivity, 0, 1)
        return heatmap

    def overlay_heatmap(self, image, heatmap=None, alpha=0.5):
        if heatmap is None:
            heatmap = self.generate_heatmap(image)
        heatmap_8u = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def get_stego_hotspots(self, heatmap, threshold=0.6):
        binary = (heatmap > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hotspots = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area > 50:
                hotspots.append({
                    "box": [x, y, x+w, y+h],
                    "area": int(area),
                    "mean_energy": float(heatmap[y:y+h, x:x+w].mean()),
                })
        hotspots.sort(key=lambda h: h["mean_energy"], reverse=True)
        return hotspots
