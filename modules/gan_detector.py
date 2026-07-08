"""GAN / AI-generated image detection using ViT + frequency heuristics"""

import os
import numpy as np
import cv2
from io import BytesIO
from PIL import Image


class GANDetector:
    def __init__(self):
        self._model = None
        self._processor = None

    def _lazy_load_model(self):
        if self._model is not None:
            return True
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            repo = "dima806/deepfake_vs_real_image_detection"
            self._processor = AutoImageProcessor.from_pretrained(repo)
            self._model = AutoModelForImageClassification.from_pretrained(repo)
            self._model.eval()
            return True
        except Exception as e:
            print(f"GAN ViT model load failed: {e}")
            return False

    def _frequency_features(self, gray):
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        power = np.abs(fshift) ** 2
        power_log = np.log1p(power)

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
        max_r = min(cx, cy)
        radial = np.array([power_log[dist == r].mean() if (dist == r).sum() > 0 else 0 for r in range(max_r)])

        low = radial[:max_r // 3].sum()
        mid = radial[max_r // 3: 2 * max_r // 3].sum()
        high = radial[2 * max_r // 3:].sum()
        total = low + mid + high + 1e-9

        hf_ratio = high / total
        mf_ratio = mid / total
        lf_ratio = low / total

        return {
            "hf_ratio": float(hf_ratio),
            "mf_ratio": float(mf_ratio),
            "lf_ratio": float(lf_ratio),
            "spectral_entropy": float(-np.sum((radial / (radial.sum() + 1e-9)) * np.log2(radial / (radial.sum() + 1e-9) + 1e-9))),
        }

    def _noise_correlation(self, gray):
        noise = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)
        local_std = cv2.blur(noise ** 2, (16, 16))
        corr = float(np.corrcoef(noise[:100, :100].flatten(), noise[100:200, 100:200].flatten())[0, 1])
        return {
            "noise_std_mean": float(np.sqrt(local_std).mean()),
            "noise_correlation": abs(corr) if not np.isnan(corr) else 0.0,
        }

    def predict(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        freq = self._frequency_features(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        noise = self._noise_correlation(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

        model_prob = None
        if self._lazy_load_model():
            try:
                pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
                inputs = self._processor(images=pil_img, return_tensors="pt")
                import torch
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    model_prob = float(probs[0, 1].item())
            except Exception as e:
                print(f"GAN model inference failed: {e}")

        # Heuristic score based on frequency + noise analysis
        hf_score = max(0, min(1, (freq["hf_ratio"] - 0.05) * 10))
        noise_score = max(0, min(1, noise["noise_correlation"] * 5))
        heuristic_prob = 0.6 * hf_score + 0.4 * noise_score

        if model_prob is not None:
            combined = 0.7 * model_prob + 0.3 * heuristic_prob
        else:
            combined = heuristic_prob

        return {
            "prediction": "AI_GENERATED" if combined > 0.5 else "NATURAL",
            "probability": round(combined, 4),
            "model_probability": round(model_prob, 4) if model_prob is not None else None,
            "heuristic_probability": round(heuristic_prob, 4),
            "model_available": model_prob is not None,
            "frequency_analysis": freq,
            "noise_analysis": noise,
        }
