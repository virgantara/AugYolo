# utils_xray.py
import numpy as np
from PIL import Image
import cv2
import pywt

class UnsharpMask:
    """
    Unsharp masking using Gaussian blur.
    amount: edge boost strength (0.0–2.0)
    radius: Gaussian sigma (~0.5–3.0)
    threshold: ignore small differences (0–255) to avoid noise amplification
    p: probability to apply (augmentation use); set p=1.0 for preprocessing
    """
    def __init__(self, amount=0.8, radius=1.0, threshold=0, p=1.0):
        self.amount = float(amount)
        self.radius = float(radius)
        self.threshold = int(threshold)
        self.p = float(p)

    def __call__(self, img):
        if np.random.rand() > self.p: 
            return img
        arr = np.array(img.convert('L'))  # ensure grayscale
        blur = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=self.radius)
        mask = cv2.subtract(arr, blur)
        if self.threshold > 0:
            mask[np.abs(mask) < self.threshold] = 0
        sharp = cv2.addWeighted(arr, 1.0, mask, self.amount, 0)
        return Image.fromarray(sharp).convert('RGB')  # replicate to 3ch for ImageNet models

class WaveletDenoise:
    """
    BayesShrink soft-thresholding in wavelet domain.
    wavelet: 'db2', 'db3', 'sym4' often work well
    level: decomposition levels (2–4 for 512–1024px)
    p: probability to apply
    """
    def __init__(self, wavelet='db2', level=2, p=1.0):
        self.wavelet = wavelet
        self.level = int(level)
        self.p = float(p)

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        x = np.array(img.convert('L'), dtype=np.float32) / 255.0
        coeffs = pywt.wavedec2(x, self.wavelet, level=self.level)
        cA, cDs = coeffs[0], coeffs[1:]

        # Estimate noise sigma from finest detail coeffs
        cH, cV, cD = cDs[-1]
        sigma = np.median(np.abs(cD)) / 0.6745 + 1e-8

        # BayesShrink thresholds per subband
        new_cDs = []
        for (cH, cV, cD) in cDs:
            for band in (cH, cV, cD):
                var = np.mean(band**2)
                t = (sigma**2 / max(np.sqrt(var), 1e-8))
                band[:] = np.sign(band) * np.maximum(np.abs(band) - t, 0.0)
            new_cDs.append((cH, cV, cD))

        rec = pywt.waverec2((cA, new_cDs), self.wavelet)
        rec = np.clip(rec * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(rec).convert('RGB')

class StructureMap:
    """
    Produces a binary/float structure map using adaptive threshold + morphology,
    then concatenates it as a 3-channel RGB (original in gray replicated).
    mode='concat' returns RGB where G channel is the structure map mixed in.
    """
    def __init__(self, block_size=31, C=5, morph_open=1, morph_close=2, alpha=0.4, p=1.0):
        self.block_size = int(block_size) | 1  # must be odd
        self.C = int(C)
        self.morph_open = int(morph_open)
        self.morph_close = int(morph_close)
        self.alpha = float(alpha)
        self.p = float(p)

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        g = np.array(img.convert('L'))
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, self.block_size, self.C)
        k_open = (self.morph_open*2+1, self.morph_open*2+1)
        k_close = (self.morph_close*2+1, self.morph_close*2+1)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_open))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_close))
        th = (th / 255.0).astype(np.float32)

        # Blend structure map into luminance
        g_norm = (g / 255.0).astype(np.float32)
        enhanced = np.clip((1 - self.alpha) * g_norm + self.alpha * th, 0, 1)
        rgb = (enhanced * 255.0).astype(np.uint8)
        return Image.fromarray(rgb).convert('RGB')