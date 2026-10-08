"""Referencias en numpy para comparar con los shaders.

Mismas convenciones que la GPU (D2, S1): img[y, x] con la fila 0 arriba y
kernels como correlación (k[r, c] multiplica a img[y + r - 1, x + c - 1]).
"""

from __future__ import annotations

import numpy as np

# u_border → modo de numpy.pad (R10.5).
NP_MODES = {0: "constant", 1: "edge", 2: "reflect", 3: "wrap"}


def pad(img: np.ndarray, r: int, border: int) -> np.ndarray:
    """Rellena r píxeles por cada lado en los ejes y, x (no en los canales)."""
    width = [(r, r), (r, r)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, width, mode=NP_MODES[border])


def shift(img: np.ndarray, ox: int, oy: int, border: int) -> np.ndarray:
    """out[y, x] = img[y + oy, x + ox], con el modo de borde dado."""
    h, w = img.shape[:2]
    r = max(abs(ox), abs(oy))
    p = pad(img, r, border)
    return p[r + oy : r + oy + h, r + ox : r + ox + w].copy()


def correlate3x3(img: np.ndarray, kernel, border: int) -> np.ndarray:
    """Correlación 3x3 de img (H×W×C) con 9 pesos, fila superior primero."""
    k = np.asarray(kernel, dtype=np.float64).reshape(3, 3)
    src = img.astype(np.float64)
    out = np.zeros_like(src)
    for r in range(3):
        for c in range(3):
            out += k[r, c] * shift(src, c - 1, r - 1, border)
    return out


def blur_1d(img: np.ndarray, weights, direction: tuple[int, int], border: int) -> np.ndarray:
    """out[y, x] = Σ_i w[i + r] · img[(y, x) + i·direction]."""
    w = np.asarray(weights, dtype=np.float64)
    r = (len(w) - 1) // 2
    src = img.astype(np.float64)
    out = np.zeros_like(src)
    for i in range(-r, r + 1):
        out += w[i + r] * shift(src, i * direction[0], i * direction[1], border)
    return out


def blur_separable(img: np.ndarray, weights, border: int) -> np.ndarray:
    return blur_1d(blur_1d(img, weights, (1, 0), border), weights, (0, 1), border)


def grayscale(img: np.ndarray) -> np.ndarray:
    """Y Rec.601 de img[..., :3] (S4)."""
    return img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


SOBEL_X = (-1, 0, 1, -2, 0, 2, -1, 0, 1)
SOBEL_Y = (-1, -2, -1, 0, 0, 0, 1, 2, 1)


def sobel(gray: np.ndarray, border: int) -> np.ndarray:
    """H×W×4 con (Gx, Gy, |G|, θ = arctan2(Gy, Gx))."""
    g = gray[..., None]
    gx = correlate3x3(g, SOBEL_X, border)[..., 0]
    gy = correlate3x3(g, SOBEL_Y, border)[..., 0]
    return np.stack([gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)], axis=-1)


def nms(grad: np.ndarray, border: int) -> np.ndarray:
    """Supresión de no máximos de nms.frag sobre grad[..., 2] = |G| y grad[..., 3] = θ.

    Se calcula en float32, como la GPU, para que la cuantización de θ coincida.
    """
    m = grad[..., 2].astype(np.float32)
    a = np.mod(np.degrees(grad[..., 3].astype(np.float32)) + np.float32(180), np.float32(180))
    dirs = np.select(
        [(a < 22.5) | (a >= 157.5), a < 67.5, a < 112.5],
        [0, 1, 2],
        default=3,
    )
    offsets = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    out = np.zeros_like(m)
    for k, (dx, dy) in enumerate(offsets):
        n1 = shift(m, dx, dy, border)
        n2 = shift(m, -dx, -dy, border)
        keep = (dirs == k) & (m >= n1) & (m > n2)
        out[keep] = m[keep]
    return out


def double_threshold(m: np.ndarray, low: float, high: float, gain: float, border: int):
    """(bordes 0/1, clasificación 0/0.5/1) de threshold.frag, con vecinos según el borde."""
    v = m.astype(np.float32) * np.float32(gain)
    strong = v >= np.float32(high)
    weak = (v >= np.float32(low)) & ~strong
    near_strong = np.zeros_like(strong)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                near_strong |= shift(strong.astype(np.float32), dx, dy, border) > 0.5
    edges = strong | (weak & near_strong)
    cls = np.where(strong, 1.0, np.where(weak, 0.5, 0.0))
    return edges.astype(np.float64), cls
