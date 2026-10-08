"""Kernels 3x3 preestablecidos, normalización (Módulo A) y pesos 1D de desenfoque.

Los kernels se aplican como correlación (S1, R5.2): el peso k[r*3 + c] multiplica
al vecino (x + c - 1, y + r - 1), con y hacia abajo. Fila superior primero.
Los presets se guardan sin normalizar (R8.3): el box son nueve unos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Mapeos de visualización de display.frag (R9.2): u_map.
MAP_CLIP = 0    # recorte a [0, 1]
MAP_ABS = 1     # |v|
MAP_OFFSET = 2  # 0.5 + 0.5·v
MAP_NAMES = ("recorte", "|v|", "offset 0.5+0.5v")

# Por debajo de este valor absoluto la suma se considera 0 (R8.1, R8.2).
NORMALIZE_EPS = 1e-6

CUSTOM_KEY = "0"
CUSTOM_NAME = "Personalizado"
IDENTITY = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    weights: tuple[float, ...]  # 9 pesos, fila superior primero
    default_map: int


def _w(*values: float) -> tuple[float, ...]:
    assert len(values) == 9
    return tuple(float(v) for v in values)


# Tabla de design.md, teclas 1 a 8 (R6.1, R9.3).
PRESETS: dict[str, Preset] = {
    p.key: p
    for p in (
        Preset("1", "Identidad", _w(0, 0, 0, 0, 1, 0, 0, 0, 0), MAP_CLIP),
        Preset("2", "Box", _w(1, 1, 1, 1, 1, 1, 1, 1, 1), MAP_CLIP),
        Preset("3", "Gaussiano", _w(1, 2, 1, 2, 4, 2, 1, 2, 1), MAP_CLIP),
        Preset("4", "Sharpen", _w(0, -1, 0, -1, 5, -1, 0, -1, 0), MAP_CLIP),
        Preset("5", "Sobel Gx", _w(-1, 0, 1, -2, 0, 2, -1, 0, 1), MAP_ABS),
        Preset("6", "Sobel Gy", _w(-1, -2, -1, 0, 0, 0, 1, 2, 1), MAP_ABS),
        Preset("7", "Laplaciano", _w(0, 1, 0, 1, -4, 1, 0, 1, 0), MAP_ABS),
        Preset("8", "Emboss", _w(-2, -1, 0, -1, 1, 1, 0, 1, 2), MAP_CLIP),
    )
}


def normalize(weights) -> tuple[tuple[float, ...], bool]:
    """Divide los pesos por su suma si |suma| > 1e-6.

    Devuelve (pesos, sin_normalizar). `sin_normalizar` es True cuando la suma es
    prácticamente 0 y los pesos se devuelven sin cambios (R8.2).
    """
    w = tuple(float(v) for v in weights)
    total = sum(w)
    if abs(total) <= NORMALIZE_EPS:
        return w, True
    return tuple(v / total for v in w), False


# --- desenfoque 1D (R11, S8) ---------------------------------------------

# Longitud de `uniform float u_weights[61]` en blur_1d.frag (y blur_2d.frag).
# Un test comprueba que coincide con los shaders.
MAX_TAPS = 61
MAX_RADIUS = (MAX_TAPS - 1) // 2  # 30
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP = 0.5, 10.0, 0.5

PROFILE_GAUSS, PROFILE_BOX = "gauss", "box"
PROFILE_NAMES = {PROFILE_GAUSS: "gaussiano", PROFILE_BOX: "box"}


def _check_sigma(sigma: float) -> float:
    sigma = float(sigma)
    if not SIGMA_MIN <= sigma <= SIGMA_MAX:
        raise ValueError(f"sigma fuera de rango [{SIGMA_MIN}, {SIGMA_MAX}]: {sigma}")
    return sigma


def gaussian_radius(sigma: float) -> int:
    """r = ceil(3σ). El -1e-9 evita que 3·σ = 4.2000000001 suba a 5 por redondeo."""
    return math.ceil(3.0 * _check_sigma(sigma) - 1e-9)


def box_radius(sigma: float) -> int:
    """Box de varianza equivalente: una uniforme en [-r, r] tiene varianza r²/3, así que r = round(√3·σ)."""
    return round(math.sqrt(3.0) * _check_sigma(sigma))  # σ = 0.5 → 1, σ = 10 → 17


def gaussian_1d(sigma: float) -> tuple[float, ...]:
    """Pesos w_i = exp(-i²/2σ²) para i = -r..r, normalizados a suma 1."""
    r = gaussian_radius(sigma)
    w = [math.exp(-(i * i) / (2.0 * sigma * sigma)) for i in range(-r, r + 1)]
    total = sum(w)
    return tuple(v / total for v in w)


def box_1d(sigma: float) -> tuple[float, ...]:
    """2r+1 pesos iguales a 1/(2r+1), con r = round(√3·σ)."""
    n = 2 * box_radius(sigma) + 1
    return (1.0 / n,) * n


def blur_weights(profile: str, sigma: float) -> tuple[float, ...]:
    """Pesos 1D del perfil (`gauss` o `box`); la longitud es 2r+1 ≤ MAX_TAPS."""
    if profile == PROFILE_GAUSS:
        return gaussian_1d(sigma)
    if profile == PROFILE_BOX:
        return box_1d(sigma)
    raise ValueError(f"perfil desconocido: {profile}")


def blur_radius(profile: str, sigma: float) -> int:
    return gaussian_radius(sigma) if profile == PROFILE_GAUSS else box_radius(sigma)


def format_matrix(weights) -> str:
    """Matriz 3x3 en texto, una fila por línea."""
    w = [float(v) for v in weights]
    rows = []
    for r in range(3):
        rows.append("  [" + "  ".join(f"{v:9.4f}" for v in w[r * 3 : r * 3 + 3]) + " ]")
    return "\n".join(rows)
