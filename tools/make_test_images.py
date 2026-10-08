"""Genera las imágenes sintéticas de prueba en assets/images/.

- orientacion.png: una "F" asimétrica con un marcador rojo arriba a la izquierda,
  para comprobar que la imagen no sale espejada ni invertida.
- escalon.png: escalón vertical negro → blanco (bordes, Sobel, Canny).
- tablero.png: tablero de ajedrez (modos de borde, wrap frente a reflect).
- circulo.png: disco blanco sobre negro (bordes en todas las direcciones).
- ruido.png: escalón suave con ruido gaussiano (desenfoque, Canny).

Uso: python tools/make_test_images.py [carpeta_destino]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "assets" / "images"


def make_orientacion(w: int = 640, h: int = 480) -> Image.Image:
    img = Image.new("RGB", (w, h), (32, 32, 48))
    d = ImageDraw.Draw(img)
    # "F": barra vertical, barra superior larga y barra media corta.
    x0, y0, thick = 200, 80, 60
    d.rectangle([x0, y0, x0 + thick - 1, h - 81], fill=(240, 240, 240))
    d.rectangle([x0, y0, x0 + 260 - 1, y0 + thick - 1], fill=(240, 240, 240))
    d.rectangle([x0, 210, x0 + 180 - 1, 210 + thick - 1], fill=(240, 240, 240))
    # Marcador rojo en la esquina superior izquierda y verde en la inferior derecha.
    d.rectangle([0, 0, 39, 39], fill=(255, 0, 0))
    d.rectangle([w - 40, h - 40, w - 1, h - 1], fill=(0, 200, 0))
    return img


def make_escalon(w: int = 320, h: int = 240) -> Image.Image:
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[:, w // 2 :] = 255
    return Image.fromarray(arr)  # uint8 2D → modo "L"


def make_tablero(w: int = 320, h: int = 240, cell: int = 20) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w]
    arr = (((x // cell) + (y // cell)) % 2 * 255).astype(np.uint8)
    return Image.fromarray(arr)  # uint8 2D → modo "L"


def make_circulo(w: int = 256, h: int = 256) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w]
    r = min(w, h) * 0.35
    arr = (((x - w / 2 + 0.5) ** 2 + (y - h / 2 + 0.5) ** 2 <= r * r) * 255).astype(np.uint8)
    return Image.fromarray(arr)  # uint8 2D → modo "L"


def make_ruido(w: int = 320, h: int = 240, seed: int = 1234) -> Image.Image:
    rng = np.random.default_rng(seed)
    x = np.arange(w, dtype=np.float64)
    ramp = 1.0 / (1.0 + np.exp(-(x - w / 2) / 4.0))  # escalón suave
    base = np.tile(0.2 + 0.6 * ramp, (h, 1))
    noisy = np.clip(base + rng.normal(0.0, 0.08, size=(h, w)), 0.0, 1.0)
    return Image.fromarray(np.round(noisy * 255).astype(np.uint8))  # uint8 2D → modo "L"


GENERATORS = {
    "orientacion.png": make_orientacion,
    "escalon.png": make_escalon,
    "tablero.png": make_tablero,
    "circulo.png": make_circulo,
    "ruido.png": make_ruido,
}


def generate(out_dir: Path = DEFAULT_OUT) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, make in GENERATORS.items():
        path = out_dir / name
        make().save(path)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[0]) if argv else DEFAULT_OUT
    for path in generate(out_dir):
        print(f"generada: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
