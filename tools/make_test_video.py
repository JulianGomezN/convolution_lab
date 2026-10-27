"""Genera la animación de prueba assets/videos/demo.gif (R14.6, R14.7).

- 320×240, 60 frames de 50 ms (3 s a 20 fps), bucle infinito.
- La "F" asimétrica de orientacion.png desplazándose en horizontal, para
  comprobar que el vídeo no sale espejado ni invertido.
- Una barra que gira 3° por frame (bordes en todas las orientaciones para
  Sobel y Canny).
- Marcador rojo fijo arriba a la izquierda y verde abajo a la derecha.
- Contador binario de 6 bits en la fila inferior que identifica el frame
  (bit más significativo a la izquierda; blanco = 1, gris = 0).

Determinista: se dibuja en modo "P" con una paleta fija de 8 colores y sin
aleatoriedad, y se guarda con optimize=False, así que dos ejecuciones producen
los mismos píxeles.

Uso: .venv\\Scripts\\python.exe tools\\make_test_video.py [carpeta_destino]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "assets" / "videos"
FILENAME = "demo.gif"

WIDTH, HEIGHT = 320, 240
N_FRAMES = 60
DURATION_MS = 50
DEGREES_PER_FRAME = 3.0
COUNTER_BITS = 6

# Paleta fija de 8 colores (índice → RGB).
BG, WHITE, RED, GREEN, YELLOW, GRAY, BLUE, BLACK = range(8)
PALETTE = [
    (32, 32, 48),  # fondo, igual que orientacion.png
    (240, 240, 240),  # "F" y bits a 1
    (255, 0, 0),  # marcador superior izquierdo
    (0, 200, 0),  # marcador inferior derecho
    (255, 200, 0),  # barra giratoria
    (96, 96, 96),  # bits a 0
    (40, 90, 220),  # eje de la barra
    (0, 0, 0),  # sin usar (completa las 8 entradas)
]


def _palette_bytes() -> list[int]:
    return [c for rgb in PALETTE for c in rgb]


def _draw_f(d: ImageDraw.ImageDraw, x0: int) -> None:
    # Misma forma que en orientacion.png, a media escala.
    y0, thick = 30, 24
    d.rectangle([x0, y0, x0 + thick - 1, 149], fill=WHITE)  # barra vertical
    d.rectangle([x0, y0, x0 + 100 - 1, y0 + thick - 1], fill=WHITE)  # barra superior
    d.rectangle([x0, 80, x0 + 70 - 1, 80 + thick - 1], fill=WHITE)  # barra media


def _draw_bar(d: ImageDraw.ImageDraw, angle_deg: float) -> None:
    cx, cy, half_len, half_w = 255.0, 185.0, 34.0, 4.0
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)  # dirección de la barra
    nx, ny = -uy, ux  # normal
    corners = [
        (cx + sx * half_len * ux + sw * half_w * nx, cy + sx * half_len * uy + sw * half_w * ny)
        for sx, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]
    d.polygon([(round(x), round(y)) for x, y in corners], fill=YELLOW)
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=BLUE)


def _draw_counter(d: ImageDraw.ImageDraw, value: int) -> None:
    y0, cell_w, cell_h, gap, x_start = 228, 16, 10, 4, 8
    for k in range(COUNTER_BITS):
        bit = (value >> (COUNTER_BITS - 1 - k)) & 1
        x = x_start + k * (cell_w + gap)
        d.rectangle([x, y0, x + cell_w - 1, y0 + cell_h - 1], fill=WHITE if bit else GRAY)


def make_frame(i: int) -> Image.Image:
    img = Image.new("P", (WIDTH, HEIGHT), BG)
    img.putpalette(_palette_bytes())
    d = ImageDraw.Draw(img)
    _draw_f(d, 40 + 2 * i)
    _draw_bar(d, DEGREES_PER_FRAME * i)
    d.rectangle([0, 0, 19, 19], fill=RED)
    d.rectangle([WIDTH - 20, HEIGHT - 20, WIDTH - 1, HEIGHT - 1], fill=GREEN)
    _draw_counter(d, i)
    return img


def make_frames(n: int = N_FRAMES) -> list[Image.Image]:
    return [make_frame(i) for i in range(n)]


def generate(out_dir: Path = DEFAULT_OUT) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILENAME
    frames = make_frames()
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=False,
        disposal=1,
    )
    return path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[0]) if argv else DEFAULT_OUT
    print(f"generada: {generate(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
