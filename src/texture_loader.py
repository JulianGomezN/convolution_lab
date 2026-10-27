"""Carga de imágenes con Pillow y subida a la GPU como textura.

Convención (D2): el array devuelto es img[y, x] con la fila 0 arriba, y se sube
tal cual, así que el texel (x, y) de la textura es el píxel img[y, x].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "images"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
# Extensiones que solo se abren con --video (R2.5); src/video.py las importa de aquí.
VIDEO_ONLY_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".webm")


class ImageLoadError(Exception):
    """Error al cargar la imagen de entrada (código de salida 2)."""

    def __init__(self, path, reason: str, assets_dir: Path = ASSETS_DIR):
        self.path = Path(path)
        self.reason = reason
        self.available = available_images(assets_dir)
        listing = (
            "\n".join(f"  - {name}" for name in self.available)
            if self.available
            else "  (ninguna; genera las de prueba con: python tools\\make_test_images.py)"
        )
        super().__init__(
            f"No se pudo cargar la imagen.\n"
            f"Ruta: {self.path}\n"
            f"Motivo: {reason}\n"
            f"Imágenes disponibles en {assets_dir}:\n{listing}"
        )


def available_images(assets_dir: Path = ASSETS_DIR) -> list[str]:
    """Nombres de las imágenes que hay en assets/images/ (orden alfabético)."""
    try:
        return sorted(
            p.name
            for p in Path(assets_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    except OSError:
        return []


def load_image(path) -> np.ndarray:
    """Carga una imagen como ndarray uint8 H×W×4 (RGBA) con alfa = 255 (R4.1, R4.5).

    No aplica la orientación EXIF (supuesto S9). Lanza ImageLoadError si el archivo
    no existe, no es legible o no es una imagen válida (R4.3), o si la extensión es
    de vídeo (R2.5). Un GIF/APNG/WebP animado carga su frame 0 (R2.4).
    """
    path = Path(path)
    if path.suffix.lower() in VIDEO_ONLY_SUFFIXES:
        raise ImageLoadError(path, "es un archivo de vídeo; ábrelo con --video RUTA")
    if not path.exists():
        raise ImageLoadError(path, "el archivo no existe")
    if not path.is_file():
        raise ImageLoadError(path, "la ruta no es un archivo")
    try:
        with Image.open(path) as img:
            img.load()
            rgb = img.convert("RGB")
    except UnidentifiedImageError:
        raise ImageLoadError(path, "no es una imagen válida o el formato no es compatible") from None
    except Image.DecompressionBombError as exc:
        raise ImageLoadError(path, f"la imagen es demasiado grande ({exc})") from None
    except (OSError, SyntaxError, ValueError) as exc:
        # Pillow señala algunos archivos corruptos con SyntaxError/ValueError.
        raise ImageLoadError(path, f"no se pudo leer el archivo ({exc})") from None

    # Se ignora el alfa de entrada: RGB + alfa opaco (S3, R4.5).
    arr = np.empty((rgb.height, rgb.width, 4), dtype=np.uint8)
    arr[..., :3] = np.asarray(rgb, dtype=np.uint8)
    arr[..., 3] = 255
    return arr


def create_texture(ctx, arr: np.ndarray):
    """Sube un array H×W×4 a una textura W×H.

    uint8 → RGBA8 (entrada normal); float32 → RGBA32F (tests de precisión).
    Filtro NEAREST, aunque texelFetch ignora el filtrado.
    """
    import moderngl

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"se esperaba un array H×W×4, se recibió {arr.shape}")
    if arr.dtype == np.uint8:
        dtype = "f1"
    elif arr.dtype == np.float32:
        dtype = "f4"
    else:
        raise ValueError(f"tipo de array no compatible: {arr.dtype}")
    h, w = arr.shape[:2]
    tex = ctx.texture((w, h), 4, np.ascontiguousarray(arr).tobytes(), dtype=dtype)
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return tex
