"""Fuentes de frames para --video: animaciones con Pillow y vídeo con imageio-ffmpeg.

Cada frame se entrega como ndarray uint8 C-contiguo (H, W, 4), RGB del archivo
sin componer con fondo y alfa = 255 (R5.3), con la fila 0 arriba (D2). La CPU
solo decodifica y convierte a RGBA; nunca filtra ni opera entre vecinos (R1.5).

`imageio` no se importa en el nivel del módulo (R1.11): solo se comprueba su
presencia con importlib cuando la extensión lo pide.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.playback import sanitize_durations_ms, sanitize_fps
from src.texture_loader import PROJECT_ROOT, VIDEO_ONLY_SUFFIXES, ImageLoadError

PILLOW_SUFFIXES = (".gif", ".png", ".apng", ".webp")
FFMPEG_SUFFIXES = VIDEO_ONLY_SUFFIXES  # (".mp4", ".avi", ".mkv", ".mov", ".webm")
VIDEO_SUFFIXES = PILLOW_SUFFIXES + FFMPEG_SUFFIXES  # las 9 de R3.2, en este orden
VIDEOS_DIR = PROJECT_ROOT / "assets" / "videos"
BACKEND_PILLOW, BACKEND_FFMPEG = "Pillow", "imageio-ffmpeg"
INSTALL_HINT = r".venv\Scripts\python.exe -m pip install -r requirements-video.txt"

# Paquetes opcionales del Backend_Video: (módulo importable, nombre de distribución).
_VIDEO_PACKAGES = (("imageio", "imageio"), ("imageio_ffmpeg", "imageio-ffmpeg"))

# Errores con los que Pillow señala archivos vacíos, truncados o corruptos.
_PILLOW_ERRORS = (
    UnidentifiedImageError,
    Image.DecompressionBombError,
    OSError,
    SyntaxError,
    ValueError,
    EOFError,
)


def print_warning(msg: str) -> None:
    """Escribe un aviso en stderr tal cual (los mensajes ya empiezan por "Aviso:")."""
    print(msg, file=sys.stderr)


def available_videos(videos_dir: Path = VIDEOS_DIR) -> list[str]:
    """Nombres de los vídeos admitidos que hay en assets/videos/ (orden alfabético)."""
    try:
        return sorted(
            p.name
            for p in Path(videos_dir).iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        )
    except OSError:
        return []


class VideoOpenError(ImageLoadError):
    """No se pudo abrir el vídeo (código 2).

    Subclase de ImageLoadError para que cualquier manejador existente la trate
    igual. No llama a ImageLoadError.__init__ (listaría assets/images).
    """

    def __init__(self, path, reason: str, videos_dir: Path = VIDEOS_DIR):
        self.path = Path(path)
        self.reason = reason
        self.available = available_videos(videos_dir)
        listing = (
            "\n".join(f"  - {name}" for name in self.available)
            if self.available
            else "  (ninguno; genera el de prueba con: python tools\\make_test_video.py)"
        )
        Exception.__init__(
            self,
            f"No se pudo abrir el vídeo.\n"
            f"Ruta: {self.path}\n"
            f"Motivo: {reason}\n"
            f"Vídeos en assets\\videos:\n{listing}",
        )


class FrameReadError(Exception):
    """Un frame con índice > 0 no se pudo decodificar durante la reproducción (R3.6)."""

    def __init__(self, index: int, reason: str):
        self.index = int(index)
        self.reason = reason
        super().__init__(f"no se pudo decodificar el frame {self.index} ({reason})")


class FrameSource(Protocol):
    path: Path
    backend: str  # BACKEND_PILLOW o BACKEND_FFMPEG
    size: tuple[int, int]  # (W, H) del frame 0 (V4)
    n_frames: int | None  # None = desconocido (solo ffmpeg)
    durations: tuple[float, ...] | None  # segundos por frame; None = 1/fps constante
    fps: float  # FPS_Nominal ya saneado (R7.6)
    fps_warning: str | None  # aviso de FPS inválido, o None

    def open(self) -> None: ...

    def read(self, index: int) -> np.ndarray | None: ...

    def close(self) -> None: ...

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, *exc) -> None: ...


def _to_rgba(rgb) -> np.ndarray:
    """RGB (H, W, 3) → RGBA uint8 C-contiguo con alfa = 255 (R5.3, S3).

    Misma conversión que load_image (R5.5); copia canal a canal, sin vecinos (R1.5).
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = rgb[..., :3]
    out[..., 3] = 255
    return out


def _reason(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class PillowFrameSource:
    """Backend_Pillow: GIF, PNG/APNG y WebP (animados o de 1 frame)."""

    backend = BACKEND_PILLOW

    def __init__(self, path):
        self.path = Path(path)
        self.size: tuple[int, int] = (0, 0)
        self.n_frames: int | None = None
        self.durations: tuple[float, ...] | None = None
        self.fps: float = 0.0
        self.fps_warning: str | None = None
        self._img = None
        self._cache: tuple[int, np.ndarray] | None = None

    def open(self) -> None:
        """Abre el archivo, lee duraciones y decodifica el frame 0 (R2.1, R3.4, R7.7)."""
        try:
            img = Image.open(self.path)
        except _PILLOW_ERRORS as exc:
            raise VideoOpenError(self.path, f"no se pudo leer el archivo ({_reason(exc)})") from None
        self._img = img
        try:
            n = int(getattr(img, "n_frames", 1) or 1)
            raw = []
            for k in range(n):
                img.seek(k)
                raw.append(img.info.get("duration"))
            img.seek(0)
            frame0 = _to_rgba(img.convert("RGB"))
        except _PILLOW_ERRORS as exc:
            self.close()
            raise VideoOpenError(
                self.path, f"no se pudo decodificar el primer frame ({_reason(exc)})"
            ) from None
        self.n_frames = n
        self.durations = sanitize_durations_ms(raw)
        self.fps = n / sum(self.durations)
        self.fps_warning = None
        h, w = frame0.shape[:2]
        self.size = (w, h)
        self._cache = (0, frame0)

    def read(self, index: int) -> np.ndarray | None:
        """Frame `index` como RGBA (H, W, 4); None si está más allá del final."""
        index = int(index)
        if index < 0:
            raise ValueError(f"índice de frame negativo: {index}")
        if self._img is None:
            raise FrameReadError(index, "la fuente de frames está cerrada")
        if self.n_frames is not None and index >= self.n_frames:
            return None
        if self._cache is not None and self._cache[0] == index:
            return self._cache[1]
        try:
            self._img.seek(index)
            frame = _to_rgba(self._img.convert("RGB"))
        except _PILLOW_ERRORS as exc:
            raise FrameReadError(index, _reason(exc)) from None
        self._cache = (index, frame)
        return frame

    def close(self) -> None:
        """Libera el archivo (R12.5); idempotente (R12.6)."""
        img, self._img = self._img, None
        self._cache = None
        if img is not None:
            img.close()

    def __enter__(self) -> "PillowFrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def missing_video_packages() -> list[str]:
    """Nombres de distribución de los paquetes del Backend_Video que no se importan (R3.3)."""
    missing = []
    for module, dist in _VIDEO_PACKAGES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(dist)
    return missing


class ImageioFrameSource:
    """Backend_Video: MP4, AVI, MKV, MOV y WebM con imageio + imageio-ffmpeg.

    Lectura solo secuencial (V12): para retroceder se reabre el lector. Los
    frames intermedios no se convierten a RGBA (VD4).
    """

    backend = BACKEND_FFMPEG

    def __init__(self, path):
        self.path = Path(path)
        self.size: tuple[int, int] = (0, 0)
        self.n_frames: int | None = None
        self.durations: tuple[float, ...] | None = None  # 1/fps constante (R7.5)
        self.fps: float = 0.0
        self.fps_warning: str | None = None
        self._iio = None  # módulo imageio.v2, importado en open() (R1.11)
        self._reader = None
        self._next = 0  # índice del próximo frame que dará get_next_data()
        self._cache: tuple[int, np.ndarray] | None = None

    def _new_reader(self):
        return self._iio.get_reader(str(self.path), format="FFMPEG")

    def open(self) -> None:
        """Crea el lector, lee metadatos y decodifica el frame 0 (R2.2, R3.4, R7.6)."""
        try:
            import imageio.v2 as iio  # import perezoso (R1.11)
            import imageio_ffmpeg  # noqa: F401 - asegura el ejecutable de ffmpeg
        except ImportError as exc:
            raise VideoOpenError(self.path, f"no se pudo cargar imageio-ffmpeg ({_reason(exc)})") from None
        self._iio = iio
        try:
            self._reader = self._new_reader()
            meta = self._reader.get_meta_data() or {}
            try:
                n = int(self._reader.count_frames())
                self.n_frames = n if n > 0 else None
            except Exception:  # noqa: BLE001 - total desconocido; se descubre en el EOF
                self.n_frames = None
            self.fps, self.fps_warning = sanitize_fps(meta.get("fps"))
            frame0 = _to_rgba(self._reader.get_next_data())
        except Exception as exc:  # noqa: BLE001 - cualquier fallo al abrir → código 2
            self.close()
            raise VideoOpenError(
                self.path, f"no se pudo decodificar el primer frame ({_reason(exc)})"
            ) from None
        self._next = 1
        h, w = frame0.shape[:2]
        self.size = (w, h)
        self._cache = (0, frame0)

    def read(self, index: int) -> np.ndarray | None:
        """Frame `index` como RGBA (H, W, 4); None si está más allá del final."""
        index = int(index)
        if index < 0:
            raise ValueError(f"índice de frame negativo: {index}")
        if self._reader is None:
            raise FrameReadError(index, "la fuente de frames está cerrada")
        if self.n_frames is not None and index >= self.n_frames:
            return None
        if self._cache is not None and self._cache[0] == index:
            return self._cache[1]
        try:
            if index < self._next - 1:  # retroceso: reabrir (V12)
                reader, self._reader = self._reader, None
                reader.close()
                self._reader = self._new_reader()
                self._next = 0
            raw = None
            while self._next <= index:
                try:
                    raw = self._reader.get_next_data()
                except IndexError:  # EOF: se fija el total real
                    if self.n_frames is None or self._next < self.n_frames:
                        self.n_frames = self._next
                    return None
                self._next += 1
            frame = _to_rgba(raw)
        except FrameReadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FrameReadError(index, _reason(exc)) from None
        self._cache = (index, frame)
        return frame

    def close(self) -> None:
        """Termina ffmpeg y libera el archivo (R12.4, R12.5); idempotente (R12.6)."""
        reader, self._reader = self._reader, None
        self._cache = None
        if reader is not None:
            reader.close()

    def __enter__(self) -> "ImageioFrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _ffmpeg_source(path: Path) -> FrameSource:
    """Crea la fuente del Backend_Video para `path`."""
    return ImageioFrameSource(path)


def open_source(path, warn: Callable[[str], None] = print_warning) -> FrameSource:
    """Abre `path` con el backend que corresponde a su extensión y decodifica el frame 0.

    Lanza VideoOpenError (código 2) en los casos de R3.1–R3.4. Si open() falla,
    la fuente se cierra antes de propagar el error. Avisa una vez si el FPS
    declarado no es válido (R7.6).
    """
    path = Path(path)
    if not path.exists():
        raise VideoOpenError(path, "el archivo no existe")  # R3.1
    if not path.is_file():
        raise VideoOpenError(path, "la ruta no es un archivo")  # R3.1
    suffix = path.suffix.lower()
    if suffix in PILLOW_SUFFIXES:
        source: FrameSource = PillowFrameSource(path)
    elif suffix in FFMPEG_SUFFIXES:
        missing = missing_video_packages()  # R3.3
        if missing:
            raise VideoOpenError(
                path,
                f"falta(n) {', '.join(missing)}; instálalo(s) con "
                f"pip install -r requirements-video.txt ({INSTALL_HINT})",
            )
        source = _ffmpeg_source(path)
    else:
        raise VideoOpenError(  # R3.2
            path,
            f"extensión «{path.suffix}» no admitida; usa una de: " + ", ".join(VIDEO_SUFFIXES),
        )
    try:
        source.open()  # R3.4
    except BaseException:
        source.close()
        raise
    if source.fps_warning:
        warn(source.fps_warning)  # R7.6
    return source


def _cp1252(text: str) -> str:
    return text.encode("cp1252", "replace").decode("cp1252")


def describe(source: FrameSource) -> str:
    """Línea de R4.6, solo con caracteres de cp1252.

    Ejemplo: "Vídeo: demo.gif | 320x240 | 60 frames | 20.00 fps | backend Pillow".
    """
    w, h = source.size
    frames = f"{source.n_frames} frames" if source.n_frames is not None else "frames desconocido"
    return _cp1252(
        f"Vídeo: {source.path.name} | {w}x{h} | {frames} | {source.fps:.2f} fps | backend {source.backend}"
    )


def safe_close(source: FrameSource | None, warn: Callable[[str], None] = print_warning) -> None:
    """Cierra la fuente; si el cierre lanza, avisa y no relanza (R12.7)."""
    if source is None:
        return
    try:
        source.close()
    except Exception as exc:  # noqa: BLE001 - el cierre nunca cambia el código de salida
        warn(f"Aviso: no se pudo cerrar el vídeo ({_reason(exc)})")
