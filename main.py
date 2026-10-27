"""Punto de entrada:
python main.py [--image RUTA | --video RUTA] [--benchmark | --experiments] [--quick].

--video reproduce un vídeo o animación (excluyente con --image).

Códigos de salida: 0 normal, 1 error inesperado, 2 error de imagen o de vídeo,
3 error de OpenGL/GLFW, 4 error de shader, 130 Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from src import app
from src.app import GLContextError
from src.shader_loader import ShaderError
from src.texture_loader import ImageLoadError
from src.video import FFMPEG_SUFFIXES, PILLOW_SUFFIXES, VIDEO_SUFFIXES, VideoOpenError

PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
FALLBACK_IMAGE = "assets/images/orientacion.png"

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_IMAGE = 2
EXIT_GL = 3
EXIT_SHADER = 4
EXIT_INTERRUPTED = 130

# argparse no trae traducción al español; se traducen sus textos fijos para que
# toda la salida visible quede en español.
# Riesgo: `argparse._` es un nombre privado (el gettext que argparse usa
# internamente). Si una versión futura de Python lo renombra o deja de usarlo,
# este parche no falla: la ayuda simplemente vuelve al inglés. Lo detecta
# tests/test_main.py::test_help_is_spanish.
_ARGPARSE_ES = {
    "usage: ": "uso: ",
    "options": "opciones",
    "show this help message and exit": "muestra esta ayuda y termina",
    "%(prog)s: error: %(message)s\n": "%(prog)s: error: %(message)s\n",
    "unrecognized arguments: %s": "argumentos no reconocidos: %s",
    "expected one argument": "se esperaba un argumento",
    "the following arguments are required: %s": "faltan los argumentos obligatorios: %s",
    "not allowed with argument %s": "no se permite junto con el argumento %s",
    "argument %(argument_name)s: %(message)s": "argumento %(argument_name)s: %(message)s",
}
_argparse_gettext = argparse._
argparse._ = lambda text: _ARGPARSE_ES.get(text, _argparse_gettext(text))


def default_image() -> Path:
    """Imagen por defecto de config/settings.json (R4.2); si falla, orientacion.png."""
    try:
        rel = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))["default_image"]
        if not isinstance(rel, str):
            raise TypeError("default_image debe ser una cadena")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Aviso: no se pudo leer default_image de {SETTINGS_PATH} ({exc}); se usa {FALLBACK_IMAGE}.",
              file=sys.stderr)
        rel = FALLBACK_IMAGE
    return PROJECT_ROOT / rel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Laboratorio interactivo de máscaras de convolución (OpenGL 3.3 core).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image",
        metavar="RUTA",
        help="imagen a cargar (por defecto: default_image de config/settings.json)",
    )
    pillow = ", ".join(s for s in VIDEO_SUFFIXES if s in PILLOW_SUFFIXES)
    ffmpeg = ", ".join(s for s in VIDEO_SUFFIXES if s in FFMPEG_SUFFIXES)
    source.add_argument(
        "--video",
        metavar="RUTA",
        help=f"vídeo o animación a reproducir (RUTA); excluyente con --image. "
        f"Extensiones: {pillow} (Pillow) y {ffmpeg} (requieren requirements-video.txt)",
    )
    batch = parser.add_mutually_exclusive_group()
    batch.add_argument(
        "--benchmark",
        action="store_true",
        help="mide el gaussiano 2D directo frente al separable sin abrir ventana "
        "y guarda results/benchmark_<fecha>.csv",
    )
    batch.add_argument(
        "--experiments",
        action="store_true",
        help="genera los experimentos E1-E5 y manifest.csv en results/experimentos_<fecha>/",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="con --benchmark o --experiments: menos repeticiones del benchmark (1 de calentamiento y 5 medidas)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.video is not None:
        path = Path(args.video)
    else:
        path = Path(args.image) if args.image else default_image()

    try:
        # Con --video, el benchmark y los experimentos usan el frame 0 (R11).
        is_video = args.video is not None
        if args.benchmark:
            return app.run_benchmark(path, quick=args.quick, video=is_video)
        if args.experiments:
            return app.run_experiments(path, quick=args.quick, video=is_video)
        if args.video is not None:
            return app.run_video(path)
        return app.run(path)
    except VideoOpenError as exc:  # subclase de ImageLoadError: va antes
        print(f"Error de vídeo: {exc}", file=sys.stderr)
        return EXIT_IMAGE
    except ImageLoadError as exc:
        print(f"Error de imagen: {exc}", file=sys.stderr)
        return EXIT_IMAGE
    except GLContextError as exc:
        print(f"Error de OpenGL: {exc}", file=sys.stderr)
        return EXIT_GL
    except ShaderError as exc:
        print(f"Error de shader: {exc}", file=sys.stderr)
        return EXIT_SHADER
    except KeyboardInterrupt:
        print("Interrumpido por el usuario (Ctrl+C).", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception:  # noqa: BLE001 - manejador de nivel superior
        print("Error inesperado:", file=sys.stderr)
        traceback.print_exc()
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
