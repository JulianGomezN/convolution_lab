"""Capturas PNG del resultado visible (R15).

La captura dibuja display.frag, con el mismo canal, mapeo y ganancia que la
pantalla, en un FBO RGBA8 W×H con u_flip_y = 0: fbo.read() devuelve la fila 0
primero, que es la de arriba (D2), así que el PNG sale con la misma orientación
que la ventana (R15.3) sin invertir nada en la CPU.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from src import kernels, state as st
from src.framebuffers import RenderTarget
from src.pipeline import View
from src.render_pass import Quad, run_pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

MODE_SLUGS = {st.MODE_KERNEL: "kernel", st.MODE_BLUR: "desenfoque", st.MODE_UNSHARP: "unsharp", st.MODE_CANNY: "canny"}
MAP_SLUGS = ("recorte", "abs", "offset")


class CaptureError(Exception):
    """No se pudo escribir la captura (la app avisa y sigue, R15.4)."""


def slug(text: str) -> str:
    """Texto en minúsculas, sin acentos y con guiones: «Dirección» → «direccion»."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    out = "".join(c if c.isalnum() else "-" for c in ascii_text.lower())
    return "-".join(part for part in out.split("-") if part)


def describe(state: st.AppState) -> str:
    """`<modo>_<kernel|etapa>_<borde>_<params>` del estado, para nombres de archivo (R15.1)."""
    border = st.BORDER_NAMES[state.border]
    method = "sep" if state.separable else "2d"
    if state.mode == st.MODE_KERNEL:
        what = f"k{state.kernel_id}-{slug(st.kernel_name(state))}"
        params = f"norm-{'si' if state.normalize else 'no'}_vista-{MAP_SLUGS[state.display_map]}"
    elif state.mode == st.MODE_BLUR:
        what = slug(kernels.PROFILE_NAMES[state.profile])
        params = f"s{state.sigma:g}_{method}"
    elif state.mode == st.MODE_UNSHARP:
        what = "gaussiano"
        params = f"s{state.sigma:g}_a{state.alpha:g}_{method}"
    else:
        what = f"e{state.stage:02d}-{slug(st.STAGE_NAMES[state.stage - 1])}"
        params = f"s{state.canny_sigma:g}_alto{state.high:.2f}_bajo{state.low:.2f}_{method}"
    return f"{MODE_SLUGS[state.mode]}_{what}_{border}_{params}"


def capture_name(state: st.AppState, now: datetime) -> str:
    """`AAAAMMDD-HHMMSS_<modo>_<kernel|etapa>_<borde>_<params>.png` (R15.1)."""
    return f"{now.strftime(TIMESTAMP_FORMAT)}_{describe(state)}.png"


def unique_path(path: Path) -> Path:
    """`path` o, si ya existe (dos capturas en el mismo segundo), `nombre_2.png`, `nombre_3.png`…"""
    path = Path(path)
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


def render_display(
    ctx: moderngl.Context, quad: Quad, prog_display: moderngl.Program, tex: moderngl.Texture, view: View
) -> np.ndarray:
    """Dibuja `tex` con display.frag en un FBO RGBA8 del mismo tamaño y lo devuelve como H×W×3 uint8."""
    target = RenderTarget.create(ctx, tuple(tex.size), dtype="f1")  # W×H (R3.4)
    try:
        run_pass(quad, prog_display, target, {"u_src": tex}, view.uniforms(flip_y=0))
        return target.read()[..., :3].copy()
    finally:
        target.release()


def save_png(rgb: np.ndarray, path: Path) -> Path:
    """Guarda H×W×3 uint8 como PNG, creando la carpeta si falta (R15.2). Lanza CaptureError."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(path, format="PNG")
    except (OSError, ValueError) as exc:
        raise CaptureError(f"no se pudo escribir {path}: {exc}") from None
    return path
