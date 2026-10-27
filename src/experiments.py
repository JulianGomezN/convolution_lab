"""Experimentos del informe por lotes (--experiments, R17.1).

Reutiliza el Processor, la captura (display.frag → FBO RGBA8) y el benchmark, así
que cada PNG es igual a lo que se vería en la ventana con ese estado.

- E1: 4 bordes × {gaussiano σ = 8, Laplaciano, Sobel Gx}.
- E2: gaussiano y box de varianza equivalente con σ ∈ {1, 2, 4, 8}.
- E3: sharpen (preset 4) frente a unsharp con α ∈ {0.5, 1, 2} y σ ∈ {1, 2}.
- E4: CSV del benchmark 2D frente a separable (--quick reduce repeticiones).
- E5: Sobel Gx y Gy sin suavizado (presets 5 y 6) frente a las 11 etapas de Canny.

E2, E3 y E5 usan el borde replicate, que no oscurece el marco de la imagen; E1
recorre los 4. El resto de parámetros son los valores por defecto de AppState
(normalización activa, σ de Canny 1.4, umbral alto 0.3 y bajo 0.12, separable).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import moderngl
import numpy as np

from src import benchmark, capture, kernels, state as st
from src.pipeline import Processor
from src.render_pass import Quad
from src.shader_loader import load_program
from src.texture_loader import create_texture

MANIFEST_COLUMNS = (
    "archivo", "experimento", "modo", "kernel_etapa", "borde", "sigma", "perfil",
    "alfa", "umbral_alto", "umbral_bajo", "metodo",
)
EXPERIMENT_BORDER = st.BORDER_REPLICATE  # E2, E3 y E5
E1_SIGMA = 8.0
E2_SIGMAS = (1.0, 2.0, 4.0, 8.0)
E3_ALPHAS = (0.5, 1.0, 2.0)
E3_SIGMAS = (1.0, 2.0)


def _kernel(kernel_id: str, border: int) -> st.AppState:
    preset = kernels.PRESETS[kernel_id]
    return st.AppState(mode=st.MODE_KERNEL, kernel_id=kernel_id, display_map=preset.default_map, border=border)


def plan() -> list[tuple[str, st.AppState]]:
    """(experimento, estado) de cada captura, en orden."""
    items = []
    for border in range(len(st.BORDER_NAMES)):
        items.append(("E1", st.AppState(mode=st.MODE_BLUR, sigma=E1_SIGMA, border=border)))
        items.append(("E1", _kernel("7", border)))  # Laplaciano
        items.append(("E1", _kernel("5", border)))  # Sobel Gx
    b = EXPERIMENT_BORDER
    for sigma in E2_SIGMAS:
        for profile in (kernels.PROFILE_GAUSS, kernels.PROFILE_BOX):
            items.append(("E2", st.AppState(mode=st.MODE_BLUR, sigma=sigma, profile=profile, border=b)))
    items.append(("E3", _kernel("4", b)))  # sharpen
    for sigma in E3_SIGMAS:
        for alpha in E3_ALPHAS:
            items.append(("E3", st.AppState(mode=st.MODE_UNSHARP, sigma=sigma, alpha=alpha, border=b)))
    items.append(("E5", _kernel("5", b)))
    items.append(("E5", _kernel("6", b)))
    for stage in range(1, st.STAGE_COUNT + 1):
        items.append(("E5", st.AppState(mode=st.MODE_CANNY, stage=stage, border=b)))
    return items


def manifest_row(filename: str, experiment: str, state: st.AppState) -> dict:
    """Parámetros de una captura; las columnas que no aplican al modo quedan vacías."""
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(
        archivo=filename,
        experimento=experiment,
        modo=st.MODE_NAMES[state.mode],
        borde=st.BORDER_NAMES[state.border],
    )
    if state.mode == st.MODE_KERNEL:
        row["kernel_etapa"] = st.kernel_name(state)
        return row
    row["metodo"] = st.method_label(state)
    if state.mode == st.MODE_BLUR:
        row.update(kernel_etapa=kernels.PROFILE_NAMES[state.profile], sigma=f"{state.sigma:g}",
                   perfil=kernels.PROFILE_NAMES[state.profile])
    elif state.mode == st.MODE_UNSHARP:
        row.update(kernel_etapa="unsharp", sigma=f"{state.sigma:g}", perfil=kernels.PROFILE_NAMES[kernels.PROFILE_GAUSS],
                   alfa=f"{state.alpha:g}")
    else:
        row.update(kernel_etapa=f"{state.stage} {st.STAGE_NAMES[state.stage - 1]}", sigma=f"{state.canny_sigma:g}",
                   perfil=kernels.PROFILE_NAMES[kernels.PROFILE_GAUSS],
                   umbral_alto=f"{state.high:.2f}", umbral_bajo=f"{state.low:.2f}")
    return row


def default_out_dir(now: datetime | None = None, results_dir: Path | None = None) -> Path:
    """`results/experimentos_AAAAMMDD-HHMMSS/` (con sufijo si ya existe)."""
    stamp = (now or datetime.now()).strftime(capture.TIMESTAMP_FORMAT)
    return capture.unique_path(Path(results_dir or capture.RESULTS_DIR) / f"experimentos_{stamp}")


def write_manifest(rows: list[dict], path: Path, columns=MANIFEST_COLUMNS) -> Path:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)


def run_experiments(
    ctx: moderngl.Context,
    image: np.ndarray,
    out_dir: Path,
    quick: bool = False,
    bench_sigmas=benchmark.SIGMAS,
    extra_columns: dict | None = None,
) -> Path:
    """Genera E1–E5 en `out_dir` y devuelve la ruta de manifest.csv.

    Con `extra_columns` (p. ej. {"video": ruta, "frame": 0}, R11.5) el manifest
    añade esas columnas tras MANIFEST_COLUMNS con el mismo valor en cada fila; los
    PNG, sus nombres y E4_benchmark.csv no cambian (R11.2).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resources = []
    try:
        quad = Quad(ctx)
        resources.append(quad)
        prog_display = load_program(ctx, "display.frag")
        resources.append(prog_display)
        tex = create_texture(ctx, image)
        resources.append(tex)
        proc = Processor(ctx, quad, tex)
        resources.append(proc)

        rows = []
        items = plan()
        for i, (experiment, state) in enumerate(items, start=1):
            filename = f"{experiment}_{capture.describe(state)}.png"
            result, view = proc.run(state)
            rgb = capture.render_display(ctx, quad, prog_display, result, view)
            capture.save_png(rgb, out_dir / filename)
            rows.append(manifest_row(filename, experiment, state))
            print(f"[{i}/{len(items)}] {filename}")

        csv_name = "E4_benchmark.csv"
        csv_path = benchmark.benchmark_and_save(
            ctx, proc, border=EXPERIMENT_BORDER, quick=quick, csv_path=out_dir / csv_name, sigmas=bench_sigmas
        )
        if csv_path is not None:
            row = dict.fromkeys(MANIFEST_COLUMNS, "")
            row.update(archivo=csv_name, experimento="E4", modo="Benchmark",
                       kernel_etapa="gaussiano 2D frente a separable",
                       borde=st.BORDER_NAMES[EXPERIMENT_BORDER],
                       sigma=" ".join(f"{s:g}" for s in bench_sigmas),
                       perfil=kernels.PROFILE_NAMES[kernels.PROFILE_GAUSS],
                       metodo=f"{st.METHOD_2D} y {st.METHOD_SEPARABLE}")
            rows.append(row)
        columns = MANIFEST_COLUMNS
        if extra_columns:
            columns = MANIFEST_COLUMNS + tuple(extra_columns)
            for row in rows:
                row.update(extra_columns)
        manifest = write_manifest(rows, out_dir / "manifest.csv", columns)
    finally:
        for obj in reversed(resources):
            obj.release()
    print(f"Experimentos guardados en {out_dir} ({len(rows)} archivos, manifest.csv)")
    return manifest
