"""Módulo D: gaussiano 2D directo frente a separable, medido en la GPU (R14.3–R14.6).

Solo se miden las pasadas de desenfoque (`Processor.blur`) con consultas de
tiempo de GPU (`ctx.query(time=True)`, GL_TIME_ELAPSED). Cada medida envuelve
una ejecución completa: una pasada de blur_2d.frag o las dos de blur_1d.frag.
No conoce GLFW (D1): sirve para la tecla P, para --benchmark y para E4.
"""

from __future__ import annotations

import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

import moderngl

from src import kernels
from src.capture import RESULTS_DIR, TIMESTAMP_FORMAT, unique_path
from src.state import BORDER_NAMES

SIGMAS = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
WARMUP, REPEATS = 5, 50  # R14.3
QUICK_WARMUP, QUICK_REPEATS = 1, 5  # --quick
METHOD_2D, METHOD_SEPARABLE = "2D", "separable"
CSV_COLUMNS = (
    "fecha", "gpu", "W", "H", "borde", "sigma", "r", "metodo", "lecturas_px",
    "media_ms", "mediana_ms", "desv_ms", "min_ms", "aceleracion",
)


def _warn(msg: str) -> None:
    print(f"Aviso: {msg}", file=sys.stderr)


class GpuTimer:
    """Tiempo de GPU de una función con `ctx.query(time=True)`.

    Si las consultas fallan, avisa una sola vez y sigue sin tiempos (R14.6).
    Se reutiliza un único objeto Query: moderngl 5.13 no expone `release()` para
    las consultas, y crear una por medida acumularía objetos GL.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.available = True
        self._query = None

    def _disable(self, exc: Exception) -> None:
        self.available = False
        _warn(f"las consultas de tiempo de GPU no están disponibles ({exc}); se continúa sin tiempos.")

    def measure(self, fn):
        """Ejecuta fn() y devuelve (resultado, milisegundos de GPU o None)."""
        if not self.available:
            return fn(), None
        if self._query is None:
            try:
                self._query = self.ctx.query(time=True)
            except Exception as exc:  # noqa: BLE001 - moderngl no unifica sus excepciones
                self._disable(exc)
                return fn(), None
        with self._query:
            result = fn()
        try:
            ns = self._query.elapsed  # espera a que la GPU termine
        except Exception as exc:  # noqa: BLE001
            self._disable(exc)
            return result, None
        return result, ns / 1e6


def reads_per_pixel(r: int, method: str) -> int:
    """Lecturas de textura por píxel: (2r+1)² en 2D, 2·(2r+1) separable (R14.1)."""
    n = 2 * r + 1
    return n * n if method == METHOD_2D else 2 * n


def run_benchmark(
    ctx: moderngl.Context,
    processor,
    border: int = 0,
    sigmas=SIGMAS,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    now: datetime | None = None,
    log=print,
) -> list[dict]:
    """Mide 2D y separable para cada σ. Devuelve las filas del CSV, o [] si no hay tiempos."""
    timer = GpuTimer(ctx)
    w, h = processor.size
    gpu = ctx.info.get("GL_RENDERER", "desconocida")
    fecha = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for sigma in sigmas:
        weights = kernels.gaussian_1d(sigma)
        r = (len(weights) - 1) // 2
        pair = {}
        for method in (METHOD_2D, METHOD_SEPARABLE):
            separable = method == METHOD_SEPARABLE

            def run(separable=separable):
                processor.last_passes = []  # cada ejecución lista solo sus pasadas
                processor.blur(processor.source, weights, border, separable)

            for _ in range(warmup):
                run()
            ctx.finish()
            times = []
            for _ in range(repeats):
                _, ms = timer.measure(run)
                if ms is None:
                    return []  # GpuTimer ya avisó (R14.6)
                times.append(ms)
            row = {
                "fecha": fecha,
                "gpu": gpu,
                "W": w,
                "H": h,
                "borde": BORDER_NAMES[border],
                "sigma": sigma,
                "r": r,
                "metodo": method,
                "lecturas_px": reads_per_pixel(r, method),
                "media_ms": statistics.fmean(times),
                "mediana_ms": statistics.median(times),
                "desv_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                "min_ms": min(times),
            }
            pair[method] = row
            rows.append(row)
            log(f"  sigma={sigma:g} r={r} {method}: media {row['media_ms']:.3f} ms")
        sep = pair[METHOD_SEPARABLE]["media_ms"]
        speedup = pair[METHOD_2D]["media_ms"] / sep if sep > 0 else float("nan")
        for row in pair.values():
            row["aceleracion"] = speedup  # media 2D / media separable
    return rows


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if value == value else ""  # NaN → vacío
    return str(value)


def default_csv_path(now: datetime | None = None, results_dir: Path | None = None) -> Path:
    """`results/benchmark_AAAAMMDD-HHMMSS.csv` (R14.4)."""
    stamp = (now or datetime.now()).strftime(TIMESTAMP_FORMAT)
    return unique_path(Path(results_dir or RESULTS_DIR) / f"benchmark_{stamp}.csv")


def write_csv(rows: list[dict], path: Path) -> Path:
    """Escribe las filas con las columnas de CSV_COLUMNS; crea la carpeta si falta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([_fmt(row[c]) for c in CSV_COLUMNS])
    return path


def format_table(rows: list[dict]) -> str:
    """Tabla de resultados para la consola (R14.4)."""
    if not rows:
        return "(sin tiempos)"
    first = rows[0]
    lines = [
        f"GPU: {first['gpu']} | imagen {first['W']}x{first['H']} | borde {first['borde']}",
        f"{'sigma':>5} {'r':>3} {'método':<10} {'lect/px':>8} {'media':>9} {'mediana':>9} "
        f"{'desv':>8} {'mín':>9} {'2D/sep':>7}",
    ]
    for row in rows:
        lines.append(
            f"{row['sigma']:>5g} {row['r']:>3} {row['metodo']:<10} {row['lecturas_px']:>8} "
            f"{row['media_ms']:>9.3f} {row['mediana_ms']:>9.3f} {row['desv_ms']:>8.3f} "
            f"{row['min_ms']:>9.3f} {row['aceleracion']:>7.2f}"
        )
    lines.append("Tiempos en ms de GPU por ejecución del desenfoque (2D: 1 pasada; separable: 2).")
    return "\n".join(lines)


def benchmark_and_save(
    ctx: moderngl.Context,
    processor,
    border: int = 0,
    quick: bool = False,
    csv_path: Path | None = None,
    sigmas=SIGMAS,
) -> Path | None:
    """Ejecuta el benchmark, imprime la tabla y guarda el CSV. Devuelve la ruta o None sin tiempos."""
    warmup, repeats = (QUICK_WARMUP, QUICK_REPEATS) if quick else (WARMUP, REPEATS)
    # Solo caracteres de cp1252 en consola: con la salida redirigida a un archivo,
    # Windows usa esa codificación y «σ» lanzaría UnicodeEncodeError.
    print(f"Benchmark: sigma = {', '.join(f'{s:g}' for s in sigmas)}; {warmup} de calentamiento "
          f"y {repeats} repeticiones por método...")
    rows = run_benchmark(ctx, processor, border, sigmas, warmup, repeats)
    if not rows:
        _warn("benchmark sin tiempos: no se guarda el CSV.")
        return None
    print(format_table(rows))
    path = write_csv(rows, csv_path or default_csv_path())
    print(f"CSV del benchmark: {path}")
    return path
