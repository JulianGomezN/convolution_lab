"""Benchmark 2D frente a separable con consultas de tiempo de GPU (R14.3–R14.7)."""

import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import main
from src import app as app_mod, benchmark
from src.pipeline import Processor
from src.texture_loader import create_texture

ROOT = Path(__file__).resolve().parent.parent


def test_constants_match_requirements():
    assert benchmark.SIGMAS == (1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
    assert (benchmark.WARMUP, benchmark.REPEATS) == (5, 50)
    assert benchmark.CSV_COLUMNS == (
        "fecha", "gpu", "W", "H", "borde", "sigma", "r", "metodo", "lecturas_px",
        "media_ms", "mediana_ms", "desv_ms", "min_ms", "aceleracion",
    )


def test_reads_per_pixel():
    assert benchmark.reads_per_pixel(3, benchmark.METHOD_2D) == 49
    assert benchmark.reads_per_pixel(3, benchmark.METHOD_SEPARABLE) == 14
    assert benchmark.reads_per_pixel(30, benchmark.METHOD_2D) == 61 * 61


def test_default_csv_path(tmp_path):
    p = benchmark.default_csv_path(datetime(2025, 1, 2, 3, 4, 5), results_dir=tmp_path)
    assert p == tmp_path / "benchmark_20250102-030405.csv"


# --- GPU --------------------------------------------------------------------


@pytest.fixture(scope="module")
def processor(ctx, quad):
    img = np.random.default_rng(41).integers(0, 256, size=(24, 32, 4), dtype=np.uint8)
    tex = create_texture(ctx, img)
    proc = Processor(ctx, quad, tex)
    yield proc
    proc.release()
    tex.release()


def test_gpu_timer_measures(ctx, processor):
    timer = benchmark.GpuTimer(ctx)
    calls = []
    result, ms = timer.measure(lambda: calls.append(1) or "ok")
    assert result == "ok" and calls == [1]
    assert ms is not None and ms >= 0.0
    _, ms2 = timer.measure(lambda: processor.blur(processor.source, (0.25, 0.5, 0.25), 0))
    assert ms2 is not None and ms2 >= 0.0


def test_run_benchmark_rows(ctx, processor):
    rows = benchmark.run_benchmark(ctx, processor, border=1, sigmas=(1.0,), warmup=1, repeats=2, log=lambda m: None)
    assert [r["metodo"] for r in rows] == [benchmark.METHOD_2D, benchmark.METHOD_SEPARABLE]
    for row in rows:
        assert set(row) == set(benchmark.CSV_COLUMNS)
        assert (row["W"], row["H"], row["borde"], row["sigma"], row["r"]) == (32, 24, "replicate", 1.0, 3)
        for key in ("media_ms", "mediana_ms", "desv_ms", "min_ms"):
            assert row[key] >= 0.0
        assert row["min_ms"] <= row["mediana_ms"]
        assert row["gpu"] == ctx.info["GL_RENDERER"]
    assert rows[0]["lecturas_px"] == 49 and rows[1]["lecturas_px"] == 14
    assert rows[0]["aceleracion"] == rows[1]["aceleracion"]
    assert processor.last_passes == ["blur H", "blur V"]  # solo pasadas de desenfoque (R14.3)


def test_write_csv(ctx, processor, tmp_path):
    rows = benchmark.run_benchmark(ctx, processor, sigmas=(1.0, 2.0), warmup=1, repeats=2, log=lambda m: None)
    path = benchmark.write_csv(rows, tmp_path / "sub" / "b.csv")
    with path.open(encoding="utf-8") as f:
        data = list(csv.reader(f))
    assert tuple(data[0]) == benchmark.CSV_COLUMNS
    assert len(data) == 1 + 4
    assert [r[7] for r in data[1:]] == ["2D", "separable", "2D", "separable"]
    assert all(float(r[9]) >= 0 for r in data[1:])
    table = benchmark.format_table(rows)
    assert "separable" in table and "2D/sep" in table and "32x24" in table


class _NoQueryCtx:
    """Contexto real salvo ctx.query, que falla (R14.6)."""

    def __init__(self, ctx, fail_on="create"):
        self._ctx = ctx
        self.fail_on = fail_on

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def query(self, **kw):
        if self.fail_on == "create":
            raise RuntimeError("consultas simuladas no disponibles")
        return _BrokenQuery()


class _BrokenQuery:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def elapsed(self):
        raise RuntimeError("resultado de consulta simulado no disponible")


@pytest.mark.parametrize("fail_on", ["create", "elapsed"])
def test_query_failure_warns_and_continues(ctx, processor, capsys, fail_on):
    fake = _NoQueryCtx(ctx, fail_on)
    timer = benchmark.GpuTimer(fake)
    result, ms = timer.measure(lambda: 7)
    assert (result, ms) == (7, None)
    assert timer.available is False
    assert timer.measure(lambda: 8) == (8, None)  # sigue funcionando, sin tiempos
    err = capsys.readouterr().err
    assert err.count("no están disponibles") == 1  # avisa una sola vez
    assert benchmark.run_benchmark(fake, processor, sigmas=(1.0,), warmup=0, repeats=1, log=lambda m: None) == []


def test_benchmark_and_save_without_queries(ctx, processor, tmp_path, capsys):
    path = benchmark.benchmark_and_save(_NoQueryCtx(ctx), processor, csv_path=tmp_path / "b.csv", sigmas=(1.0,))
    assert path is None and not (tmp_path / "b.csv").exists()
    assert "no se guarda el CSV" in capsys.readouterr().err


# --- CLI ----------------------------------------------------------------------


def test_benchmark_missing_image_exits_2(tmp_path, capsys):
    assert main.main(["--benchmark", "--image", str(tmp_path / "no.png")]) == 2
    assert "no existe" in capsys.readouterr().err


def test_benchmark_without_context_exits_3(monkeypatch, capsys):
    def fail(**kw):
        raise RuntimeError("standalone simulado no disponible")

    monkeypatch.setattr(app_mod.moderngl, "create_standalone_context", fail)
    monkeypatch.setattr(app_mod.glfw, "init", lambda: False)
    monkeypatch.setattr(app_mod.glfw, "terminate", lambda: None)
    assert main.main(["--benchmark", "--image", str(ROOT / "assets" / "images" / "escalon.png")]) == 3
    err = capsys.readouterr().err
    assert "No se pudo crear un contexto OpenGL 3.3 sin ventana" in err
    assert "standalone simulado no disponible" in err
    assert "Traceback" not in err


def test_benchmark_and_experiments_are_exclusive(capsys):
    with pytest.raises(SystemExit) as info:
        main.main(["--benchmark", "--experiments"])
    assert info.value.code == 2
    assert "no se permite junto con" in capsys.readouterr().err


def test_benchmark_cli_subprocess(ctx, tmp_path):
    """--benchmark en otro proceso (contexto propio): tabla, CSV en results/ y código 0 (R14.7)."""
    img = tmp_path / "pequena.png"
    Image.fromarray(np.random.default_rng(42).integers(0, 256, size=(12, 16, 3), dtype=np.uint8)).save(img)
    results = ROOT / "results"
    before = set(results.glob("benchmark_*.csv"))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--benchmark", "--quick", "--image", str(img)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=300,
    )
    created = set(results.glob("benchmark_*.csv")) - before
    try:
        assert proc.returncode == 0, proc.stderr
        assert "GL_RENDERER" in proc.stdout and "2D/sep" in proc.stdout
        assert len(created) == 1
        path = created.pop()
        assert re.fullmatch(r"benchmark_\d{8}-\d{6}(_\d+)?\.csv", path.name)
        with path.open(encoding="utf-8") as f:
            data = list(csv.reader(f))
        assert tuple(data[0]) == benchmark.CSV_COLUMNS
        assert len(data) == 1 + 2 * len(benchmark.SIGMAS)
        assert {r[2] for r in data[1:]} == {"16"} and {r[3] for r in data[1:]} == {"12"}
        created.add(path)
    finally:
        for p in created:
            p.unlink()


# --- Vídeo (R11.1, R11.3, R11.4, R11.6, R13.11) -------------------------------


def _make_gif(path: Path, size=(16, 12), n=3) -> Path:
    """Animación pequeña con frames distintos (W×H = size)."""
    rng = np.random.default_rng(43)
    frames = [Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)) for _ in range(n)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


def _results_listing() -> set:
    results = ROOT / "results"
    return set(results.rglob("*")) if results.exists() else set()


def test_benchmark_and_save_extra_columns(ctx, processor, tmp_path):
    extra = {"video": str(tmp_path / "anim.gif"), "frame": 0}
    path = benchmark.benchmark_and_save(
        ctx, processor, quick=True, csv_path=tmp_path / "v.csv", sigmas=(1.0,), extra_columns=extra
    )
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == benchmark.CSV_COLUMNS + benchmark.VIDEO_COLUMNS
    assert len(rows) == 2
    assert all(r["video"] == extra["video"] and r["frame"] == "0" for r in rows)

    plain = benchmark.benchmark_and_save(ctx, processor, quick=True, csv_path=tmp_path / "i.csv", sigmas=(1.0,))
    with plain.open(encoding="utf-8") as f:
        assert tuple(next(csv.reader(f))) == benchmark.CSV_COLUMNS


@pytest.mark.parametrize("batch", ["--benchmark", "--experiments"])
@pytest.mark.parametrize("name", ["no_existe.gif", "archivo.txt"])
def test_video_batch_open_error_exits_2(tmp_path, capsys, batch, name):
    """Ruta inexistente o extensión no admitida: código 2 y nada nuevo en results/ (R11.6)."""
    target = tmp_path / name
    if name.endswith(".txt"):
        target.write_text("no es un vídeo", encoding="utf-8")
    before = _results_listing()
    assert main.main(["--video", str(target), batch]) == 2
    assert _results_listing() == before
    err = capsys.readouterr().err
    assert "Error de vídeo" in err and "Traceback" not in err


def test_video_benchmark_cli_subprocess(ctx, tmp_path):
    """--video <gif> --benchmark --quick: código 0, ruta y frame 0 en consola y en cada fila (R11.1, R11.4)."""
    gif = _make_gif(tmp_path / "anim.gif")
    results = ROOT / "results"
    before = set(results.glob("benchmark_*.csv"))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--video", str(gif), "--benchmark", "--quick"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=300,
    )
    created = set(results.glob("benchmark_*.csv")) - before
    try:
        assert proc.returncode == 0, proc.stderr
        assert str(gif) in proc.stdout and "| frame 0" in proc.stdout
        assert len(created) == 1
        with next(iter(created)).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert tuple(rows[0]) == benchmark.CSV_COLUMNS + benchmark.VIDEO_COLUMNS
        assert len(rows) == 2 * len(benchmark.SIGMAS)
        assert all(r["video"] == str(gif) and r["frame"] == "0" for r in rows)
        assert {(r["W"], r["H"]) for r in rows} == {("16", "12")}
    finally:
        for p in created:
            p.unlink()
