"""Experimentos por lotes --experiments (R17.1)."""

import csv
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src import experiments, state as st

ROOT = Path(__file__).resolve().parent.parent
# E1: 4 bordes × 3 filtros; E2: 4 σ × 2 perfiles; E3: sharpen + 3 α × 2 σ; E5: Gx, Gy + 11 etapas.
EXPECTED_PNG = {"E1": 12, "E2": 8, "E3": 7, "E5": 13}


def test_plan_counts_and_unique_names():
    items = experiments.plan()
    assert Counter(e for e, _ in items) == EXPECTED_PNG
    names = [f"{e}_{experiments.capture.describe(s)}.png" for e, s in items]
    assert len(set(names)) == len(names)


def test_plan_contents():
    items = experiments.plan()
    e1 = [s for e, s in items if e == "E1"]
    assert {s.border for s in e1} == {0, 1, 2, 3}
    assert sum(s.mode == st.MODE_BLUR and s.sigma == 8.0 for s in e1) == 4
    assert sorted(s.kernel_id for s in e1 if s.mode == st.MODE_KERNEL) == ["5"] * 4 + ["7"] * 4
    e2 = [s for e, s in items if e == "E2"]
    assert {(s.profile, s.sigma) for s in e2} == {(p, x) for p in ("gauss", "box") for x in (1.0, 2.0, 4.0, 8.0)}
    e3 = [s for e, s in items if e == "E3"]
    assert [s.kernel_id for s in e3 if s.mode == st.MODE_KERNEL] == ["4"]
    assert {(s.sigma, s.alpha) for s in e3 if s.mode == st.MODE_UNSHARP} == {
        (x, a) for x in (1.0, 2.0) for a in (0.5, 1.0, 2.0)
    }
    e5 = [s for e, s in items if e == "E5"]
    assert [s.kernel_id for s in e5 if s.mode == st.MODE_KERNEL] == ["5", "6"]
    assert [s.stage for s in e5 if s.mode == st.MODE_CANNY] == list(range(1, 12))


def test_default_out_dir(tmp_path):
    d = experiments.default_out_dir(datetime(2025, 5, 6, 7, 8, 9), results_dir=tmp_path)
    assert d == tmp_path / "experimentos_20250506-070809"


def test_run_experiments_on_gpu(ctx, tmp_path):
    img = np.random.default_rng(51).integers(0, 256, size=(12, 16, 4), dtype=np.uint8)
    img[..., 3] = 255
    out = tmp_path / "exp"
    manifest = experiments.run_experiments(ctx, img, out, quick=True, bench_sigmas=(1.0,))
    assert manifest == out / "manifest.csv"
    pngs = sorted(out.glob("*.png"))
    assert Counter(p.name.split("_")[0] for p in pngs) == EXPECTED_PNG
    for p in pngs:
        with Image.open(p) as im:
            assert im.size == (16, 12)  # W×H (R3.4)
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == experiments.MANIFEST_COLUMNS
    assert len(rows) == len(pngs) + 1  # + E4_benchmark.csv
    assert {r["archivo"] for r in rows} == {p.name for p in pngs} | {"E4_benchmark.csv"}
    assert all((out / r["archivo"]).exists() for r in rows)
    for r in rows:
        assert r["archivo"].startswith(r["experimento"] + "_")
    e4 = next(r for r in rows if r["experimento"] == "E4")
    assert e4["metodo"] == "2D y separable"
    with (out / "E4_benchmark.csv").open(encoding="utf-8") as f:
        assert len(list(csv.reader(f))) == 1 + 2
    canny = [r for r in rows if r["modo"] == "Canny"]
    assert len(canny) == 11 and all(r["umbral_alto"] == "0.30" and r["umbral_bajo"] == "0.12" for r in canny)
    unsharp = [r for r in rows if r["modo"] == "Unsharp"]
    assert {r["alfa"] for r in unsharp} == {"0.5", "1", "2"}
    kernel_rows = [r for r in rows if r["modo"] == "Kernel"]
    assert all(r["sigma"] == "" and r["metodo"] == "" for r in kernel_rows)


def test_experiments_cli_subprocess(ctx, tmp_path):
    """--experiments --quick en otro proceso: results/experimentos_<fecha>/ con manifest.csv."""
    img = tmp_path / "pequena.png"
    Image.fromarray(np.random.default_rng(52).integers(0, 256, size=(10, 14, 3), dtype=np.uint8)).save(img)
    results = ROOT / "results"
    before = set(results.glob("experimentos_*"))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--experiments", "--quick", "--image", str(img)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=600,
    )
    created = set(results.glob("experimentos_*")) - before
    try:
        assert proc.returncode == 0, proc.stderr
        assert len(created) == 1
        out = next(iter(created))
        assert re.fullmatch(r"experimentos_\d{8}-\d{6}(_\d+)?", out.name)
        assert (out / "manifest.csv").exists() and (out / "E4_benchmark.csv").exists()
        assert len(list(out.glob("*.png"))) == sum(EXPECTED_PNG.values())
        assert "manifest.csv" in proc.stdout
    finally:
        for d in created:
            shutil.rmtree(d)


# --- Vídeo (R11.2, R11.3, R11.5, R13.11) ----------------------------------------


def _make_gif(path: Path, size=(14, 10), n=3) -> Path:
    """Animación pequeña con frames distintos (W×H = size)."""
    rng = np.random.default_rng(53)
    frames = [Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)) for _ in range(n)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


def test_run_experiments_extra_columns(ctx, tmp_path):
    """Con extra_columns: mismas capturas y columnas + video, frame; sin él, MANIFEST_COLUMNS exactas."""
    img = np.random.default_rng(54).integers(0, 256, size=(10, 14, 4), dtype=np.uint8)
    img[..., 3] = 255
    extra = {"video": str(tmp_path / "anim.gif"), "frame": 0}
    with_v = experiments.run_experiments(ctx, img, tmp_path / "v", quick=True, bench_sigmas=(1.0,),
                                         extra_columns=extra)
    plain = experiments.run_experiments(ctx, img, tmp_path / "i", quick=True, bench_sigmas=(1.0,))
    with with_v.open(encoding="utf-8") as f:
        rows_v = list(csv.DictReader(f))
    with plain.open(encoding="utf-8") as f:
        rows_i = list(csv.DictReader(f))
    assert tuple(rows_v[0]) == experiments.MANIFEST_COLUMNS + ("video", "frame")
    assert tuple(rows_i[0]) == experiments.MANIFEST_COLUMNS
    assert all(r["video"] == extra["video"] and r["frame"] == "0" for r in rows_v)
    assert [r["archivo"] for r in rows_v] == [r["archivo"] for r in rows_i]
    assert sorted(p.name for p in (tmp_path / "v").iterdir()) == sorted(p.name for p in (tmp_path / "i").iterdir())


def test_video_experiments_cli_subprocess(ctx, tmp_path):
    """--video <gif> --experiments --quick: código 0, ruta y frame 0 en consola y en el manifest (R11.2, R11.5)."""
    gif = _make_gif(tmp_path / "anim.gif")
    results = ROOT / "results"
    before = set(results.glob("experimentos_*"))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--video", str(gif), "--experiments", "--quick"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=600,
    )
    created = set(results.glob("experimentos_*")) - before
    try:
        assert proc.returncode == 0, proc.stderr
        assert str(gif) in proc.stdout and "| frame 0" in proc.stdout
        assert len(created) == 1
        out = next(iter(created))
        pngs = list(out.glob("*.png"))
        assert len(pngs) == sum(EXPECTED_PNG.values())
        with Image.open(pngs[0]) as im:
            assert im.size == (14, 10)
        with (out / "manifest.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert tuple(rows[0]) == experiments.MANIFEST_COLUMNS + ("video", "frame")
        assert len(rows) == len(pngs) + 1
        assert all(r["video"] == str(gif) and r["frame"] == "0" for r in rows)
    finally:
        for d in created:
            shutil.rmtree(d)
