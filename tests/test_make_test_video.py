"""Tests de tools/make_test_video.py: animación determinista y pequeña (R14.6, R14.7)."""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from src.video import PillowFrameSource

ROOT = Path(__file__).resolve().parent.parent


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_test_video", ROOT / "tools" / "make_test_video.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames_rgb(path: Path) -> list[np.ndarray]:
    with Image.open(path) as img:
        return [np.asarray(f.convert("RGB")).copy() for f in ImageSequence.Iterator(img)]


def test_generate_is_deterministic(tmp_path):
    # R14.7: dos ejecuciones → mismo número de frames, tamaño y píxeles.
    gen = _load_generator()
    a = gen.generate(tmp_path / "a")
    b = gen.generate(tmp_path / "b")
    fa, fb = _frames_rgb(a), _frames_rgb(b)
    assert len(fa) == len(fb) == gen.N_FRAMES
    for x, y in zip(fa, fb):
        assert x.shape == y.shape
        assert np.array_equal(x, y)


def test_generated_video_is_small_and_opens_with_pillow_backend(tmp_path):
    # R14.6: 2–10 s, 2–300 frames, ≤ 640×480, se abre con el Backend_Pillow.
    gen = _load_generator()
    assert 2.0 <= gen.N_FRAMES * gen.DURATION_MS / 1000.0 <= 10.0
    path = gen.generate(tmp_path)

    src = PillowFrameSource(path)
    try:
        src.open()
        assert src.n_frames is not None and 2 <= src.n_frames <= 300
        assert src.n_frames == gen.N_FRAMES
        w, h = src.size
        assert (w, h) == (gen.WIDTH, gen.HEIGHT)
        assert w <= 640 and h <= 480
        total = sum(src.durations)
        assert 2.0 <= total <= 10.0
        last = src.read(src.n_frames - 1)
        assert last is not None and last.shape == (h, w, 4)
        assert src.read(src.n_frames) is None
    finally:
        src.close()
