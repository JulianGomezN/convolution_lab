"""convolve3x3.frag y modo Kernel del Processor frente a numpy (R5.2, R5.3, R6.2, R9.1)."""

from dataclasses import replace

import numpy as np
import pytest

from src import kernels
from src.pipeline import Processor, View
from src.state import AppState, effective_kernel
from src.texture_loader import create_texture
from tests.reference import NP_MODES, correlate3x3

W, H = 7, 5
RANDOM_KERNEL = tuple(np.random.default_rng(123).uniform(-3.0, 3.0, 9).round(4).tolist())
KERNEL_CASES = [(key, None) for key in kernels.PRESETS] + [("0", RANDOM_KERNEL)]


def _image(seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 2.0, size=(H, W, 4)).astype(np.float32)


@pytest.fixture(scope="module")
def source(ctx):
    src = _image()
    tex = create_texture(ctx, src)
    yield src, tex
    tex.release()


@pytest.fixture(scope="module")
def processor(ctx, quad, source):
    proc = Processor(ctx, quad, source[1])
    yield proc
    proc.release()


def _read(tex):
    return np.frombuffer(tex.read(), dtype=np.float32).reshape(H, W, 4)


@pytest.mark.parametrize("border", [0, 1, 2, 3], ids=lambda b: NP_MODES[b])
@pytest.mark.parametrize("key, custom", KERNEL_CASES, ids=[k for k, _ in KERNEL_CASES])
@pytest.mark.parametrize("normalize", [True, False], ids=["norm", "sin-norm"])
def test_kernel_mode_matches_numpy(processor, source, key, custom, border, normalize):
    src, _ = source
    state = AppState(kernel_id=key, border=border, normalize=normalize)
    if custom is not None:
        state = replace(state, custom_kernel=custom)
    tex, view = processor.run(state)
    out = _read(tex)
    weights, _ = effective_kernel(state)
    expected = correlate3x3(src[..., :3], weights, border)
    assert np.max(np.abs(out[..., :3] - expected)) <= 1e-4
    assert np.all(out[..., 3] == 1.0)  # alfa = 1 (S3)
    assert view == View(map=state.display_map)


def test_result_is_float_and_keeps_negatives(processor):
    """Laplaciano sin recortar en RGBA32F (R9.1)."""
    tex, _ = processor.run(AppState(kernel_id="7"))
    assert tex.dtype == "f4"
    assert tex.size == (W, H)
    out = _read(tex)
    assert out[..., :3].min() < 0.0


def test_box_without_normalization_saturates(ctx, quad):
    """Box de nueve unos sin normalizar: 9 × gris medio > 1 (satura al mostrarlo)."""
    src = np.full((H, W, 4), 0.5, dtype=np.float32)
    tex = create_texture(ctx, src)
    proc = Processor(ctx, quad, tex)
    try:
        out = _read(proc.run(AppState(kernel_id="2", normalize=False, border=1))[0])
        np.testing.assert_allclose(out[..., :3], 4.5, atol=1e-5)
        out = _read(proc.run(AppState(kernel_id="2", normalize=True, border=1))[0])
        np.testing.assert_allclose(out[..., :3], 0.5, atol=1e-6)
    finally:
        proc.release()
        tex.release()


def test_rgba8_input(ctx, quad):
    """Entrada RGBA8 normal (la de la app): se lee como valores en [0, 1]."""
    rng = np.random.default_rng(8)
    src = rng.integers(0, 256, size=(H, W, 4), dtype=np.uint8)
    tex = create_texture(ctx, src)
    proc = Processor(ctx, quad, tex)
    try:
        state = AppState(kernel_id="4", border=2)
        out = _read(proc.run(state)[0])
        expected = correlate3x3(src[..., :3] / 255.0, effective_kernel(state)[0], 2)
        assert np.max(np.abs(out[..., :3] - expected)) <= 1e-4
    finally:
        proc.release()
        tex.release()
