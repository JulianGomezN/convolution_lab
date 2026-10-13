"""Modo Unsharp en GPU frente a la fórmula en numpy (R13.1, R13.2, R14.1)."""

import numpy as np
import pytest

from src import kernels
from src.pipeline import Processor, View
from src.state import MODE_UNSHARP, AppState
from src.texture_loader import create_texture
from tests.reference import NP_MODES, blur_separable

W, H = 7, 5


@pytest.fixture(scope="module")
def source(ctx):
    rng = np.random.default_rng(23)
    src = rng.uniform(-0.5, 1.5, size=(H, W, 4)).astype(np.float32)
    tex = create_texture(ctx, src)
    yield src, tex
    tex.release()


@pytest.fixture(scope="module")
def processor(ctx, quad, source):
    proc = Processor(ctx, quad, source[1])
    yield proc
    proc.release()


def _read(tex) -> np.ndarray:
    w, h = tex.size
    return np.frombuffer(tex.read(), dtype=np.float32).reshape(h, w, 4).copy()


@pytest.mark.parametrize("border", [0, 1, 2, 3], ids=lambda b: NP_MODES[b])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("separable", [True, False], ids=["separable", "2D"])
def test_unsharp_matches_formula(processor, source, alpha, border, separable):
    src, _ = source
    sigma = 2.0
    state = AppState(mode=MODE_UNSHARP, sigma=sigma, alpha=alpha, border=border, separable=separable)
    tex, view = processor.run(state)
    out = _read(tex)
    orig = src[..., :3].astype(np.float64)
    blur = blur_separable(orig, kernels.gaussian_1d(sigma), border)
    expected = orig + alpha * (orig - blur)
    assert np.max(np.abs(out[..., :3] - expected)) <= 1e-4
    assert np.all(out[..., 3] == 1.0)
    assert view == View()
    blur_passes = ["blur H", "blur V"] if separable else ["blur 2D"]
    assert processor.last_passes == blur_passes + ["unsharp"]
    assert tex is not processor.source


def test_unsharp_ignores_box_profile(processor, source):
    """Unsharp usa siempre el gaussiano; X solo cambia el perfil del modo Desenfoque."""
    gauss = _read(processor.run(AppState(mode=MODE_UNSHARP, sigma=1.0))[0])
    box = _read(processor.run(AppState(mode=MODE_UNSHARP, sigma=1.0, profile=kernels.PROFILE_BOX))[0])
    np.testing.assert_array_equal(gauss, box)


def test_alpha_zero_is_identity(processor, source):
    src, _ = source
    out = _read(processor.run(AppState(mode=MODE_UNSHARP, alpha=0.0))[0])
    np.testing.assert_allclose(out[..., :3], src[..., :3], atol=1e-6)
