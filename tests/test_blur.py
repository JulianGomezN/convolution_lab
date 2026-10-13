"""Desenfoque 1D: pesos en la CPU (R11.4, S8) y blur_1d.frag frente a numpy (R11, R12.7)."""

import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src import kernels
from src.pipeline import Processor, View
from src.state import MODE_BLUR, MODE_CANNY, AppState
from src.texture_loader import create_texture
from tests.reference import NP_MODES, blur_separable

ROOT = Path(__file__).resolve().parent.parent
SIGMAS = [0.5 * i for i in range(1, 21)]  # 0.5 .. 10, paso 0.5
W, H = 7, 5


# --- CPU ----------------------------------------------------------------


@pytest.mark.parametrize("sigma", SIGMAS + [1.4])
def test_gaussian_weights(sigma):
    w = kernels.gaussian_1d(sigma)
    r = math.ceil(3 * sigma - 1e-9)
    assert kernels.gaussian_radius(sigma) == r
    assert len(w) == 2 * r + 1 <= kernels.MAX_TAPS
    assert sum(w) == pytest.approx(1.0, abs=1e-12)
    assert w == pytest.approx(w[::-1])  # simétrico
    assert max(w) == w[r]  # máximo en el centro
    ref = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma**2))
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-12)


def test_gaussian_radius_values():
    assert [kernels.gaussian_radius(s) for s in (0.5, 1.0, 1.4, 2.0, 10.0)] == [2, 3, 5, 6, 30]


@pytest.mark.parametrize("sigma", SIGMAS)
def test_box_weights(sigma):
    w = kernels.box_1d(sigma)
    r = round(math.sqrt(3) * sigma)
    assert kernels.box_radius(sigma) == r
    assert len(w) == 2 * r + 1
    assert all(v == w[0] for v in w)
    assert sum(w) == pytest.approx(1.0)


def test_box_radius_values():
    assert [kernels.box_radius(s) for s in (0.5, 1.0, 2.0, 4.0, 10.0)] == [1, 2, 3, 7, 17]


@pytest.mark.parametrize("sigma", [0.0, 0.4, 10.5, float("nan")])
def test_sigma_out_of_range(sigma):
    with pytest.raises(ValueError):
        kernels.gaussian_1d(sigma)
    with pytest.raises(ValueError):
        kernels.box_1d(sigma)


def test_blur_weights_by_profile():
    assert kernels.blur_weights("gauss", 2.0) == kernels.gaussian_1d(2.0)
    assert kernels.blur_weights("box", 2.0) == kernels.box_1d(2.0)
    with pytest.raises(ValueError):
        kernels.blur_weights("otro", 2.0)


def test_max_taps_matches_shaders():
    """MAX_TAPS de Python = longitud de u_weights en todos los shaders que la declaran."""
    found = {}
    for path in (ROOT / "shaders").glob("*.frag"):
        m = re.search(r"uniform\s+float\s+u_weights\s*\[\s*(\d+)\s*\]", path.read_text(encoding="utf-8"))
        if m:
            found[path.name] = int(m.group(1))
    assert "blur_1d.frag" in found
    assert set(found.values()) == {kernels.MAX_TAPS}
    assert kernels.MAX_TAPS == 2 * kernels.MAX_RADIUS + 1 == 61


# --- GPU ----------------------------------------------------------------


@pytest.fixture(scope="module")
def source(ctx):
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.0, 2.0, size=(H, W, 4)).astype(np.float32)
    tex = create_texture(ctx, src)
    yield src, tex
    tex.release()


@pytest.fixture(scope="module")
def processor(ctx, quad, source):
    proc = Processor(ctx, quad, source[1])
    yield proc
    proc.release()


def test_u_weights_array_length_on_gpu(processor):
    assert processor.prog_blur_1d["u_weights"].array_length == kernels.MAX_TAPS


@pytest.mark.parametrize("border", [0, 1, 2, 3], ids=lambda b: NP_MODES[b])
@pytest.mark.parametrize("profile", ["gauss", "box"])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 10.0])  # σ = 10: r = 30 > tamaño de la imagen
def test_blur_mode_matches_numpy(processor, source, profile, sigma, border):
    src, _ = source
    state = AppState(mode=MODE_BLUR, profile=profile, sigma=sigma, border=border)
    tex, view = processor.run(state)
    out = np.frombuffer(tex.read(), dtype=np.float32).reshape(H, W, 4)
    expected = blur_separable(src[..., :3], kernels.blur_weights(profile, sigma), border)
    assert np.max(np.abs(out[..., :3] - expected)) <= 1e-4
    assert np.all(out[..., 3] == 1.0)
    assert tex.dtype == "f4"
    assert view == View()
    assert processor.last_passes == ["blur H", "blur V"]


# --- gaussiano 2D directo frente a separable (R14.1, R14.2) ------------------


def _read(tex) -> np.ndarray:
    w, h = tex.size
    return np.frombuffer(tex.read(), dtype=np.float32).reshape(h, w, 4).copy()


def test_blur_2d_declares_same_weights(processor):
    assert processor.prog_blur_2d["u_weights"].array_length == kernels.MAX_TAPS


@pytest.mark.parametrize("border", [0, 1, 2, 3], ids=lambda b: NP_MODES[b])
@pytest.mark.parametrize("profile", ["gauss", "box"])
@pytest.mark.parametrize("sigma", [0.5, 2.0, 10.0])  # σ = 10: r = 30, mayor que 7×5
def test_blur_2d_matches_separable(processor, source, profile, sigma, border):
    src, _ = source
    state = AppState(mode=MODE_BLUR, profile=profile, sigma=sigma, border=border)
    sep = _read(processor.run(state)[0])
    tex, view = processor.run(replace(state, separable=False))
    assert processor.last_passes == ["blur 2D"]  # una sola pasada, (2r+1)² lecturas
    two_d = _read(tex)
    assert np.max(np.abs(two_d[..., :3] - sep[..., :3])) <= 1e-4
    expected = blur_separable(src[..., :3], kernels.blur_weights(profile, sigma), border)
    assert np.max(np.abs(two_d[..., :3] - expected)) <= 1e-4
    assert np.all(two_d[..., 3] == 1.0)
    assert view == View() and tex.dtype == "f4" and tex.size == (W, H)


@pytest.mark.parametrize("border", [0, 1, 2, 3], ids=lambda b: NP_MODES[b])
def test_canny_2d_matches_separable(processor, border):
    """S en Canny: el blur en 2D es una pasada; Sobel sobre él coincide con el separable."""
    for stage in (3, 4, 5, 6, 7):
        state = AppState(mode=MODE_CANNY, stage=stage, border=border, canny_sigma=1.4)
        sep = _read(processor.run(state)[0])
        two_d = _read(processor.run(replace(state, separable=False))[0])
        expected = ["gris", "blur 2D"] + (["sobel"] if stage >= 5 else [])
        assert processor.last_passes == expected
        channels = slice(0, 3) if stage >= 5 else slice(0, 1)
        if stage == 3:
            # En 2D la etapa Blur H ya muestra el desenfoque completo (etapa 4 separable).
            sep = _read(processor.run(replace(state, stage=4))[0])
        assert np.max(np.abs(two_d[..., channels] - sep[..., channels])) <= 1e-4, stage


def test_canny_2d_full_pipeline_passes(processor):
    processor.run(AppState(mode=MODE_CANNY, stage=11, separable=False))
    assert processor.last_passes == ["gris", "blur 2D", "sobel", "nms", "umbral"]
