"""Pasadas de render en GPU: quad, run_pass, RenderTarget y display.frag (R2, R3.4, R5.1)."""

import numpy as np
import pytest

from src.framebuffers import RenderTarget
from src.render_pass import run_pass, set_uniform
from src.shader_loader import load_program
from src.texture_loader import create_texture

IDENTITY = [0, 0, 0, 0, 1, 0, 0, 0, 0]
W, H = 7, 5  # tamaño impar y no cuadrado para detectar ejes intercambiados


@pytest.fixture(scope="module")
def convolve(ctx):
    prog = load_program(ctx, "convolve3x3.frag")
    yield prog
    prog.release()


@pytest.fixture(scope="module")
def display(ctx):
    prog = load_program(ctx, "display.frag")
    yield prog
    prog.release()


def _random_f4(seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.uniform(-2.0, 3.0, size=(H, W, 4)).astype(np.float32)  # negativos y > 1
    return arr


def test_render_target_is_f4_and_exact_size(ctx):
    rt = RenderTarget.create(ctx, (W, H))
    try:
        assert rt.texture.size == (W, H)
        assert rt.texture.dtype == "f4"
        assert rt.texture.components == 4
        assert rt.read().shape == (H, W, 4)
    finally:
        rt.release()


def test_identity_roundtrip_float(ctx, quad, convolve):
    src = _random_f4()
    tex = create_texture(ctx, src)
    rt = RenderTarget.create(ctx, (W, H))
    try:
        run_pass(quad, convolve, rt, {"u_src": tex}, {"u_kernel": IDENTITY, "u_border": 0})
        out = rt.read()
        np.testing.assert_allclose(out[..., :3], src[..., :3], atol=1e-6)
        assert np.all(out[..., 3] == 1.0)  # alfa = 1 (S3)
    finally:
        rt.release()
        tex.release()


def test_identity_roundtrip_rgba8(ctx, quad, convolve):
    rng = np.random.default_rng(1)
    src = rng.integers(0, 256, size=(H, W, 4), dtype=np.uint8)
    tex = create_texture(ctx, src)
    rt = RenderTarget.create(ctx, (W, H))
    try:
        run_pass(quad, convolve, rt, {"u_src": tex}, {"u_kernel": IDENTITY})
        out = rt.read()
        np.testing.assert_allclose(out[..., :3], src[..., :3] / 255.0, atol=1e-6)
    finally:
        rt.release()
        tex.release()


def test_kernel_offset_follows_numpy_convention(ctx, quad, convolve):
    """Peso en la fila 0, columna 2 → lee el vecino (x+1, y-1); y hacia abajo (R5.2)."""
    src = _random_f4(2)
    tex = create_texture(ctx, src)
    rt = RenderTarget.create(ctx, (W, H))
    try:
        run_pass(quad, convolve, rt, {"u_src": tex}, {"u_kernel": [0, 0, 1, 0, 0, 0, 0, 0, 0]})
        out = rt.read()[..., :3]
        expected = np.zeros_like(src[..., :3])
        expected[1:, :-1] = src[:-1, 1:, :3]  # out[y, x] = src[y-1, x+1], cero fuera
        np.testing.assert_allclose(out, expected, atol=1e-6)
    finally:
        rt.release()
        tex.release()


def test_feedback_assertion(ctx, quad, convolve):
    rt = RenderTarget.create(ctx, (W, H))
    try:
        with pytest.raises(AssertionError, match="feedback"):
            run_pass(quad, convolve, rt, {"u_src": rt.texture}, {"u_kernel": IDENTITY})
    finally:
        rt.release()


def test_unknown_uniform_is_ignored(ctx, quad, convolve):
    tex = create_texture(ctx, _random_f4(3))
    rt = RenderTarget.create(ctx, (W, H))
    try:
        run_pass(quad, convolve, rt, {"u_src": tex}, {"u_kernel": IDENTITY, "u_no_existe": 1.0})
        set_uniform(convolve, "u_tampoco", 3)
    finally:
        rt.release()
        tex.release()


@pytest.mark.parametrize("flip", [0, 1])
def test_display_flip(ctx, quad, convolve, display, flip):
    """display.frag con u_flip_y = 0 conserva la orientación; con 1 invierte filas."""
    rng = np.random.default_rng(4)
    src = rng.uniform(0.0, 1.0, size=(H, W, 4)).astype(np.float32)
    src[0, 0, :3] = 2.0  # se recorta a 1
    tex = create_texture(ctx, src)
    out_rt = RenderTarget.create(ctx, (W, H), dtype="f1")
    try:
        run_pass(quad, display, out_rt, {"u_src": tex}, {"u_flip_y": flip, "u_map": 0, "u_gain": 1.0})
        out = out_rt.read()
        expected = np.clip(src[..., :3], 0.0, 1.0)
        if flip:
            expected = expected[::-1]
        np.testing.assert_allclose(out[..., :3] / 255.0, expected, atol=1.0 / 255.0 + 1e-6)
        assert np.all(out[..., 3] == 255)
    finally:
        out_rt.release()
        tex.release()


@pytest.mark.parametrize(
    "display_map, gain, func",
    [
        (0, 1.0, lambda v: np.clip(v, 0.0, 1.0)),
        (1, 1.0, lambda v: np.clip(np.abs(v), 0.0, 1.0)),
        (2, 1.0, lambda v: np.clip(0.5 + 0.5 * v, 0.0, 1.0)),
        (1, 0.25, lambda v: np.clip(np.abs(v) * 0.25, 0.0, 1.0)),
    ],
    ids=["recorte", "abs", "offset", "abs-ganancia"],
)
def test_display_maps(ctx, quad, display, display_map, gain, func):
    """Mapeos de visualización de valores negativos (R9.2)."""
    src = _random_f4(5)  # valores en [-2, 3]
    tex = create_texture(ctx, src)
    out_rt = RenderTarget.create(ctx, (W, H), dtype="f1")
    try:
        run_pass(quad, display, out_rt, {"u_src": tex}, {"u_flip_y": 0, "u_map": display_map, "u_gain": gain})
        out = out_rt.read()
        np.testing.assert_allclose(out[..., :3] / 255.0, func(src[..., :3]), atol=1.0 / 255.0 + 1e-6)
    finally:
        out_rt.release()
        tex.release()


# --- lupa (R21) ------------------------------------------------------------


def test_display_lens_off_matches_plain(ctx, quad, display):
    """Con u_lens_on = 0 el resultado es idéntico al de siempre, aunque se pase u_original."""
    src = _random_f4(6)
    original = _random_f4(7)
    tex = create_texture(ctx, src)
    orig = create_texture(ctx, original)
    plain = RenderTarget.create(ctx, (W, H), dtype="f1")
    lensed = RenderTarget.create(ctx, (W, H), dtype="f1")
    try:
        base = {"u_flip_y": 0, "u_map": 0, "u_gain": 1.0}
        run_pass(quad, display, plain, {"u_src": tex}, base)
        run_pass(quad, display, lensed, {"u_src": tex, "u_original": orig}, {**base, "u_lens_on": 0})
        np.testing.assert_array_equal(plain.read(), lensed.read())
    finally:
        plain.release()
        lensed.release()
        orig.release()
        tex.release()


def test_display_lens_on_mixes_src_and_original(ctx, quad, display):
    """Con u_lens_on = 1: centro del círculo ~ u_src, lejos ~ u_original, borde intermedio."""
    src = np.zeros((H, W, 4), np.float32)
    src[..., :3] = 0.8  # procesado = constante clara
    original = np.zeros((H, W, 4), np.float32)
    original[..., :3] = 0.1  # original = constante oscura, distinta
    tex = create_texture(ctx, src)
    orig = create_texture(ctx, original)
    out_rt = RenderTarget.create(ctx, (W, H), dtype="f1")
    try:
        # Centro en la esquina (0, 0); banda de transición en d ∈ [0.5, 2.5] px:
        # el píxel (0, 1) (d = 1) cae dentro de la transición.
        run_pass(
            quad, display, out_rt,
            {"u_src": tex, "u_original": orig},
            {"u_flip_y": 0, "u_map": 0, "u_gain": 1.0, "u_lens_on": 1,
             "u_lens_center": (0.0, 0.0), "u_lens_radius": 2.5, "u_lens_edge": 2.0},
        )
        out = out_rt.read()[..., :3] / 255.0
    finally:
        out_rt.release()
        orig.release()
        tex.release()
    # Píxel del centro (0, 0), d = 0: dentro del círculo → procesado (0.8).
    np.testing.assert_allclose(out[0, 0], 0.8, atol=1.0 / 255.0 + 1e-6)
    # Píxel lejano (esquina opuesta), d ≫ radio: fuera → original (0.1).
    np.testing.assert_allclose(out[H - 1, W - 1], 0.1, atol=1.0 / 255.0 + 1e-6)
    # Píxel (0, 1), d = 1 px dentro de la banda: valor estrictamente entre ambos.
    mid = out[1, 0, 0]
    assert 0.1 + 1e-3 < mid < 0.8 - 1e-3
