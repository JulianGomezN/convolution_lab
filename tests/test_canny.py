"""Pipeline Canny simplificado en GPU (R12): ping-pong, etapas, gris, Sobel, NMS y doble umbral."""

import colorsys
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

from src import kernels
from src.framebuffers import PingPong, RenderTarget
from src.pipeline import CANNY_PASSES, MAG_GAIN, STAGE_PASSES, STAGE_VIEWS, Processor, View
from src.render_pass import run_pass
from src.shader_loader import load_program
from src.state import MODE_CANNY, STAGE_COUNT, AppState
from src.texture_loader import create_texture
from tests import reference as ref
from tests.reference import NP_MODES

BORDERS = pytest.mark.parametrize("border", [0, 1, 2, 3], ids=lambda b: NP_MODES[b])


def _read(tex) -> np.ndarray:
    w, h = tex.size
    return np.frombuffer(tex.read(), dtype=np.float32).reshape(h, w, 4).copy()


def _random(w, h, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(h, w, 4)).astype(np.float32)


@contextmanager
def _processor(ctx, quad, arr):
    tex = create_texture(ctx, arr)
    proc = Processor(ctx, quad, tex)
    try:
        yield proc
    finally:
        proc.release()
        tex.release()


def _canny(stage, border=1, **kw):
    return AppState(mode=MODE_CANNY, stage=stage, border=border, **kw)


# --- ping-pong y armazón de etapas (R12.2–R12.5) ------------------------


def test_pingpong_swap_never_aliases(ctx):
    pp = PingPong.create(ctx, (7, 5))
    try:
        seen = set()
        for _ in range(10):
            assert pp.src is not pp.dst
            assert pp.src.texture is not pp.dst.texture
            assert pp.src.texture.dtype == pp.dst.texture.dtype == "f4"
            assert pp.src.size == pp.dst.size == (7, 5)
            seen.add(id(pp.src))
            old_src, old_dst = pp.src, pp.dst
            pp.swap()
            assert (pp.src, pp.dst) == (old_dst, old_src)
        assert len(seen) == 2
    finally:
        pp.release()


def test_pingpong_rejects_same_target(ctx):
    rt = RenderTarget.create(ctx, (7, 5))
    try:
        with pytest.raises(ValueError):
            PingPong(rt, rt)
    finally:
        rt.release()


def test_stage_tables_cover_eleven_stages():
    assert set(STAGE_PASSES) == set(STAGE_VIEWS) == set(range(1, STAGE_COUNT + 1))
    assert STAGE_COUNT == 11
    assert max(STAGE_PASSES.values()) == len(CANNY_PASSES)


@pytest.mark.parametrize("stage", range(1, STAGE_COUNT + 1))
def test_runs_only_needed_passes(ctx, quad, stage):
    """Solo las pasadas hasta la etapa visible (R12.5); ninguna lee su propio destino (R12.3)."""
    src = _random(7, 5, 1)
    with _processor(ctx, quad, src) as proc:
        tex, view = proc.run(_canny(stage))
        assert proc.last_passes == list(CANNY_PASSES[: STAGE_PASSES[stage]])
        assert view == STAGE_VIEWS[stage]
        if stage == 1:
            assert tex is proc.source  # Original: la entrada sin procesar
        else:
            assert tex.dtype == "f4" and tex.size == (7, 5)  # RGBA32F W×H (R3.4, R9.1)


def test_stage_views():
    assert STAGE_VIEWS[1] == View()
    assert STAGE_VIEWS[5] == View(map=kernels.MAP_ABS, gain=0.25, channel=1)
    assert STAGE_VIEWS[6] == View(map=kernels.MAP_ABS, gain=0.25, channel=2)
    assert STAGE_VIEWS[7].channel == 3 and STAGE_VIEWS[8].channel == 4
    assert STAGE_VIEWS[10].channel == 2 and STAGE_VIEWS[11].channel == 1
    assert MAG_GAIN == 0.25


# --- gris y desenfoque (R12.6, R12.7) ------------------------------------


def test_gray_matches_numpy_f4(ctx, quad):
    src = np.random.default_rng(2).uniform(-1.0, 2.0, size=(5, 7, 4)).astype(np.float32)
    with _processor(ctx, quad, src) as proc:
        out = _read(proc.run(_canny(2))[0])
    y = ref.grayscale(src)
    for c in range(3):
        np.testing.assert_allclose(out[..., c], y, atol=1e-6)
    assert np.all(out[..., 3] == 1.0)


def test_gray_matches_numpy_rgba8(ctx, quad):
    src = np.random.default_rng(3).integers(0, 256, size=(5, 7, 4), dtype=np.uint8)
    with _processor(ctx, quad, src) as proc:
        out = _read(proc.run(_canny(2))[0])
    np.testing.assert_allclose(out[..., 0], ref.grayscale(src / 255.0), atol=1e-6)


@BORDERS
@pytest.mark.parametrize("sigma", [0.5, 1.4, 3.0])
def test_blur_stages_match_numpy(ctx, quad, border, sigma):
    src = _random(7, 5, 4)
    w = kernels.gaussian_1d(sigma)
    gray = ref.grayscale(src)
    with _processor(ctx, quad, src) as proc:
        blur_h = _read(proc.run(_canny(3, border, canny_sigma=sigma))[0])[..., 0]
        blur = _read(proc.run(_canny(4, border, canny_sigma=sigma))[0])[..., 0]
    assert np.max(np.abs(blur_h - ref.blur_1d(gray, w, (1, 0), border))) <= 1e-4
    assert np.max(np.abs(blur - ref.blur_separable(gray, w, border))) <= 1e-4


def test_canny_always_uses_gaussian(ctx, quad):
    """El perfil box del modo Desenfoque no afecta a Canny (R12.7)."""
    src = _random(7, 5, 5)
    with _processor(ctx, quad, src) as proc:
        a = _read(proc.run(_canny(4, profile=kernels.PROFILE_GAUSS))[0])
        b = _read(proc.run(_canny(4, profile=kernels.PROFILE_BOX))[0])
    np.testing.assert_array_equal(a, b)


# --- Sobel (R12.8) ---------------------------------------------------------


def _angle_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


@BORDERS
def test_sobel_matches_numpy(ctx, quad, border):
    src = _random(13, 9, 6)
    with _processor(ctx, quad, src) as proc:
        grads = {k: _read(proc.run(_canny(k, border))[0]) for k in (5, 6, 7, 8)}
    blurred = ref.blur_separable(ref.grayscale(src), kernels.gaussian_1d(1.4), border)
    expected = ref.sobel(blurred, border)
    # Las etapas 5-8 son la misma textura (Gx, Gy, |G|, θ); cambia la vista.
    for k in (6, 7, 8):
        np.testing.assert_array_equal(grads[k], grads[5])
    out = grads[5]
    for c in range(3):
        assert np.max(np.abs(out[..., c] - expected[..., c])) <= 1e-4, "GxGy|G|"[c]
    mask = expected[..., 2] > 1e-2  # θ está mal condicionado con |G| ≈ 0
    assert mask.any()
    assert np.max(_angle_diff(out[..., 3], expected[..., 3])[mask]) <= 1e-3


@pytest.fixture(scope="module")
def sobel_prog(ctx):
    prog = load_program(ctx, "sobel.frag")
    yield prog
    prog.release()


def _sobel_direct(ctx, quad, prog, gray, border=1):
    h, w = gray.shape
    arr = np.zeros((h, w, 4), np.float32)
    arr[..., 0] = gray
    tex = create_texture(ctx, arr)
    rt = RenderTarget.create(ctx, (w, h))
    try:
        run_pass(quad, prog, rt, {"u_src": tex}, {"u_border": border})
        return rt.read()
    finally:
        rt.release()
        tex.release()


def test_sobel_sign_horizontal_step(ctx, quad, sobel_prog):
    """Oscuro → claro hacia la derecha: Gx > 0, Gy = 0; |G| = 4 en el escalón ideal (D3)."""
    gray = np.zeros((6, 10), np.float32)
    gray[:, 5:] = 1.0
    out = _sobel_direct(ctx, quad, sobel_prog, gray)
    np.testing.assert_allclose(out[:, 4:6, 0], 4.0)
    np.testing.assert_allclose(out[:, 4:6, 2], 4.0)
    np.testing.assert_allclose(out[:, 4:6, 3], 0.0, atol=1e-6)  # θ = 0
    assert np.all(out[..., 0] >= 0.0)
    np.testing.assert_array_equal(out[..., 1], 0.0)


def test_sobel_sign_vertical_step(ctx, quad, sobel_prog):
    """Oscuro arriba, claro abajo (y hacia abajo): Gy > 0, Gx = 0, θ = 90°."""
    gray = np.zeros((10, 6), np.float32)
    gray[5:, :] = 1.0
    out = _sobel_direct(ctx, quad, sobel_prog, gray)
    np.testing.assert_allclose(out[4:6, :, 1], 4.0)
    np.testing.assert_allclose(out[4:6, :, 3], np.pi / 2, atol=1e-6)
    assert np.all(out[..., 1] >= 0.0)
    np.testing.assert_array_equal(out[..., 0], 0.0)


def test_sobel_flat_image_has_zero_angle(ctx, quad, sobel_prog):
    """atan(0, 0) no está definido en GLSL; el shader devuelve 0 como numpy."""
    out = _sobel_direct(ctx, quad, sobel_prog, np.full((5, 7), 0.5, np.float32))
    np.testing.assert_array_equal(out, np.zeros_like(out))


# --- NMS (R12.9) -----------------------------------------------------------


@BORDERS
def test_nms_matches_numpy(ctx, quad, border):
    src = _random(13, 9, 7)
    with _processor(ctx, quad, src) as proc:
        grad = _read(proc.run(_canny(5, border))[0])
        out = _read(proc.run(_canny(9, border))[0])
    expected = ref.nms(grad, border)
    np.testing.assert_allclose(out[..., 0], expected, atol=1e-6)
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    assert (expected > 0).any() and (expected == 0).any()


def _kept(nms):
    return nms > 0.01 * nms.max()


def test_nms_vertical_step_gives_one_pixel_line(ctx, quad):
    """Escalón vertical suavizado: |G| ocupa varias columnas y NMS deja 1 px por fila."""
    src = np.zeros((12, 32, 4), np.float32)
    src[:, 16:] = 1.0
    with _processor(ctx, quad, src) as proc:
        mag = _read(proc.run(_canny(7))[0])[..., 2]
        nms = _read(proc.run(_canny(9))[0])[..., 0]
    assert np.all((mag > 0.01 * mag.max()).sum(axis=1) >= 5)
    kept = _kept(nms)
    assert np.all(kept.sum(axis=1) == 1)
    assert set(np.nonzero(kept)[1]) <= {15, 16}


def test_nms_diagonal_step_is_thin(ctx, quad):
    """Escalón a 45°: NMS deja una línea diagonal de 1 px (escalera en conectividad 4).

    Con vecinos discretos a 45°, la diagonal k = x - y se compara con k ± 2, así
    que sobreviven el máximo k = 0 y uno de sus vecinos k = ±1: nunca un bloque 2x2.
    """
    n = 24
    y, x = np.mgrid[0:n, 0:n]
    src = np.zeros((n, n, 4), np.float32)
    src[..., :3] = np.where(x > y, 1.0, np.where(x == y, 0.5, 0.0))[..., None]
    with _processor(ctx, quad, src) as proc:
        mag = _read(proc.run(_canny(7))[0])[..., 2]
        nms = _read(proc.run(_canny(9))[0])[..., 0]
    win = (slice(8, 16), slice(8, 16))  # lejos de las esquinas, donde el borde rompe la simetría
    k = (x - y)[win]
    kept = _kept(nms)[win]
    assert len(set(k[mag[win] > 0.01 * mag.max()])) >= 5  # antes de NMS, banda ancha
    ks = sorted(set(k[kept]))
    assert 0 in ks and len(ks) == 2 and ks[1] - ks[0] == 1
    assert np.all(kept.any(axis=1))  # la línea es continua: algún píxel en cada fila
    blocks = kept[:-1, :-1] & kept[1:, :-1] & kept[:-1, 1:] & kept[1:, 1:]
    assert not blocks.any()


# --- doble umbral (R12.10, R12.11) ------------------------------------------


@pytest.fixture(scope="module")
def threshold_prog(ctx):
    prog = load_program(ctx, "threshold.frag")
    yield prog
    prog.release()


def _threshold_direct(ctx, quad, prog, m, low, high, border=0):
    h, w = m.shape
    arr = np.zeros((h, w, 4), np.float32)
    arr[..., 0] = m
    tex = create_texture(ctx, arr)
    rt = RenderTarget.create(ctx, (w, h))
    try:
        run_pass(
            quad, prog, rt, {"u_src": tex},
            {"u_border": border, "u_low": low, "u_high": high, "u_mag_gain": MAG_GAIN},
        )
        return rt.read()
    finally:
        rt.release()
        tex.release()


def test_threshold_synthetic_cases(ctx, quad, threshold_prog):
    """alto = 0.5 y bajo = 0.2 sobre m·0.25: fuerte si m ≥ 2, débil si 0.8 ≤ m < 2."""
    m = np.zeros((5, 7), np.float32)
    m[1, 1] = 2.0   # fuerte (justo en el umbral alto)
    m[1, 2] = 1.0   # débil con vecino fuerte a la izquierda → borde
    m[2, 2] = 0.8   # débil (justo en el bajo) con vecino fuerte en diagonal → borde
    m[3, 5] = 1.5   # débil aislado → no es borde
    m[4, 0] = 0.79  # bajo el umbral bajo
    m[0, 6] = 3.0   # fuerte en una esquina
    out = _threshold_direct(ctx, quad, threshold_prog, m, low=0.2, high=0.5)
    edges = np.zeros((5, 7))
    edges[1, 1] = edges[1, 2] = edges[2, 2] = edges[0, 6] = 1.0
    cls = np.zeros((5, 7))
    cls[1, 1] = cls[0, 6] = 1.0
    cls[1, 2] = cls[2, 2] = cls[3, 5] = 0.5
    np.testing.assert_array_equal(out[..., 0], edges)
    np.testing.assert_array_equal(out[..., 1], cls)
    np.testing.assert_array_equal(out[..., 2], edges)
    assert np.all(out[..., 3] == 1.0)


@pytest.mark.parametrize("border, expected", [(0, 0.0), (3, 1.0)], ids=["zero", "wrap"])
def test_threshold_neighbors_follow_border_mode(ctx, quad, threshold_prog, border, expected):
    """Un débil en la columna 0 solo ve al fuerte de la última columna con wrap."""
    m = np.zeros((5, 7), np.float32)
    m[2, 0] = 1.0
    m[2, 6] = 3.0
    out = _threshold_direct(ctx, quad, threshold_prog, m, low=0.2, high=0.5, border=border)
    assert out[2, 0, 0] == expected
    assert out[2, 0, 1] == 0.5


@BORDERS
def test_threshold_stages_match_numpy(ctx, quad, border):
    src = _random(13, 9, 8)
    # σ pequeño para que el ruido conserve gradientes fuertes, débiles y nulos.
    state = dict(high=0.3, low_ratio=0.4, canny_sigma=0.5)
    with _processor(ctx, quad, src) as proc:
        nms = _read(proc.run(_canny(9, border, **state))[0])[..., 0]
        cls = _read(proc.run(_canny(10, border, **state))[0])
        edges = _read(proc.run(_canny(11, border, **state))[0])
    np.testing.assert_array_equal(cls, edges)  # misma textura, distinta vista
    exp_edges, exp_cls = ref.double_threshold(nms, 0.4 * 0.3, 0.3, MAG_GAIN, border)
    np.testing.assert_array_equal(edges[..., 0], exp_edges)
    np.testing.assert_array_equal(edges[..., 1], exp_cls)
    assert set(np.unique(exp_cls)) >= {0.0, 0.5, 1.0}  # el caso ejercita las tres clases


def test_default_values_show_edges_of_default_image(ctx, quad):
    """Con los valores por defecto (σ = 1.4, alto = 0.3) la etapa 11 muestra el contorno de la F (S7)."""
    from src.config import SETTINGS_PATH, load_settings
    from src.texture_loader import load_image

    settings = load_settings(SETTINGS_PATH)
    img = load_image(SETTINGS_PATH.parent.parent / settings.data["default_image"])
    state = _canny(11, border=0, canny_sigma=settings.canny_sigma, high=settings.canny_high,
                   low_ratio=settings.canny_low_ratio)
    with _processor(ctx, quad, img) as proc:
        edges = _read(proc.run(state)[0])[..., 0]
    # Barra superior de la F: borde horizontal en y ≈ 80 entre x = 200 y 460 (orientacion.png 640x480).
    top_bar = edges[78:83, 220:440]
    assert np.all(top_bar.any(axis=0))  # contorno continuo a lo largo de la barra
    assert top_bar.sum(axis=0).max() <= 2  # fino (NMS)
    assert 1000 < edges.sum() < 0.02 * edges.size
    # Con el umbral anterior (0.6) no saldría nada: el máximo de |G|/4 tras el blur es ≈ 0.41.
    with _processor(ctx, quad, img) as proc:
        assert _read(proc.run(replace(state, high=0.6))[0])[..., 0].sum() == 0


def test_higher_threshold_gives_fewer_edges(ctx, quad):
    src = _random(32, 24, 9)
    with _processor(ctx, quad, src) as proc:
        counts = [
            _read(proc.run(_canny(11, high=h, canny_sigma=0.5))[0])[..., 0].sum() for h in (0.1, 0.3, 0.6)
        ]
    assert counts[0] > counts[1] > counts[2]


# --- display.frag: selección de canal y dirección -----------------------------


@pytest.fixture(scope="module")
def display(ctx):
    prog = load_program(ctx, "display.frag")
    yield prog
    prog.release()


def _display(ctx, quad, prog, arr, **uniforms):
    h, w = arr.shape[:2]
    tex = create_texture(ctx, arr)
    rt = RenderTarget.create(ctx, (w, h), dtype="f1")
    try:
        run_pass(quad, prog, rt, {"u_src": tex}, {"u_flip_y": 0, "u_map": 0, "u_gain": 1.0, **uniforms})
        return rt.read()[..., :3] / 255.0
    finally:
        rt.release()
        tex.release()


@pytest.mark.parametrize("channel", [1, 2, 3])
def test_display_single_channel_as_gray(ctx, quad, display, channel):
    arr = _random(7, 5, 10)
    out = _display(ctx, quad, display, arr, u_channel=channel)
    for c in range(3):
        np.testing.assert_allclose(out[..., c], arr[..., channel - 1], atol=1 / 255 + 1e-6)


def test_display_direction_hsv(ctx, quad, display):
    rng = np.random.default_rng(12)
    arr = np.zeros((5, 7, 4), np.float32)
    arr[..., 2] = rng.uniform(0.0, 6.0, (5, 7))  # |G|; con ganancia 0.25 parte satura
    arr[..., 3] = rng.uniform(-np.pi, np.pi, (5, 7))  # θ
    arr[0, :4, 3] = [-np.pi, -np.pi / 2, 0.0, np.pi / 2]
    arr[0, :4, 2] = 4.0
    out = _display(ctx, quad, display, arr, u_channel=4, u_gain=0.25)
    expected = np.array(
        [
            [colorsys.hsv_to_rgb((t + np.pi) / (2 * np.pi), 1.0, min(1.0, m * 0.25)) for m, t in row]
            for row in arr[..., 2:4].astype(np.float64)
        ]
    )
    np.testing.assert_allclose(out, expected, atol=1 / 255 + 1e-4)
    np.testing.assert_allclose(out[0, 0], (1, 0, 0), atol=1 / 255)  # θ = -π: rojo
    np.testing.assert_allclose(out[0, 2], (0, 1, 1), atol=1 / 255)  # θ = 0: cian
