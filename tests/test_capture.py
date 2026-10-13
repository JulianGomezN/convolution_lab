"""Capturas con G (R15.1–R15.4, R3.4)."""

import re
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from src import app as app_mod, capture, kernels, state as st
from src.pipeline import CH_R, Processor, View
from src.render_pass import run_pass
from src.shader_loader import load_program
from src.state import AppState
from src.texture_loader import create_texture

NOW = datetime(2025, 3, 7, 9, 5, 2)
W, H = 7, 5


# --- nombres (CPU) --------------------------------------------------------


def test_slug():
    assert capture.slug("Dirección") == "direccion"
    assert capture.slug("Sobel Gx") == "sobel-gx"
    assert capture.slug("Blur H") == "blur-h"
    assert capture.slug("Clasificación") == "clasificacion"


@pytest.mark.parametrize(
    "state, expected",
    [
        (AppState(kernel_id="5", display_map=kernels.MAP_ABS, border=st.BORDER_WRAP),
         "kernel_k5-sobel-gx_wrap_norm-si_vista-abs"),
        (AppState(kernel_id="2", normalize=False), "kernel_k2-box_zero_norm-no_vista-recorte"),
        (AppState(kernel_id="0"), "kernel_k0-personalizado_zero_norm-si_vista-recorte"),
        (AppState(mode=st.MODE_BLUR, sigma=2.5, profile=kernels.PROFILE_BOX, border=st.BORDER_REFLECT),
         "desenfoque_box_reflect_s2.5_sep"),
        (AppState(mode=st.MODE_BLUR, separable=False), "desenfoque_gaussiano_zero_s2_2d"),
        (AppState(mode=st.MODE_UNSHARP, sigma=1.0, alpha=0.5, border=st.BORDER_REPLICATE),
         "unsharp_gaussiano_replicate_s1_a0.5_sep"),
        (AppState(mode=st.MODE_CANNY, stage=8), "canny_e08-direccion_zero_s1.4_alto0.30_bajo0.12_sep"),
        (AppState(mode=st.MODE_CANNY, stage=11, high=0.5), "canny_e11-bordes_zero_s1.4_alto0.50_bajo0.20_sep"),
    ],
)
def test_describe(state, expected):
    assert capture.describe(state) == expected


def test_capture_name_format():
    name = capture.capture_name(AppState(kernel_id="4"), NOW)
    assert name == "20250307-090502_kernel_k4-sharpen_zero_norm-si_vista-recorte.png"
    assert re.fullmatch(r"\d{8}-\d{6}_[a-z]+_[\w.-]+_(zero|replicate|reflect|wrap)_[\w.-]+\.png", name)


def test_all_states_give_valid_windows_names():
    states = [AppState(kernel_id=k) for k in list(kernels.PRESETS) + ["0"]]
    states += [AppState(mode=st.MODE_CANNY, stage=s) for s in range(1, st.STAGE_COUNT + 1)]
    for s in states:
        name = capture.capture_name(s, NOW)
        assert name.isascii() and not re.search(r'[<>:"/\\|?*\s]', name), name


def test_unique_path(tmp_path):
    p = tmp_path / "a.png"
    assert capture.unique_path(p) == p
    p.write_bytes(b"")
    assert capture.unique_path(p) == tmp_path / "a_2.png"
    (tmp_path / "a_2.png").write_bytes(b"")
    assert capture.unique_path(p) == tmp_path / "a_3.png"


def test_save_png_creates_folder(tmp_path):
    rgb = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
    path = capture.save_png(rgb, tmp_path / "results" / "x.png")  # results/ no existe (R15.2)
    with Image.open(path) as img:
        assert img.size == (W, H) and img.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(img), rgb)


def test_save_png_write_error(tmp_path):
    blocker = tmp_path / "soy_un_archivo"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(capture.CaptureError, match="no se pudo escribir"):
        capture.save_png(np.zeros((H, W, 3), np.uint8), blocker / "x.png")


def test_app_capture_error_is_reported_and_app_continues(tmp_path, monkeypatch, capsys):
    """R15.4: un fallo de escritura muestra el error y no cierra la app (GPU simulada)."""
    monkeypatch.setattr(app_mod.glfw, "set_window_title", lambda w, t: None)
    settings = tmp_path / "settings.json"
    settings.write_text('{"custom_kernel": [0, 0, 0, 0, 1, 0, 0, 0, 0]}', encoding="utf-8")
    a = app_mod.App(np.zeros((H, W, 4), np.uint8), tmp_path / "img.png", settings_path=settings)

    class FakeProcessor:
        def run(self, state):
            return None, View()

    a.processor = FakeProcessor()
    a.quad = a.prog_display = None  # sin init(): render_display está simulado
    monkeypatch.setattr(capture, "render_display", lambda *args: np.zeros((H, W, 3), np.uint8))
    blocker = tmp_path / "bloqueo"
    blocker.write_text("x", encoding="utf-8")
    a.results_dir = blocker / "results"
    assert a.capture() is None
    assert "Error al guardar la captura" in capsys.readouterr().err
    a.results_dir = tmp_path / "results"
    path = a.capture()
    assert path is not None and path.parent == tmp_path / "results" and path.exists()
    assert "Captura guardada" in capsys.readouterr().out


# --- GPU --------------------------------------------------------------------


@pytest.fixture(scope="module")
def display(ctx):
    prog = load_program(ctx, "display.frag")
    yield prog
    prog.release()


def test_identity_capture_equals_input(ctx, quad, display, tmp_path):
    """Modo identidad: el PNG es la entrada exacta, W×H y con la fila 0 arriba (R15.3, R3.4)."""
    rng = np.random.default_rng(31)
    img = rng.integers(0, 256, size=(H, W, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[0, 0, :3] = (255, 0, 0)  # esquina superior izquierda roja
    tex = create_texture(ctx, img)
    proc = Processor(ctx, quad, tex)
    try:
        result, view = proc.run(AppState())
        rgb = capture.render_display(ctx, quad, display, result, view)
        path = capture.save_png(rgb, tmp_path / capture.capture_name(AppState(), NOW))
    finally:
        proc.release()
        tex.release()
    with Image.open(path) as png:
        assert png.size == (W, H)
        saved = np.asarray(png)
    np.testing.assert_array_equal(saved, img[..., :3])
    assert tuple(saved[0, 0]) == (255, 0, 0)


def test_capture_matches_screen_mapping(ctx, quad, display):
    """La captura aplica el mismo canal, mapeo y ganancia que la pantalla, sin invertir filas."""
    src = np.random.default_rng(32).uniform(-3.0, 3.0, size=(H, W, 4)).astype(np.float32)
    tex = create_texture(ctx, src)
    try:
        view = View(map=kernels.MAP_ABS, gain=0.25, channel=CH_R)
        rgb = capture.render_display(ctx, quad, display, tex, view)
    finally:
        tex.release()
    expected = np.clip(np.abs(src[..., 0]) * 0.25, 0.0, 1.0)
    for c in range(3):
        np.testing.assert_allclose(rgb[..., c] / 255.0, expected, atol=1.0 / 255.0 + 1e-6)


def test_capture_is_screen_flipped(ctx, quad, display):
    """Con u_flip_y = 1 (pantalla) las filas salen invertidas respecto a la captura: en
    pantalla la fila 0 del framebuffer es la de abajo, así que ambas muestran lo mismo."""
    from src.framebuffers import RenderTarget

    src = np.random.default_rng(33).uniform(0.0, 1.0, size=(H, W, 4)).astype(np.float32)
    tex = create_texture(ctx, src)
    screen_like = RenderTarget.create(ctx, (W, H), dtype="f1")
    try:
        cap = capture.render_display(ctx, quad, display, tex, View())
        run_pass(quad, display, screen_like, {"u_src": tex}, View().uniforms(flip_y=1))
        screen = screen_like.read()[..., :3]
    finally:
        screen_like.release()
        tex.release()
    np.testing.assert_array_equal(cap, screen[::-1])
