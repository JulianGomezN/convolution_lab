"""Vídeo en GPU (R5.5, R4.4, R13.9, R13.10): el Player sube cada frame a la misma
Textura_de_Entrada y el Processor da lo mismo que con el frame cargado como imagen fija.

Usa los fixtures `ctx` y `quad` de conftest.py: sin contexto OpenGL 3.3 estos tests
se saltan con el motivo explícito del fixture (R13.10).
"""

from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from src import capture
from src.pipeline import Processor
from src.playback import EffectiveFps, PlaybackClock, Player, Timeline
from src.shader_loader import load_program
from src.state import MODE_BLUR, MODE_CANNY, MODE_KERNEL, MODE_UNSHARP, AppState
from src.texture_loader import create_texture
from src.video import PillowFrameSource
from tests.reference import NP_MODES

W, H = 7, 5  # impar y no cuadrado: cualquier flip o transposición cambia el resultado
N_FRAMES = 3
FRAME_MS = 100

STATES = {
    "kernel5": lambda b: AppState(mode=MODE_KERNEL, kernel_id="5", border=b),
    "blur": lambda b: AppState(mode=MODE_BLUR, sigma=1.5, border=b),
    "unsharp": lambda b: AppState(mode=MODE_UNSHARP, sigma=1.5, alpha=1.5, border=b),
    "canny11": lambda b: AppState(mode=MODE_CANNY, stage=11, high=0.1, border=b),
}


def _frames() -> list[np.ndarray]:
    """3 frames RGB aleatorios distintos (no simétricos en ningún eje)."""
    rng = np.random.default_rng(1234)
    frames = [rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8) for _ in range(N_FRAMES)]
    for f in frames:
        assert not np.array_equal(f, f[::-1]) and not np.array_equal(f, f[:, ::-1])
    return frames


@pytest.fixture
def apng(tmp_path):
    frames = _frames()
    path = tmp_path / "anim.png"
    imgs = [Image.fromarray(f, "RGB") for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=FRAME_MS, loop=0)
    rgba = [np.dstack([f, np.full((H, W), 255, np.uint8)]) for f in frames]
    return path, rgba


def _read(tex) -> np.ndarray:
    w, h = tex.size
    return np.frombuffer(tex.read(), dtype=np.float32).reshape(h, w, 4).copy()


def _read_rgba8(tex) -> np.ndarray:
    w, h = tex.size
    return np.frombuffer(tex.read(), dtype=np.uint8).reshape(h, w, 4).copy()


@contextmanager
def _video(ctx, quad, path):
    """Fuente Pillow + Textura_de_Entrada del frame 0 + Player (pausado en t = 0) + Processor."""
    source = PillowFrameSource(path)
    source.open()
    input_tex = proc = None
    try:
        input_tex = create_texture(ctx, source.read(0))
        clock = PlaybackClock(Timeline.from_durations(source.durations), now=0.0, playing=False)
        player = Player(source, input_tex, clock, EffectiveFps(0.0), warn=lambda msg: None)
        proc = Processor(ctx, quad, input_tex)
        yield player, input_tex, proc
    finally:
        if proc is not None:
            proc.release()
        if input_tex is not None:
            input_tex.release()
        source.close()


def _still_output(ctx, quad, frame, state) -> np.ndarray:
    tex = create_texture(ctx, frame)
    proc = Processor(ctx, quad, tex)
    try:
        return _read(proc.run(state)[0])
    finally:
        proc.release()
        tex.release()


@pytest.mark.parametrize("border", [0, 1, 2, 3], ids=lambda b: NP_MODES[b])
@pytest.mark.parametrize("name", list(STATES))
def test_player_frame_matches_still_image(ctx, quad, apng, name, border):
    """R5.5: frame k subido por el Player ≡ create_texture(frame_k), error ≤ 1e-6."""
    path, frames = apng
    state = STATES[name](border)
    with _video(ctx, quad, path) as (player, input_tex, proc):
        assert player.source.n_frames == N_FRAMES and player.source.size == (W, H)
        for k in range(N_FRAMES):
            if k > 0:
                now = float(k)
                player.clock.step(+1, now)
                assert player.update(now).frame
            assert player.shown_index == k
            assert np.array_equal(_read_rgba8(input_tex), frames[k])
            got = _read(proc.run(state)[0])
            expected = _still_output(ctx, quad, frames[k], state)
            assert np.max(np.abs(got - expected)) <= 1e-6, f"frame {k}"
    # la textura de entrada cambia de verdad entre frames (Canny puede dar el mismo
    # mapa de bordes en frames distintos, así que se comprueba la entrada)
    assert not np.array_equal(frames[0], frames[1]) and not np.array_equal(frames[1], frames[2])


def test_input_texture_is_reused_across_updates(ctx, quad, apng):
    """R4.4, R13.9: tras 3+ updates (con vuelta al inicio en bucle) es la misma textura."""
    path, frames = apng
    with _video(ctx, quad, path) as (player, input_tex, proc):
        glo, size = input_tex.glo, input_tex.size
        player.clock.toggle_play(0.0)  # reproduciendo con bucle desde t = 0
        dt = FRAME_MS / 1000.0
        expected_idx = [1, 2, 0, 1]  # en t = 0.15, 0.25, 0.35 (bucle), 0.45 s
        for i, k in enumerate(expected_idx):
            now = (i + 1.5) * dt
            changes = player.update(now)
            assert changes.frame and player.shown_index == k
            proc.run(AppState(mode=MODE_KERNEL, kernel_id="5"))
            assert proc.source is input_tex
            assert player.texture is input_tex
            assert input_tex.glo == glo and input_tex.size == size == (W, H)
            assert np.array_equal(_read_rgba8(input_tex), frames[k])


@pytest.fixture(scope="module")
def display(ctx):
    prog = load_program(ctx, "display.frag")
    yield prog
    prog.release()


def test_capture_of_frame_2_is_png_with_frame_suffix(ctx, quad, display, apng, tmp_path):
    """R10.1, R10.2, R4.5: la captura del frame 2 es un PNG W×H `_f00002` con ese frame (identidad)."""
    path, frames = apng
    state = AppState(mode=MODE_KERNEL, kernel_id="1")  # identidad
    now = datetime(2025, 3, 7, 9, 5, 2)
    with _video(ctx, quad, path) as (player, input_tex, proc):
        for t in (1.0, 2.0):  # frame 0 → 1 → 2 paso a paso (pausado)
            player.clock.step(+1, t)
            assert player.update(t).frame
        assert player.shown_index == 2
        result, view = proc.run(state)
        rgb = capture.render_display(ctx, quad, display, result, view, original=input_tex)
        name = capture.capture_name(state, now, frame_index=player.shown_index)
        out = capture.save_png(rgb, tmp_path / name)
    assert out.name == name and name.endswith("_f00002.png")
    assert name == capture.capture_name(state, now)[: -len(".png")] + "_f00002.png"
    with Image.open(out) as png:
        assert png.format == "PNG" and png.size == (W, H)
        saved = np.asarray(png.convert("RGB"))
    np.testing.assert_array_equal(saved, frames[2][..., :3])
    assert not np.array_equal(saved, frames[1][..., :3])
