"""Tests del Backend_Video (ImageioFrameSource) con un MP4 real.

Se saltan si imageio o imageio-ffmpeg no están instalados. `imageio` no se
importa en el nivel del módulo (R1.11): solo dentro del fixture.

Validates: Requirements 13.7, 13.8, 12.4, 12.5, 2.2
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest

from src.video import BACKEND_FFMPEG, ImageioFrameSource, open_source

W, H, N = 32, 24, 5
# Grises crecientes y bien separados: sobreviven a la compresión con pérdida.
LEVELS = (20, 70, 120, 170, 220)


def _has_backend() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("imageio", "imageio_ffmpeg"))


pytestmark = pytest.mark.skipif(not _has_backend(), reason="backend de vídeo no instalado")


@pytest.fixture
def mp4(tmp_path):
    import imageio.v2 as iio  # import perezoso

    frames = [np.full((H, W, 3), v, dtype=np.uint8) for v in LEVELS]
    path = tmp_path / "prueba.mp4"
    iio.mimwrite(str(path), frames, fps=10, macro_block_size=1)
    return path


def _mean(frame: np.ndarray) -> float:
    return float(frame[..., :3].mean())


def _assert_frame(frame, level):
    assert frame.shape == (H, W, 4)
    assert frame.dtype == np.uint8
    assert frame.flags["C_CONTIGUOUS"]
    assert np.all(frame[..., 3] == 255)
    assert abs(_mean(frame) - level) <= 12  # tolerancia por la pérdida del códec


def test_open_metadatos_y_frame0(mp4):
    with open_source(mp4) as src:
        assert isinstance(src, ImageioFrameSource)
        assert src.backend == BACKEND_FFMPEG
        assert src.n_frames == N
        assert src.size == (W, H)
        assert src.fps == pytest.approx(10.0, abs=0.1)
        _assert_frame(src.read(0), LEVELS[0])


def test_lectura_secuencial_orden_de_brillo(mp4):
    with open_source(mp4) as src:
        frames = [src.read(i) for i in range(N)]
    for frame, level in zip(frames, LEVELS):
        _assert_frame(frame, level)
    means = [_mean(f) for f in frames]
    assert means == sorted(means)


def test_retroceso_reabre_el_lector(mp4):
    with open_source(mp4) as src:
        last = src.read(N - 1)
        reader_antes = src._reader
        back = src.read(1)
        assert src._reader is not reader_antes  # se reabrió (V12)
        _assert_frame(back, LEVELS[1])
        assert _mean(back) < _mean(last)
        _assert_frame(src.read(0), LEVELS[0])


def test_eof_devuelve_none(mp4):
    with open_source(mp4) as src:
        assert src.read(N) is None
        assert src.read(N + 10) is None
        assert src.n_frames == N
        # tras el EOF se puede seguir leyendo frames válidos
        _assert_frame(src.read(N - 1), LEVELS[-1])


def test_eof_con_total_desconocido(mp4):
    with open_source(mp4) as src:
        src.n_frames = None  # simula un contenedor sin recuento
        _assert_frame(src.read(N - 1), LEVELS[-1])
        assert src.read(N) is None
        assert src.n_frames == N


def test_close_rapido_y_archivo_renombrable(mp4):
    src = open_source(mp4)
    src.read(2)
    t0 = time.perf_counter()
    src.close()
    assert time.perf_counter() - t0 < 2.0
    src.close()  # idempotente
    renamed = mp4.with_name("renombrado.mp4")
    mp4.rename(renamed)  # en Windows falla si ffmpeg mantiene el archivo abierto
    assert renamed.exists() and not mp4.exists()
