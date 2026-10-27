"""Tests de src/video.py (fuentes de frames para --video)."""

from __future__ import annotations

import os
import string
import subprocess
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from PIL import Image

from src.texture_loader import PROJECT_ROOT, ImageLoadError, load_image
from src.video import (
    BACKEND_PILLOW,
    VIDEO_SUFFIXES,
    PillowFrameSource,
    VideoOpenError,
    describe,
    open_source,
    safe_close,
)

# Feature: soporte-video, Property 17: Extensiones no admitidas
# **Validates: Requirements 3.2, 2.1, 2.2**

# Caracteres válidos en nombres de archivo de Windows (sin <>:"/\|?* ni puntos
# o espacios finales): letras, dígitos, "_" y "-".
_EXT_CHARS = string.ascii_letters + string.digits + "_-"

_unsupported_suffix = (
    st.text(alphabet=_EXT_CHARS, min_size=1, max_size=6)
    .map(lambda s: "." + s)
    .filter(lambda s: s.lower() not in VIDEO_SUFFIXES)
)

# Una de las 9 extensiones admitidas con cualquier combinación de mayúsculas.
_supported_suffix = st.sampled_from(VIDEO_SUFFIXES).flatmap(
    lambda s: st.tuples(*(st.sampled_from((c.lower(), c.upper())) for c in s)).map("".join)
)


@given(suffix=_unsupported_suffix)
def test_property_extension_no_admitida(tmp_path_factory, suffix):
    path = tmp_path_factory.mktemp("ext") / f"archivo{suffix}"
    path.write_bytes(b"no es un video")
    try:
        open_source(path)
    except VideoOpenError as exc:
        msg = str(exc)
        assert "no admitida" in exc.reason
        assert str(path) in msg
        for ext in VIDEO_SUFFIXES:
            assert ext in msg
    else:
        raise AssertionError(f"open_source aceptó la extensión {suffix!r}")


@given(suffix=_supported_suffix)
def test_property_extension_admitida_sin_error_de_extension(tmp_path_factory, suffix):
    # Archivo vacío: puede fallar al leerse o por falta del backend, pero nunca
    # con el error de extensión no admitida.
    path = tmp_path_factory.mktemp("ext") / f"archivo{suffix}"
    path.write_bytes(b"")
    try:
        source = open_source(path)
    except VideoOpenError as exc:
        assert "no admitida" not in exc.reason
    else:
        source.close()


# Feature: soporte-video, Property 11: Round trip del Backend_Pillow
# **Validates: Requirements 2.1, 5.3, 13.5**


@st.composite
def _apng_case(draw):
    w = draw(st.integers(min_value=1, max_value=16))
    h = draw(st.integers(min_value=1, max_value=16))
    n = draw(st.integers(min_value=3, max_value=6))
    # Frames distintos: Pillow fusiona frames idénticos consecutivos al guardar APNG.
    frames = draw(
        st.lists(
            arrays(np.uint8, (h, w, 4)),
            min_size=n,
            max_size=n,
            unique_by=lambda a: a.tobytes(),
        )
    )
    durations_ms = draw(
        st.lists(st.integers(min_value=1, max_value=50).map(lambda d: d * 10), min_size=n, max_size=n)
    )
    return frames, durations_ms


@settings(deadline=None)
@given(case=_apng_case())
def test_property_round_trip_backend_pillow(tmp_path_factory, case):
    frames, durations_ms = case
    h, w = frames[0].shape[:2]
    path = tmp_path_factory.mktemp("apng") / "anim.png"
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, format="PNG", save_all=True, append_images=images[1:], duration=durations_ms)

    source = open_source(path)
    try:
        assert isinstance(source, PillowFrameSource)
        assert source.backend == BACKEND_PILLOW
        assert source.n_frames == len(frames)
        assert source.size == (w, h)
        assert source.durations == pytest.approx([d / 1000 for d in durations_ms])
        for k, written in enumerate(frames):
            got = source.read(k)
            assert got.dtype == np.uint8
            assert got.flags["C_CONTIGUOUS"]
            assert got.shape == (h, w, 4)
            np.testing.assert_array_equal(got[..., :3], written[..., :3])
            assert np.all(got[..., 3] == 255)
    finally:
        source.close()


# --- Tests unitarios ---------------------------------------------------------

_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
_W, _H = 5, 3


def _solid_frames(n=3):
    return [Image.new("RGB", (_W, _H), _COLORS[k % len(_COLORS)]) for k in range(n)]


def _save_gif(path, durations_ms):
    frames = _solid_frames(len(durations_ms))
    frames[0].save(
        path, format="GIF", save_all=True, append_images=frames[1:], duration=durations_ms, loop=0
    )
    return path


def _save_apng(path, durations_ms):
    frames = _solid_frames(len(durations_ms))
    frames[0].save(path, format="PNG", save_all=True, append_images=frames[1:], duration=durations_ms)
    return path


def _check_frames(source, n):
    assert source.n_frames == n
    assert source.size == (_W, _H)
    for k in range(n):
        frame = source.read(k)
        assert frame.dtype == np.uint8
        assert frame.shape == (_H, _W, 4)
        assert np.all(frame[..., 3] == 255)
        assert tuple(int(c) for c in frame[0, 0, :3]) == _COLORS[k]
    assert source.read(n) is None


def test_gif_animado(tmp_path):
    durations_ms = [100, 200, 50]  # múltiplos de 10 ms (resolución del GIF)
    path = _save_gif(tmp_path / "anim.gif", durations_ms)
    with open_source(path) as source:
        assert source.backend == BACKEND_PILLOW
        assert source.durations == pytest.approx([d / 1000 for d in durations_ms])
        _check_frames(source, 3)


def test_apng_animado(tmp_path):
    durations_ms = [40, 80, 120, 30]
    path = _save_apng(tmp_path / "anim.apng", durations_ms)
    with open_source(path) as source:
        assert source.durations == pytest.approx([d / 1000 for d in durations_ms])
        _check_frames(source, 4)


def test_png_estatico_es_video_de_un_frame(tmp_path):
    path = tmp_path / "fijo.png"
    Image.new("RGB", (_W, _H), _COLORS[0]).save(path)
    with open_source(path) as source:
        _check_frames(source, 1)
        assert len(source.durations) == 1


def test_load_image_de_gif_animado_devuelve_frame_0(tmp_path):
    path = _save_gif(tmp_path / "anim.gif", [100, 100, 100])
    img = load_image(path)
    assert img.shape == (_H, _W, 4)
    assert img.dtype == np.uint8
    assert tuple(int(c) for c in img[0, 0, :3]) == _COLORS[0]
    assert np.all(img[..., 3] == 255)


def test_load_image_de_mp4_pide_usar_video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    with pytest.raises(ImageLoadError) as info:
        load_image(path)
    assert "--video" in info.value.reason
    assert "vídeo" in info.value.reason


def test_open_source_inexistente(tmp_path):
    path = tmp_path / "no_existe.gif"
    with pytest.raises(VideoOpenError) as info:
        open_source(path)
    assert info.value.reason == "el archivo no existe"
    assert "No se pudo abrir el vídeo" in str(info.value)
    assert str(path) in str(info.value)


def test_open_source_directorio(tmp_path):
    path = tmp_path / "carpeta.gif"
    path.mkdir()
    with pytest.raises(VideoOpenError) as info:
        open_source(path)
    assert info.value.reason == "la ruta no es un archivo"


def test_open_source_extension_no_admitida(tmp_path):
    path = tmp_path / "foto.jpg"
    Image.new("RGB", (2, 2)).save(path)
    with pytest.raises(VideoOpenError) as info:
        open_source(path)
    assert "no admitida" in info.value.reason
    assert ".jpg" in info.value.reason


def test_open_source_sin_backend_video(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio", None)
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 16)
    with pytest.raises(VideoOpenError) as info:
        open_source(path)
    reason = info.value.reason
    assert "imageio" in reason
    assert "pip install -r requirements-video.txt" in reason


@pytest.mark.parametrize("contenido", [b"", b"esto no es un gif"])
def test_open_source_archivo_vacio_o_corrupto(tmp_path, contenido):
    path = tmp_path / "malo.gif"
    path.write_bytes(contenido)
    with pytest.raises(VideoOpenError) as info:
        open_source(path)
    assert info.value.reason.startswith("no se pudo")
    assert "(" in info.value.reason  # incluye la causa


def test_open_source_frame0_truncado(tmp_path):
    full = tmp_path / "completo.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(full)
    path = tmp_path / "truncado.png"
    path.write_bytes(full.read_bytes()[:200])
    with pytest.raises(VideoOpenError) as info:
        open_source(path)
    assert info.value.reason.startswith("no se pudo")
    # El archivo queda liberado aunque open() haya fallado.
    os.replace(path, tmp_path / "renombrado.png")


def test_close_idempotente_y_libera_archivo(tmp_path):
    path = _save_gif(tmp_path / "anim.gif", [100, 100, 100])
    source = open_source(path)
    source.close()
    source.close()
    target = tmp_path / "renombrado.gif"
    os.replace(path, target)
    assert target.exists()


def test_safe_close_con_close_que_lanza():
    class Rota:
        def close(self):
            raise RuntimeError("boom")

    avisos = []
    safe_close(Rota(), warn=avisos.append)
    assert len(avisos) == 1
    assert avisos[0].startswith("Aviso:")
    assert "boom" in avisos[0]
    safe_close(None, warn=avisos.append)
    assert len(avisos) == 1


def test_describe(tmp_path):
    path = _save_gif(tmp_path / "anim.gif", [100, 100, 100])
    with open_source(path) as source:
        line = describe(source)
    assert line == f"Vídeo: anim.gif | {_W}x{_H} | 3 frames | 10.00 fps | backend Pillow"
    line.encode("cp1252")  # solo caracteres de cp1252


def test_imports_no_cargan_imageio():
    code = "import sys, src.video, src.app; print('imageio' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
