"""Carga de imagen (R4.1, R4.3, R4.5) y generador de imágenes de prueba."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.texture_loader import ImageLoadError, available_images, load_image

ROOT = Path(__file__).resolve().parent.parent


def test_missing_file_raises_with_path_reason_and_listing(tmp_path):
    missing = tmp_path / "no_existe.png"
    with pytest.raises(ImageLoadError) as info:
        load_image(missing)
    msg = str(info.value)
    assert str(missing) in msg
    assert "no existe" in msg
    assert "Imágenes disponibles" in msg
    assert info.value.available == available_images()


def test_directory_raises(tmp_path):
    with pytest.raises(ImageLoadError, match="no es un archivo"):
        load_image(tmp_path)


def test_invalid_file_raises(tmp_path):
    bad = tmp_path / "falsa.png"
    bad.write_bytes(b"esto no es una imagen")
    with pytest.raises(ImageLoadError, match="no es una imagen válida"):
        load_image(bad)


def test_truncated_file_raises(tmp_path):
    good = tmp_path / "buena.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(good)
    truncated = tmp_path / "truncada.png"
    truncated.write_bytes(good.read_bytes()[:60])
    with pytest.raises(ImageLoadError):
        load_image(truncated)


@pytest.mark.parametrize("mode", ["L", "P", "RGB", "RGBA", "LA"])
def test_modes_convert_to_rgba8_with_opaque_alpha(tmp_path, mode):
    w, h = 7, 5  # no cuadrada, para detectar ejes intercambiados
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    src = Image.fromarray(rgb)
    if mode == "RGBA":
        rgba = np.dstack([rgb, np.zeros((h, w), np.uint8)])  # alfa 0: debe ignorarse
        src = Image.fromarray(rgba)
    elif mode != "RGB":
        src = src.convert(mode)
    path = tmp_path / f"img_{mode}.png"
    src.save(path)

    arr = load_image(path)
    assert arr.shape == (h, w, 4)
    assert arr.dtype == np.uint8
    assert arr.flags["C_CONTIGUOUS"]
    assert np.all(arr[..., 3] == 255)
    expected = np.asarray(Image.open(path).convert("RGB"))
    np.testing.assert_array_equal(arr[..., :3], expected)
    if mode in ("RGB", "RGBA"):
        np.testing.assert_array_equal(arr[..., :3], rgb)


def test_row_zero_is_top(tmp_path):
    arr = np.zeros((5, 7, 3), np.uint8)
    arr[0, 0] = (255, 0, 0)  # arriba a la izquierda
    path = tmp_path / "esquina.png"
    Image.fromarray(arr).save(path)
    out = load_image(path)
    assert tuple(out[0, 0, :3]) == (255, 0, 0)
    assert out[..., :3].sum() == 255


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_test_images", ROOT / "tools" / "make_test_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_writes_all_images(tmp_path):
    gen = _load_generator()
    written = gen.generate(tmp_path)
    names = {p.name for p in written}
    assert names == {"orientacion.png", "escalon.png", "tablero.png", "circulo.png", "ruido.png"}
    for p in written:
        arr = load_image(p)
        assert arr.ndim == 3 and arr.shape[2] == 4

    orient = load_image(tmp_path / "orientacion.png")
    h, w = orient.shape[:2]
    assert (w, h) == (640, 480)
    assert tuple(orient[0, 0, :3]) == (255, 0, 0)  # marcador rojo arriba-izquierda
    assert tuple(orient[h - 1, w - 1, :3]) == (0, 200, 0)  # verde abajo-derecha
    # "F" asimétrica: la barra superior es más larga que la media.
    white = orient[..., 0] == 240
    assert white[100].sum() > white[230].sum() > 0

    step = load_image(tmp_path / "escalon.png")
    assert step[0, 0, 0] == 0 and step[0, -1, 0] == 255


def test_generator_is_deterministic(tmp_path):
    gen = _load_generator()
    a = tmp_path / "a"
    b = tmp_path / "b"
    gen.generate(a)
    gen.generate(b)
    for name in gen.GENERATORS:
        np.testing.assert_array_equal(load_image(a / name), load_image(b / name))


def test_default_assets_exist():
    for name in ("orientacion.png", "escalon.png", "tablero.png", "circulo.png", "ruido.png"):
        assert name in available_images(), f"falta assets/images/{name}: ejecuta tools/make_test_images.py"


@pytest.mark.parametrize("name", ["clip.mp4", "clip.MP4", "a.avi", "a.mkv", "a.Mov", "a.webm"])
def test_video_suffix_rejected_before_existence_check(tmp_path, name):
    # R2.5: el archivo no existe, pero el motivo es la extensión de vídeo.
    with pytest.raises(ImageLoadError) as info:
        load_image(tmp_path / name)
    assert "--video" in str(info.value)
    assert "vídeo" in info.value.reason
    assert "no existe" not in info.value.reason


def test_animated_gif_loads_frame_zero(tmp_path):
    # R2.4: un GIF animado por --image sigue cargando solo el frame 0.
    w, h = 6, 4
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    frames = [Image.new("RGB", (w, h), c) for c in colors]
    path = tmp_path / "anim.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    with Image.open(path) as check:
        assert check.n_frames == 3

    arr = load_image(path)
    assert arr.shape == (h, w, 4)
    assert arr.dtype == np.uint8
    assert np.all(arr[..., 3] == 255)
    assert np.all(arr[..., :3] == np.array(colors[0], np.uint8))
