"""CLI y códigos de salida (R1.3, R4.2, R4.3)."""

import subprocess
import sys
from pathlib import Path

import pytest

import main
from src import app

ROOT = Path(__file__).resolve().parent.parent


def test_missing_image_exits_2(tmp_path, capsys):
    missing = tmp_path / "no_existe.png"
    assert main.main(["--image", str(missing)]) == 2
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "no existe" in err
    assert "orientacion.png" in err  # lista de assets/images


def test_invalid_image_exits_2(tmp_path, capsys):
    bad = tmp_path / "falsa.png"
    bad.write_text("no soy una imagen", encoding="utf-8")
    assert main.main(["--image", str(bad)]) == 2
    assert "no es una imagen válida" in capsys.readouterr().err


def test_missing_image_subprocess_without_traceback(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--image", str(tmp_path / "nada.png")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        timeout=60,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "Error de imagen" in proc.stderr


def test_texture_limit_is_image_error():
    """R3.3: si la imagen supera GL_MAX_TEXTURE_SIZE se trata como error de imagen (código 2)."""
    from src.texture_loader import ImageLoadError

    app.check_texture_limit("a.png", (16384, 10), 16384)  # en el límite: válido
    with pytest.raises(ImageLoadError, match="GL_MAX_TEXTURE_SIZE"):
        app.check_texture_limit("a.png", (10, 16385), 16384)


def test_window_size_leaves_room_for_full_state_title():
    # Panel lateral + imagen a su derecha; el alto nunca baja de PANEL_MIN_HEIGHT.
    assert app.window_size((640, 480)) == (app.PANEL_WIDTH + 640, app.PANEL_MIN_HEIGHT)
    assert app.window_size((1920, 1080)) == (app.PANEL_WIDTH + 1920, 1080)


def test_panel_min_height_fits_tallest_panel():
    """PANEL_MIN_HEIGHT cubre el panel más alto (modo Canny) con margen inferior.

    Se reconstruye el desglose de alturas del panel tal como lo acumula la variable
    `y` en App._update_status_texture, para el estado MÁS ALTO (Canny: 3 filas de
    parámetros). Si alguien añade secciones o filas y supera el mínimo, este test
    falla antes de que CONTROLES quede recortado en pantalla.
    """
    y = 18  # margen superior inicial
    y += 24 + 30  # título en dos líneas
    # Cada sección: header (+20) y luego sus valores/filas; gap() añade 12 al final.
    y += 20 + 22 + 22 + 12  # IMAGEN: header + nombre + tamaño
    y += 20 + 22 + 22 + 12  # MODO: header + nombre del modo + resumen
    canny_param_rows = 3  # Sigma, Alto/bajo, Método (el modo con más filas)
    y += 20 + canny_param_rows * 20 + 12  # PARÁMETROS
    y += 20 + 22 + 4 * 18 + 12  # BORDE: header + activo + leyenda de 4 bordes
    y += 20 + 22 + 12  # LUPA: header + etiqueta
    y += 20 + 3 * 18  # CONTROLES: header + 3 líneas de atajos

    estimated_panel_height = y
    bottom_margin = 20  # holgura mínima bajo el contenido
    assert app.PANEL_MIN_HEIGHT >= estimated_panel_height + bottom_margin


def test_image_viewport():
    # Imagen pegada a la derecha del panel, centrada en vertical si sobra alto.
    assert app.image_viewport((app.PANEL_WIDTH + 640, 520), (640, 480)) == (app.PANEL_WIDTH, 20, 640, 480)
    # Imagen tan alta como la ventana: sin desplazamiento vertical.
    assert app.image_viewport((app.PANEL_WIDTH + 640, 720), (640, 720)) == (app.PANEL_WIDTH, 0, 640, 720)


def test_default_image_comes_from_settings():
    assert main.default_image() == ROOT / "assets" / "images" / "orientacion.png"


def test_help_is_spanish(capsys):
    with pytest.raises(SystemExit) as info:
        main.main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "uso:" in out and "opciones" in out and "--image" in out


def test_glfw_init_failure_exits_3(monkeypatch, capsys):
    monkeypatch.setattr(app.glfw, "init", lambda: False)
    monkeypatch.setattr(app.glfw, "terminate", lambda: None)
    assert main.main(["--image", str(ROOT / "assets" / "images" / "escalon.png")]) == 3
    err = capsys.readouterr().err
    assert "No se pudo inicializar GLFW" in err
    assert "Escritorio Remoto" in err
    assert "Traceback" not in err


def test_window_creation_failure_exits_3(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(app.glfw, "init", lambda: True)
    monkeypatch.setattr(app.glfw, "window_hint", lambda *a: calls.append(a))
    monkeypatch.setattr(app.glfw, "create_window", lambda *a: None)
    monkeypatch.setattr(app.glfw, "terminate", lambda: None)
    assert main.main(["--image", str(ROOT / "assets" / "images" / "escalon.png")]) == 3
    assert "No se pudo crear la ventana" in capsys.readouterr().err
    hints = dict(calls)
    assert hints[app.glfw.CONTEXT_VERSION_MAJOR] == 3
    assert hints[app.glfw.CONTEXT_VERSION_MINOR] == 3
    assert hints[app.glfw.OPENGL_PROFILE] == app.glfw.OPENGL_CORE_PROFILE
    assert hints[app.glfw.OPENGL_FORWARD_COMPAT] is True
    assert hints[app.glfw.RESIZABLE] is False


def _fake_window(monkeypatch):
    """GLFW simulado: init y ventana correctos, sin abrir nada real."""
    monkeypatch.setattr(app.glfw, "init", lambda: True)
    monkeypatch.setattr(app.glfw, "window_hint", lambda *a: None)
    monkeypatch.setattr(app.glfw, "create_window", lambda *a: object())
    monkeypatch.setattr(app.glfw, "make_context_current", lambda w: None)
    monkeypatch.setattr(app.glfw, "swap_interval", lambda n: None)
    monkeypatch.setattr(app.glfw, "destroy_window", lambda w: None)
    monkeypatch.setattr(app.glfw, "terminate", lambda: None)


def test_moderngl_context_failure_exits_3(monkeypatch, capsys):
    _fake_window(monkeypatch)

    def fail(require):
        raise RuntimeError("contexto simulado no disponible")

    monkeypatch.setattr(app.moderngl, "create_context", fail)
    assert main.main(["--image", str(ROOT / "assets" / "images" / "escalon.png")]) == 3
    err = capsys.readouterr().err
    assert "No se pudo crear el contexto moderngl con OpenGL 3.3" in err
    assert "contexto simulado no disponible" in err
    assert "Escritorio Remoto" in err
    assert "Traceback" not in err


def test_old_opengl_version_exits_3(monkeypatch, capsys):
    _fake_window(monkeypatch)
    released = []

    class OldContext:
        version_code = 320

        def release(self):
            released.append(True)

    monkeypatch.setattr(app.moderngl, "create_context", lambda require: OldContext())
    assert main.main(["--image", str(ROOT / "assets" / "images" / "escalon.png")]) == 3
    err = capsys.readouterr().err
    assert "Se requiere OpenGL 3.3 y el contexto ofrece 320" in err
    assert "Traceback" not in err
    assert released == [True]  # el contexto se libera aunque falle la validación


# --- soporte de vídeo: CLI y panel (R2.3, R2.5, R2.7, R3.1–R3.4, R8.9, R13.6) ---

import types
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from src import state as st
from src.video import VIDEO_SUFFIXES


@pytest.fixture
def no_window(monkeypatch):
    """Falla el test si se intenta crear la ventana (App.init o glfw.init)."""

    def forbidden(*args, **kwargs):
        pytest.fail("no se debe crear ventana ante un error de apertura")

    monkeypatch.setattr(app.App, "init", forbidden)
    monkeypatch.setattr(app.glfw, "init", forbidden)


def test_video_and_image_are_mutually_exclusive(capsys):
    """R2.3: --video con --image → código 2 y mensaje de argparse en español."""
    with pytest.raises(SystemExit) as info:
        main.main(["--video", "clip.gif", "--image", "foto.png"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "uso:" in err
    assert "no se permite junto con el argumento" in err


def test_help_mentions_video_and_extensions(monkeypatch, capsys):
    """R2.7: la ayuda menciona --video RUTA, --image y las 9 extensiones."""
    monkeypatch.setenv("COLUMNS", "200")
    with pytest.raises(SystemExit) as info:
        main.main(["--help"])
    assert info.value.code == 0
    out = " ".join(capsys.readouterr().out.split())  # argparse puede partir líneas
    assert "--video RUTA" in out and "--image" in out and "RUTA" in out
    assert len(VIDEO_SUFFIXES) == 9
    for suffix in VIDEO_SUFFIXES:
        assert suffix in out, suffix


def _truncated_png(path):
    """PNG con cabecera válida y datos cortados: Image.open funciona, decodificar no."""
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    return path


@pytest.mark.parametrize(
    "case, reason",
    [
        ("missing", "el archivo no existe"),  # R3.1
        ("directory", "la ruta no es un archivo"),  # R3.1
        ("bad_suffix", "no admitida"),  # R3.2
        ("no_backend", "falta(n) imageio"),  # R3.3
        ("corrupt_frame0", "no se pudo decodificar el primer frame"),  # R3.4
    ],
)
def test_video_open_errors_exit_2_without_window(case, reason, tmp_path, monkeypatch, capsys, no_window):
    if case == "missing":
        path = tmp_path / "no_existe.mp4"
    elif case == "directory":
        path = tmp_path / "carpeta.gif"
        path.mkdir()
    elif case == "bad_suffix":
        path = tmp_path / "clip.xyz"
        path.write_bytes(b"datos")
    elif case == "no_backend":
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"datos")
        monkeypatch.setitem(sys.modules, "imageio", None)  # import imageio → ImportError
    else:
        path = _truncated_png(tmp_path / "roto.png")
    assert main.main(["--video", str(path)]) == 2
    err = capsys.readouterr().err
    assert "Error de vídeo:" in err
    assert reason in err
    assert str(path) in err
    assert "Traceback" not in err


def test_image_with_video_suffix_exits_2_without_window(tmp_path, capsys, no_window):
    """R2.5: --image clip.mp4 → error de imagen que sugiere --video."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"datos")
    assert main.main(["--image", str(path)]) == 2
    err = capsys.readouterr().err
    assert "Error de imagen:" in err
    assert "--video" in err
    assert "Traceback" not in err


def _fake_player(n=12345, k=12345, start=999.99, fps=999.9):
    """Player falso con lo que leen _draw_panel/_video_rows (valores de máxima longitud)."""
    timeline = types.SimpleNamespace(n_frames=n, start=lambda index: start)
    clock = types.SimpleNamespace(timeline=timeline, playing=True, loop=True)
    return types.SimpleNamespace(
        clock=clock,
        shown_index=k,
        source=types.SimpleNamespace(fps=fps),
        fps=types.SimpleNamespace(value=lambda now: fps),
    )


def test_panel_min_height_fits_tallest_panel_with_video(monkeypatch):
    """R8.9, R13.6: el panel con VÍDEO en modo Canny cabe en alto y en ancho."""
    frame0 = np.zeros((1080, 1920, 4), dtype=np.uint8)
    video_app = app.App(frame0, Path("animacion_de_prueba_larga.webm"), source=object(), now=lambda: 100.0)
    video_app.player = _fake_player()
    video_app.state = replace(video_app.state, mode=st.MODE_CANNY, high=0.99, low_ratio=0.99)
    video_app._init_fonts()

    lines = []
    original_text = ImageDraw.ImageDraw.text

    def recording_text(self, xy, text, *args, font=None, **kwargs):
        lines.append((xy[0], text, font))
        return original_text(self, xy, text, *args, font=font, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", recording_text)
    _, content_height = video_app._draw_panel()

    assert content_height + 20 <= app.PANEL_MIN_HEIGHT
    texts = [text for _, text, _ in lines]
    assert "1920 x 1080 · frame 12345/12345 · 999.99 s" in texts
    assert "Reproduciendo · bucle: sí" in texts
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    for x, text, font in lines:
        assert x < app.PANEL_WIDTH
        right = x + measure.textlength(text, font=font)
        assert right <= app.PANEL_WIDTH - 16, (text, right)
