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
