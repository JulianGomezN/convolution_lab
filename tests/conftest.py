"""Fixtures de GPU: contexto OpenGL 3.3 sin ventana compartido por toda la sesión.

Se intenta `moderngl.create_standalone_context(require=330)` (WGL en Windows) y,
como respaldo, una ventana GLFW oculta. Si nada funciona, los tests de GPU se
saltan con el motivo (R17.3).
"""

from __future__ import annotations

import moderngl
import pytest
from hypothesis import settings

# Perfil común de Hypothesis: 100 ejemplos, sin base de datos de ejemplos
# (no crea `.hypothesis/`) y sin deadline (la GPU/IO puede tardar).
settings.register_profile("proyecto", max_examples=100, database=None, deadline=None)
settings.load_profile("proyecto")

_state: dict = {}


def _create_context():
    if "ctx" in _state or "error" in _state:
        return _state.get("ctx"), _state.get("error")
    errors = []
    try:
        _state["ctx"] = moderngl.create_standalone_context(require=330)
        _state["backend"] = "standalone"
        return _state["ctx"], None
    except Exception as exc:  # noqa: BLE001 - se informa el motivo en el skip
        errors.append(f"standalone: {exc}")
    try:
        import glfw

        if not glfw.init():
            raise RuntimeError("glfw.init() falló")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, False)
        window = glfw.create_window(16, 16, "tests", None, None)
        if not window:
            raise RuntimeError("no se pudo crear la ventana oculta")
        glfw.make_context_current(window)
        _state["window"] = window
        _state["ctx"] = moderngl.create_context(require=330)
        _state["backend"] = "glfw oculta"
        return _state["ctx"], None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"glfw oculta: {exc}")
    _state["error"] = "; ".join(errors)
    return None, _state["error"]


def pytest_report_header(config):
    ctx, error = _create_context()
    if ctx is None:
        return f"OpenGL: no disponible ({error}); los tests de GPU se saltarán"
    info = ctx.info
    return (
        f"OpenGL ({_state['backend']}): {info['GL_VERSION']} | GL_RENDERER: {info['GL_RENDERER']} "
        f"| GL_VENDOR: {info['GL_VENDOR']} | version_code: {ctx.version_code}"
    )


@pytest.fixture(scope="session")
def ctx():
    context, error = _create_context()
    if context is None:
        pytest.skip(f"sin contexto OpenGL 3.3: {error}")
    return context


@pytest.fixture(scope="session")
def quad(ctx):
    from src.render_pass import Quad

    q = Quad(ctx)
    yield q
    q.release()


def pytest_sessionfinish(session, exitstatus):
    ctx = _state.pop("ctx", None)
    if ctx is not None:
        ctx.release()
    window = _state.pop("window", None)
    if window is not None:
        import glfw

        glfw.destroy_window(window)
        glfw.terminate()
