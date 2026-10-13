"""settings.json: carga, validación y recarga del kernel personalizado (R7.1–R7.4)."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from src import app as app_mod, kernels
from src.config import SETTINGS_PATH, ConfigError, SettingsWatcher, load_settings, parse_kernel

SHARPEN = (0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0)


def _write(path, data):
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


def test_repository_settings_is_valid():
    s = load_settings(SETTINGS_PATH)
    assert s.custom_kernel == SHARPEN
    assert s.data["default_image"] == "assets/images/orientacion.png"


def test_valid_nested_kernel(tmp_path):
    p = _write(tmp_path / "s.json", {"custom_kernel": [[1, 2, 3], [4, 5.5, 6], [7, 8, -9]]})
    assert load_settings(p).custom_kernel == (1, 2, 3, 4, 5.5, 6, 7, 8, -9)


def test_valid_flat_kernel():
    assert parse_kernel([0, 0, 0, 0, 1, 0, 0, 0, 0]) == kernels.IDENTITY


def test_bom_is_accepted(tmp_path):
    p = tmp_path / "s.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"custom_kernel": [[0] * 3, [0, 1, 0], [0] * 3]}).encode())
    assert load_settings(p).custom_kernel == kernels.IDENTITY


def test_syntax_error_reports_line_and_column(tmp_path):
    p = _write(tmp_path / "s.json", '{\n  "custom_kernel": [[0, 0, 0],\n    [0, 1 0], [0, 0, 0]]\n}\n')
    with pytest.raises(ConfigError) as info:
        load_settings(p)
    assert info.value.line == 3
    assert info.value.column == 11  # el "0" inesperado tras "1 "
    assert "línea 3, columna 11" in str(info.value)


@pytest.mark.parametrize(
    "kernel",
    [[[1, 2, 3], [4, 5, 6], [7, 8]], [1, 2, 3, 4, 5, 6, 7, 8]],
    ids=["anidado", "plano"],
)
def test_eight_values_rejected(tmp_path, kernel):
    p = _write(tmp_path / "s.json", {"custom_kernel": kernel})
    with pytest.raises(ConfigError, match="9 valores") as info:
        load_settings(p)
    assert "8" in str(info.value)


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_rejected(tmp_path, text):
    p = _write(tmp_path / "s.json", '{"custom_kernel": [[0, 0, 0], [0, %s, 0], [0, 0, 0]]}' % text)
    with pytest.raises(ConfigError, match="no es finito"):
        load_settings(p)


@pytest.mark.parametrize("bad", ['"1"', "true", "null", "[1]"])
def test_non_numbers_rejected(tmp_path, bad):
    p = _write(tmp_path / "s.json", '{"custom_kernel": [[0, 0, 0], [0, %s, 0], [0, 0, 0]]}' % bad)
    with pytest.raises(ConfigError):
        load_settings(p)


@pytest.mark.parametrize("data", ["[1, 2]", "{}", '{"custom_kernel": 5}'])
def test_bad_structure(tmp_path, data):
    with pytest.raises(ConfigError):
        load_settings(_write(tmp_path / "s.json", data))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="no se pudo leer"):
        load_settings(tmp_path / "no_existe.json")


def test_watcher_detects_changes(tmp_path):
    p = _write(tmp_path / "s.json", {"custom_kernel": list(kernels.IDENTITY)})
    w = SettingsWatcher(p)
    assert w.changed() is False
    _write(p, {"custom_kernel": list(SHARPEN)})
    st_ = os.stat(p)
    os.utime(p, ns=(st_.st_atime_ns, st_.st_mtime_ns + 2_000_000_000))  # mtime distinto seguro
    assert w.changed() is True
    assert w.changed() is False


# --- integración con App (sin ventana: GLFW simulado) ----------------------


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    titles = []
    monkeypatch.setattr(app_mod.glfw, "set_window_title", lambda w, t: titles.append(t))
    monkeypatch.setattr(app_mod.glfw, "set_window_should_close", lambda w, v: None)
    path = tmp_path / "settings.json"

    def make(kernel):
        _write(path, {"custom_kernel": kernel})
        a = app_mod.App(np.zeros((5, 7, 4), np.uint8), tmp_path / "img.png", settings_path=path)
        a.titles = titles
        return a, path

    return make


def _touch_later(path):
    st_ = os.stat(path)
    os.utime(path, ns=(st_.st_atime_ns, st_.st_mtime_ns + 2_000_000_000))


def test_app_loads_custom_kernel(make_app):
    a, _ = make_app([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    assert a.state.custom_kernel == (0, 0, 0, 0, 2, 0, 0, 0, 0)


def test_broken_json_keeps_previous_config(make_app, capsys):
    a, path = make_app([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    a.apply_key("0")
    before = a.state
    _write(path, '{"custom_kernel": [[0, 0, 0], [0, 3 0]]')
    _touch_later(path)
    a.poll_settings()
    assert a.state.custom_kernel == before.custom_kernel
    err = capsys.readouterr().err
    assert "línea 1" in err and "columna" in err
    assert "Se conserva la configuración anterior" in err
    # No repite el error en cada sondeo mientras el archivo no cambie.
    a.poll_settings()
    assert capsys.readouterr().err == ""


def test_invalid_kernel_keeps_previous_config(make_app, capsys):
    a, path = make_app(list(SHARPEN))
    a.apply_key("0")
    _write(path, {"custom_kernel": [1, 2, 3, 4, 5, 6, 7, 8]})
    a.apply_key("R")
    assert a.state.custom_kernel == SHARPEN
    assert "9 valores" in capsys.readouterr().err


def test_mtime_change_reloads_when_custom_active(make_app, capsys):
    a, path = make_app(list(SHARPEN))
    a.apply_key("0")
    a.state = replace(a.state, dirty=False)
    _write(path, {"custom_kernel": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]})
    _touch_later(path)
    a.poll_settings()
    assert a.state.custom_kernel == kernels.IDENTITY
    assert a.state.dirty  # se redibuja
    out = capsys.readouterr().out
    assert "Configuración recargada" in out
    assert "Kernel 0 (Personalizado)" in out


def test_mtime_ignored_while_preset_active(make_app):
    a, path = make_app(list(SHARPEN))
    _write(path, {"custom_kernel": list(kernels.IDENTITY)})
    _touch_later(path)
    a.poll_settings()
    assert a.state.custom_kernel == SHARPEN  # kernel 1 activo: no se vigila
    a.apply_key("0")  # al activar el personalizado se lee el archivo nuevo
    assert a.state.kernel_id == "0"
    assert a.state.custom_kernel == kernels.IDENTITY


def test_broken_settings_at_start_falls_back_to_identity(tmp_path, capsys):
    path = _write(tmp_path / "settings.json", "{roto")
    a = app_mod.App(np.zeros((5, 7, 4), np.uint8), tmp_path / "img.png", settings_path=path)
    assert a.state.custom_kernel == kernels.IDENTITY
    assert "línea 1" in capsys.readouterr().err


def test_apply_key_updates_title_and_help(make_app, capsys):
    a, _ = make_app(list(SHARPEN))
    a.apply_key("2")
    assert "Kernel: Box" in a.titles[-1]
    a.apply_key("H")
    assert "Controles:" in capsys.readouterr().out
    n = len(a.titles)
    a.apply_key("9")  # sin asignar: nada cambia
    assert len(a.titles) == n


def test_key_map_covers_state_keys():
    mapped = set(app_mod.KEY_MAP.values())
    assert {str(i) for i in range(9)} <= mapped
    assert {"B", "N", "V", "R", "H", "ESCAPE"} <= mapped
    assert {"M", "X", "C", "UP", "DOWN", "LEFT", "RIGHT", "PAGE_UP", "PAGE_DOWN"} <= mapped
    assert {"S", "G", "P"} <= mapped


def test_alpha_setting(tmp_path, make_app):
    assert load_settings(SETTINGS_PATH).alpha == 1.0
    p = _write(tmp_path / "s.json", {"custom_kernel": list(SHARPEN)})
    assert load_settings(p).alpha == 1.0  # opcional
    for bad in (-0.25, 5.5, "1"):
        _write(p, {"custom_kernel": list(SHARPEN), "alpha": bad})
        with pytest.raises(ConfigError, match="alpha"):
            load_settings(p)
    a, path = make_app(list(SHARPEN))
    _write(path, {"custom_kernel": list(SHARPEN), "alpha": 2.5})
    a = app_mod.App(np.zeros((5, 7, 4), np.uint8), tmp_path / "img.png", settings_path=path)
    assert a.state.alpha == 2.5


# --- σ y umbrales de Canny (R11.2, R12.7, R12.11) -------------------------------


def test_repository_numeric_settings():
    s = load_settings(SETTINGS_PATH)
    assert (s.sigma, s.canny_sigma, s.canny_high, s.canny_low_ratio) == (2.0, 1.4, 0.3, 0.4)


def test_numeric_settings_are_optional(tmp_path):
    s = load_settings(_write(tmp_path / "s.json", {"custom_kernel": list(SHARPEN)}))
    assert (s.sigma, s.canny_sigma, s.canny_high, s.canny_low_ratio) == (2.0, 1.4, 0.3, 0.4)


def test_defaults_agree_between_state_config_and_settings():
    """Mismo umbral alto por defecto en AppState, config.py y settings.json (S7)."""
    from src.state import AppState

    s = load_settings(SETTINGS_PATH)
    defaults = AppState()
    assert defaults.high == s.canny_high == 0.3
    assert defaults.low_ratio == s.canny_low_ratio == 0.4
    assert defaults.low == pytest.approx(0.12)


@pytest.mark.parametrize(
    "key, value",
    [("sigma", 0.4), ("sigma", 10.5), ("canny_sigma", "1.4"), ("canny_high", 0.0),
     ("canny_high", 1.5), ("canny_low_ratio", -0.1), ("canny_low_ratio", True)],
)
def test_numeric_settings_validated(tmp_path, key, value):
    p = _write(tmp_path / "s.json", {"custom_kernel": list(SHARPEN), key: value})
    with pytest.raises(ConfigError, match=key):
        load_settings(p)


def test_numeric_settings_non_finite(tmp_path):
    p = _write(tmp_path / "s.json", '{"custom_kernel": [1, 0, 0, 0, 0, 0, 0, 0, 0], "sigma": NaN}')
    with pytest.raises(ConfigError, match="sigma"):
        load_settings(p)


def _write_settings(path, **values):
    _write(path, {"custom_kernel": list(SHARPEN), **values})


def test_app_takes_initial_values_from_settings(make_app, tmp_path):
    a, path = make_app(list(SHARPEN))
    _write_settings(path, sigma=3.5, canny_sigma=2.0, canny_high=0.5, canny_low_ratio=0.3)
    a = app_mod.App(np.zeros((5, 7, 4), np.uint8), tmp_path / "img.png", settings_path=path)
    s = a.state
    assert (s.sigma, s.canny_sigma, s.high, s.low_ratio) == (3.5, 2.0, 0.5, 0.3)


def test_reload_applies_only_changed_keys(make_app):
    a, path = make_app(list(SHARPEN))
    _write_settings(path, sigma=2.0, canny_high=0.6, canny_low_ratio=0.4)
    a.apply_key("R")
    a.apply_key("C")
    a.apply_key("PAGE_UP")  # alto 0.65 con el teclado
    a.state = replace(a.state, dirty=False)
    _write_settings(path, sigma=2.0, canny_high=0.6, canny_low_ratio=0.25)
    a.apply_key("R")
    assert a.state.low_ratio == 0.25 and a.state.dirty
    assert a.state.high == pytest.approx(0.65)  # canny_high no cambió en el archivo
    _write_settings(path, sigma=2.0, canny_high=0.3, canny_low_ratio=0.25)
    a.apply_key("R")
    assert a.state.high == 0.3
    assert "alto=0.30" in a.titles[-1]


def test_on_key_repeat_only_for_arrows_and_page_keys(make_app):
    glfw = app_mod.glfw
    a, _ = make_app(list(SHARPEN))
    a.apply_key("M")  # Desenfoque
    sigma = a.state.sigma
    a._on_key(None, glfw.KEY_UP, 0, glfw.REPEAT, 0)
    assert a.state.sigma == sigma + 0.5
    a._on_key(None, glfw.KEY_X, 0, glfw.REPEAT, 0)  # X no se repite
    assert a.state.profile == kernels.PROFILE_GAUSS
    a._on_key(None, glfw.KEY_X, 0, glfw.PRESS, 0)
    assert a.state.profile == kernels.PROFILE_BOX
    a._on_key(None, glfw.KEY_UP, 0, glfw.RELEASE, 0)
    assert a.state.sigma == sigma + 0.5
