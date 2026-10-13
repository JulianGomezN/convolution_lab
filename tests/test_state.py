"""Estado y teclado, sin GPU (R6.1, R6.3, R8.3, R9.2–R9.4, R10.2, R16.2–R16.5)."""

from dataclasses import replace
from pathlib import Path

import pytest

from src import kernels, state as st
from src.state import AppState, handle_key

ROOT = Path(__file__).resolve().parent.parent


def test_defaults():
    s = AppState()
    assert s.mode == st.MODE_KERNEL
    assert s.kernel_id == "1"
    assert s.normalize is True  # activa por defecto (R8.3)
    assert s.border == st.BORDER_ZERO
    assert s.display_map == kernels.MAP_CLIP
    assert s.dirty is True


@pytest.mark.parametrize("key", [str(i) for i in range(1, 9)])
def test_keys_1_to_8_select_preset_and_default_map(key):
    s0 = replace(AppState(kernel_id="1", display_map=kernels.MAP_OFFSET), dirty=False)
    s1, action = handle_key(s0, key)
    assert action is None
    assert s1.kernel_id == key
    assert s1.mode == st.MODE_KERNEL
    assert s1.display_map == kernels.PRESETS[key].default_map  # R9.3
    assert s1.dirty is True


def test_key_0_selects_custom():
    s1, action = handle_key(replace(AppState(kernel_id="5", display_map=kernels.MAP_ABS), dirty=False), "0")
    assert action is None
    assert s1.kernel_id == "0"
    assert s1.display_map == kernels.MAP_CLIP
    assert s1.dirty


def test_b_cycles_borders():
    s = replace(AppState(), dirty=False)
    seen = []
    for _ in range(5):
        s, action = handle_key(s, st.KEY_B)
        assert action is None and s.dirty
        seen.append(st.BORDER_NAMES[s.border])
        s = replace(s, dirty=False)
    assert seen == ["replicate", "reflect", "wrap", "zero", "replicate"]


def test_n_toggles_normalization():
    s, _ = handle_key(replace(AppState(), dirty=False), st.KEY_N)
    assert s.normalize is False and s.dirty
    s, _ = handle_key(s, st.KEY_N)
    assert s.normalize is True


def test_v_cycles_display_map():
    s = replace(AppState(), dirty=False)
    maps = []
    for _ in range(4):
        s, _ = handle_key(s, st.KEY_V)
        maps.append(s.display_map)
    assert maps == [kernels.MAP_ABS, kernels.MAP_OFFSET, kernels.MAP_CLIP, kernels.MAP_ABS]
    assert s.dirty


def test_v_keeps_kernel_and_border():
    s0 = AppState(kernel_id="6", border=2)
    s1, _ = handle_key(s0, st.KEY_V)
    assert (s1.kernel_id, s1.border, s1.normalize) == (s0.kernel_id, s0.border, s0.normalize)


@pytest.mark.parametrize(
    "key, action",
    [(st.KEY_R, st.ACTION_RELOAD), (st.KEY_H, st.ACTION_HELP), (st.KEY_ESCAPE, st.ACTION_QUIT)],
)
def test_action_keys_do_not_change_state(key, action):
    s0 = replace(AppState(), dirty=False)
    s1, got = handle_key(s0, key)
    assert got == action
    assert s1 is s0


@pytest.mark.parametrize("key", ["9", "Q", "Z", "", None, 42])
def test_unassigned_keys_are_ignored(key):
    s0 = replace(AppState(kernel_id="3"), dirty=False)
    s1, action = handle_key(s0, key)
    assert s1 is s0 and action is None  # R16.5


def test_effective_kernel_respects_normalization():
    w, zero = st.effective_kernel(AppState(kernel_id="2"))
    assert w == pytest.approx((1 / 9,) * 9) and not zero
    w, zero = st.effective_kernel(AppState(kernel_id="2", normalize=False))
    assert w == (1.0,) * 9 and not zero
    w, zero = st.effective_kernel(AppState(kernel_id="5"))
    assert w == kernels.PRESETS["5"].weights and zero


def test_custom_kernel_uses_state_weights():
    custom = (1.0, 0, 0, 0, 1, 0, 0, 0, 2)
    s = AppState(kernel_id="0", custom_kernel=custom)
    assert st.raw_kernel(s) == custom
    assert st.kernel_name(s) == "Personalizado"
    assert st.effective_kernel(s)[0] == pytest.approx(tuple(v / 4 for v in custom))


def test_title_shows_state():
    s = AppState(kernel_id="2", border=st.BORDER_REFLECT, normalize=False, display_map=kernels.MAP_OFFSET)
    t = st.title(s, "foto.png", (640, 480))
    assert t.startswith("Laboratorio de convolución | foto.png 640x480")
    assert "Kernel: Box" in t
    assert "Borde: reflect" in t
    assert "Norm: no" in t
    assert "Vista: offset" in t


def test_title_marks_zero_sum():
    t = st.title(AppState(kernel_id="7", display_map=kernels.MAP_ABS), "a.png", (7, 5))
    assert "suma 0: sin normalizar" in t  # R8.2
    assert "Vista: |v|" in t


def test_kernel_report_shows_effective_matrix_and_sum():
    report = st.kernel_report(AppState(kernel_id="2"))
    assert "Box" in report
    assert "0.1111" in report
    assert "suma: 1" in report and "original: 9" in report
    report = st.kernel_report(AppState(kernel_id="2", normalize=False))
    assert "1.0000" in report and "suma: 9" in report
    assert "suma 0: sin normalizar" in st.kernel_report(AppState(kernel_id="5"))


def test_help_lists_controls():
    for token in ("1-8", "0", "B", "N", "V", "R", "H", "Esc", "M", "X", "C", "Arriba/Abajo",
                  "Izquierda/Derecha", "RePág/AvPág", "S ", "G ", "P ", "Unsharp", "2D"):
        assert token in st.HELP_TEXT


# --- Desenfoque (R11) ------------------------------------------------------


def _clean(**kw):
    return replace(AppState(**kw), dirty=False)


def test_m_cycles_kernel_blur_unsharp():
    s, action = handle_key(_clean(), st.KEY_M)
    assert action is None and s.mode == st.MODE_BLUR and s.dirty
    s, _ = handle_key(replace(s, dirty=False), st.KEY_M)
    assert s.mode == st.MODE_UNSHARP and s.dirty
    s, _ = handle_key(replace(s, dirty=False), st.KEY_M)
    assert s.mode == st.MODE_KERNEL


def test_m_from_canny_advances_from_previous_mode():
    s, _ = handle_key(_clean(mode=st.MODE_CANNY, prev_mode=st.MODE_KERNEL), st.KEY_M)
    assert s.mode == st.MODE_BLUR


def test_x_toggles_profile():
    s, _ = handle_key(_clean(), st.KEY_X)
    assert s.profile == kernels.PROFILE_BOX and s.dirty
    s, _ = handle_key(s, st.KEY_X)
    assert s.profile == kernels.PROFILE_GAUSS


def test_up_down_step_sigma_in_blur():
    s = _clean(mode=st.MODE_BLUR, sigma=2.0)
    s, _ = handle_key(s, st.KEY_UP)
    assert s.sigma == 2.5 and s.dirty
    s, _ = handle_key(s, st.KEY_DOWN)
    s, _ = handle_key(s, st.KEY_DOWN)
    assert s.sigma == 1.5
    assert s.canny_sigma == AppState().canny_sigma  # no toca el σ de Canny


def test_sigma_limits():
    s = _clean(mode=st.MODE_BLUR, sigma=10.0)
    s1, _ = handle_key(s, st.KEY_UP)
    assert s1 is s  # en el límite no cambia ni redibuja
    s = _clean(mode=st.MODE_BLUR, sigma=0.5)
    assert handle_key(s, st.KEY_DOWN)[0] is s
    s = _clean(mode=st.MODE_BLUR, sigma=0.5)
    for _ in range(40):
        s, _ = handle_key(s, st.KEY_UP)
    assert s.sigma == 10.0
    assert kernels.gaussian_radius(s.sigma) == 30


def test_up_down_ignored_in_kernel_mode():
    s = _clean()
    assert handle_key(s, st.KEY_UP)[0] is s
    assert handle_key(s, st.KEY_DOWN)[0] is s


def test_title_blur():
    t = st.title(AppState(mode=st.MODE_BLUR, sigma=2.0), "a.png", (7, 5))
    assert "Desenfoque: gaussiano σ=2 r=6" in t
    t = st.title(AppState(mode=st.MODE_BLUR, sigma=2.0, profile=kernels.PROFILE_BOX), "a.png", (7, 5))
    assert "Desenfoque: box σ=2 r=3" in t


# --- Canny (R12.1, R12.4, R12.11) ----------------------------------------------


@pytest.mark.parametrize("prev", [st.MODE_KERNEL, st.MODE_BLUR])
def test_c_toggles_canny_and_restores_previous_mode(prev):
    s0 = _clean(mode=prev, kernel_id="5", sigma=3.0)
    s1, action = handle_key(s0, st.KEY_C)
    assert action is None and s1.mode == st.MODE_CANNY and s1.canny_on and s1.dirty
    s2, _ = handle_key(replace(s1, dirty=False), st.KEY_C)
    assert s2.mode == prev and not s2.canny_on and s2.dirty
    assert (s2.kernel_id, s2.sigma) == ("5", 3.0)


def test_canny_starts_on_original_and_keeps_stage():
    s, _ = handle_key(_clean(), st.KEY_C)
    assert s.stage == 1
    s, _ = handle_key(s, st.KEY_RIGHT)
    s, _ = handle_key(s, st.KEY_C)
    s, _ = handle_key(s, st.KEY_C)
    assert s.stage == 2


def test_left_right_walk_stages_with_limits():
    s = _clean(mode=st.MODE_CANNY, stage=1)
    assert handle_key(s, st.KEY_LEFT)[0] is s  # no baja de 1
    stages = []
    for _ in range(12):
        s, _ = handle_key(s, st.KEY_RIGHT)
        stages.append(s.stage)
    assert stages == list(range(2, 12)) + [11, 11]  # no pasa de 11
    s = _clean(mode=st.MODE_CANNY, stage=11)
    assert handle_key(s, st.KEY_RIGHT)[0] is s
    s1, _ = handle_key(s, st.KEY_LEFT)
    assert s1.stage == 10 and s1.dirty


def test_left_right_ignored_outside_canny():
    for mode in (st.MODE_KERNEL, st.MODE_BLUR):
        s = _clean(mode=mode, stage=5)
        assert handle_key(s, st.KEY_LEFT)[0] is s
        assert handle_key(s, st.KEY_RIGHT)[0] is s


def test_up_down_adjust_canny_sigma():
    s = _clean(mode=st.MODE_CANNY, canny_sigma=1.4, sigma=2.0)
    s, _ = handle_key(s, st.KEY_UP)
    assert s.canny_sigma == pytest.approx(1.9) and s.sigma == 2.0 and s.dirty
    s, _ = handle_key(s, st.KEY_DOWN)
    s, _ = handle_key(s, st.KEY_DOWN)
    s, _ = handle_key(s, st.KEY_DOWN)
    assert s.canny_sigma == 0.5


def test_default_thresholds():
    s = AppState()
    assert s.high == 0.3 and s.low_ratio == 0.4
    assert s.low == pytest.approx(0.12)
    assert "alto=0.30 bajo=0.12" in st.title(replace(s, mode=st.MODE_CANNY), "a.png", (7, 5))


def test_page_keys_adjust_high_threshold_in_canny():
    s = _clean(mode=st.MODE_CANNY, high=0.6, low_ratio=0.4)
    s, _ = handle_key(s, st.KEY_PAGE_UP)
    assert s.high == pytest.approx(0.65) and s.dirty
    assert s.low == pytest.approx(0.4 * 0.65)
    s, _ = handle_key(s, st.KEY_PAGE_DOWN)
    s, _ = handle_key(s, st.KEY_PAGE_DOWN)
    assert s.high == pytest.approx(0.55)
    for _ in range(30):
        s, _ = handle_key(s, st.KEY_PAGE_DOWN)
    assert s.high == pytest.approx(st.HIGH_MIN)
    for _ in range(30):
        s, _ = handle_key(s, st.KEY_PAGE_UP)
    assert s.high == pytest.approx(st.HIGH_MAX)


@pytest.mark.parametrize("mode", [st.MODE_KERNEL, st.MODE_BLUR])
def test_page_keys_ignored_outside_canny(mode):
    s = _clean(mode=mode)
    assert handle_key(s, st.KEY_PAGE_UP)[0] is s
    assert handle_key(s, st.KEY_PAGE_DOWN)[0] is s


def test_preset_key_leaves_canny():
    """R6.1: las teclas 1-8 entran en modo Kernel también desde Canny."""
    s, _ = handle_key(_clean(mode=st.MODE_CANNY, prev_mode=st.MODE_BLUR), "3")
    assert s.mode == st.MODE_KERNEL and s.kernel_id == "3"


@pytest.mark.parametrize("stage", range(1, 12))
def test_title_canny_stage(stage):
    t = st.title(AppState(mode=st.MODE_CANNY, stage=stage, high=0.6, low_ratio=0.4), "a.png", (7, 5))
    assert f"Canny {stage}/11: {st.STAGE_NAMES[stage - 1]}" in t
    assert "alto=0.60 bajo=0.24" in t
    assert "σ=1.4 r=5" in t
    assert "Borde: zero" in t


def test_stage_names():
    assert st.STAGE_NAMES == (
        "Original", "Gris", "Blur H", "Blur", "Gx", "Gy",
        "Magnitud", "Dirección", "NMS", "Clasificación", "Bordes",
    )


# --- 2D/separable, unsharp, captura y benchmark (R13, R14, R15) --------------


@pytest.mark.parametrize("mode", [st.MODE_BLUR, st.MODE_UNSHARP, st.MODE_CANNY])
def test_s_toggles_separable_in_blur_modes(mode):
    s0 = _clean(mode=mode)
    assert s0.separable is True  # separable por defecto
    s1, action = handle_key(s0, st.KEY_S)
    assert action is None and s1.separable is False and s1.dirty
    assert s1.mode == mode
    s2, _ = handle_key(replace(s1, dirty=False), st.KEY_S)
    assert s2.separable is True


def test_s_ignored_in_kernel_mode():
    s = _clean()
    assert handle_key(s, st.KEY_S)[0] is s


def test_separable_survives_mode_changes():
    s, _ = handle_key(_clean(mode=st.MODE_BLUR), st.KEY_S)
    s, _ = handle_key(s, st.KEY_M)  # Unsharp
    assert s.separable is False
    s, _ = handle_key(s, st.KEY_C)  # Canny
    assert s.separable is False


def test_up_down_step_sigma_in_unsharp():
    s = _clean(mode=st.MODE_UNSHARP, sigma=2.0)
    s, _ = handle_key(s, st.KEY_UP)
    assert s.sigma == 2.5 and s.dirty


def test_page_keys_adjust_alpha_in_unsharp():
    s = _clean(mode=st.MODE_UNSHARP)
    assert s.alpha == 1.0  # por defecto (R13.2)
    s, action = handle_key(s, st.KEY_PAGE_UP)
    assert action is None and s.alpha == 1.25 and s.dirty
    assert s.high == AppState().high  # no toca el umbral de Canny
    s, _ = handle_key(replace(s, dirty=False), st.KEY_PAGE_DOWN)
    s, _ = handle_key(s, st.KEY_PAGE_DOWN)
    assert s.alpha == 0.75
    for _ in range(30):
        s, _ = handle_key(s, st.KEY_PAGE_DOWN)
    assert s.alpha == st.ALPHA_MIN == 0.0
    assert handle_key(s, st.KEY_PAGE_DOWN)[0] is s
    for _ in range(30):
        s, _ = handle_key(s, st.KEY_PAGE_UP)
    assert s.alpha == st.ALPHA_MAX == 5.0
    assert handle_key(s, st.KEY_PAGE_UP)[0] is s


@pytest.mark.parametrize("key, action", [(st.KEY_G, st.ACTION_CAPTURE), (st.KEY_P, st.ACTION_BENCHMARK)])
@pytest.mark.parametrize("mode", [st.MODE_KERNEL, st.MODE_BLUR, st.MODE_UNSHARP, st.MODE_CANNY])
def test_capture_and_benchmark_actions(key, action, mode):
    s0 = _clean(mode=mode)
    s1, got = handle_key(s0, key)
    assert got == action and s1 is s0


def test_title_shows_method():
    t = st.title(AppState(mode=st.MODE_BLUR), "a.png", (7, 5))
    assert "| separable" in t
    t = st.title(AppState(mode=st.MODE_BLUR, separable=False), "a.png", (7, 5))
    assert "| 2D" in t
    t = st.title(AppState(mode=st.MODE_CANNY, stage=7, separable=False), "a.png", (7, 5))
    assert "| 2D" in t and "Canny 7/11: Magnitud" in t
    assert "separable" not in st.title(AppState(), "a.png", (7, 5))  # el modo Kernel no desenfoca


def test_title_blur_h_in_2d():
    t = st.title(AppState(mode=st.MODE_CANNY, stage=3, separable=False), "a.png", (7, 5))
    assert "Canny 3/11: Blur H (n/a en 2D)" in t
    t = st.title(AppState(mode=st.MODE_CANNY, stage=3), "a.png", (7, 5))
    assert "Canny 3/11: Blur H |" in t


def test_title_unsharp():
    t = st.title(AppState(mode=st.MODE_UNSHARP, alpha=1.5, sigma=2.0), "a.png", (7, 5))
    assert "Unsharp: α=1.50 gaussiano σ=2 r=6" in t
    assert "Borde: zero" in t and "separable" in t


def test_title_gpu_time():
    assert "GPU: 1.23 ms" in st.title(AppState(), "a.png", (7, 5), gpu_ms=1.234)
    assert "GPU" not in st.title(AppState(), "a.png", (7, 5))


def test_repeat_keys():
    assert st.REPEAT_KEYS == {
        st.KEY_UP, st.KEY_DOWN, st.KEY_LEFT, st.KEY_RIGHT, st.KEY_PAGE_UP, st.KEY_PAGE_DOWN
    }


@pytest.mark.parametrize(
    "module", ["state.py", "kernels.py", "config.py", "pipeline.py", "capture.py", "benchmark.py", "experiments.py"]
)
def test_core_modules_do_not_import_glfw(module):
    """Solo app.py importa glfw (D1)."""
    text = (ROOT / "src" / module).read_text(encoding="utf-8")
    assert "glfw" not in text
