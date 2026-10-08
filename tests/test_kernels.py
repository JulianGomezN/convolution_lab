"""Presets y normalización (R6, R8.1, R8.2, R9.3)."""

import pytest

from src import kernels
from src.kernels import MAP_ABS, MAP_CLIP, PRESETS, normalize

# Tabla de design.md: tecla → (nombre, suma, vista por defecto).
TABLE = {
    "1": ("Identidad", 1, MAP_CLIP),
    "2": ("Box", 9, MAP_CLIP),
    "3": ("Gaussiano", 16, MAP_CLIP),
    "4": ("Sharpen", 1, MAP_CLIP),
    "5": ("Sobel Gx", 0, MAP_ABS),
    "6": ("Sobel Gy", 0, MAP_ABS),
    "7": ("Laplaciano", 0, MAP_ABS),
    "8": ("Emboss", 1, MAP_CLIP),
}


def test_eight_presets_on_keys_1_to_8():
    assert list(PRESETS) == [str(i) for i in range(1, 9)]
    for key, preset in PRESETS.items():
        assert preset.key == key
        assert len(preset.weights) == 9


@pytest.mark.parametrize("key", list(TABLE))
def test_preset_table(key):
    name, total, view = TABLE[key]
    preset = PRESETS[key]
    assert preset.name == name
    assert sum(preset.weights) == total
    assert preset.default_map == view


def test_presets_are_stored_unnormalized():
    assert PRESETS["2"].weights == (1.0,) * 9
    assert PRESETS["3"].weights == (1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0)


def test_sobel_and_laplacian_weights():
    """Correlación, fila superior primero, y hacia abajo (S1)."""
    assert PRESETS["5"].weights == (-1, 0, 1, -2, 0, 2, -1, 0, 1)
    assert PRESETS["6"].weights == (-1, -2, -1, 0, 0, 0, 1, 2, 1)
    assert PRESETS["7"].weights == (0, 1, 0, 1, -4, 1, 0, 1, 0)
    assert PRESETS["8"].weights == (-2, -1, 0, -1, 1, 1, 0, 1, 2)


def test_normalize_box():
    w, zero_sum = normalize(PRESETS["2"].weights)
    assert not zero_sum
    assert w == pytest.approx((1 / 9,) * 9)
    assert sum(w) == pytest.approx(1.0)


def test_normalize_gauss():
    w, zero_sum = normalize(PRESETS["3"].weights)
    assert not zero_sum
    assert w == pytest.approx(tuple(v / 16 for v in (1, 2, 1, 2, 4, 2, 1, 2, 1)))


@pytest.mark.parametrize("key", ["5", "6", "7"])
def test_normalize_zero_sum_unchanged(key):
    w, zero_sum = normalize(PRESETS[key].weights)
    assert zero_sum
    assert w == PRESETS[key].weights


def test_normalize_threshold_1e6():
    tiny = (1e-7, 0, 0, 0, 0, 0, 0, 0, 0)
    w, zero_sum = normalize(tiny)
    assert zero_sum and w == tuple(float(v) for v in tiny)
    edge = (1e-6, 0, 0, 0, 0, 0, 0, 0, 0)
    assert normalize(edge)[1] is True  # |suma| ≤ 1e-6 → sin normalizar
    small = (2e-6, 0, 0, 0, 0, 0, 0, 0, 0)
    w, zero_sum = normalize(small)
    assert not zero_sum and w[0] == pytest.approx(1.0)


def test_normalize_negative_sum():
    w, zero_sum = normalize((0, 0, 0, 0, -2, 0, 0, 0, 0))
    assert not zero_sum and w[4] == pytest.approx(1.0)


def test_normalize_sum_one_is_identity():
    assert normalize(PRESETS["4"].weights) == (PRESETS["4"].weights, False)
    assert normalize(PRESETS["8"].weights) == (PRESETS["8"].weights, False)


def test_format_matrix_has_three_rows():
    text = kernels.format_matrix(PRESETS["7"].weights)
    assert len(text.splitlines()) == 3
    assert "-4.0000" in text
