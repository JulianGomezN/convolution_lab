"""Carga y validación de config/settings.json (R7).

`load_settings` lanza ConfigError con un mensaje en español (y línea/columna si
el JSON está mal escrito). Quien la llama decide conservar la configuración
anterior (R7.4). `SettingsWatcher` detecta cambios por mtime para la recarga
automática (R7.3).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"


class ConfigError(Exception):
    """settings.json no se puede leer o no es válido."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None):
        super().__init__(message)
        self.line = line
        self.column = column


# Claves numéricas opcionales: (valor por defecto, mínimo, máximo). Coinciden
# con los límites de las teclas (σ: R11.2; umbral alto: R12.11, S7).
NUMERIC_KEYS = {
    "sigma": (2.0, 0.5, 10.0),
    "canny_sigma": (1.4, 0.5, 10.0),
    "canny_high": (0.3, 0.05, 1.0),
    "canny_low_ratio": (0.4, 0.0, 1.0),
}


@dataclass(frozen=True)
class Settings:
    custom_kernel: tuple[float, ...]  # 9 pesos, fila superior primero
    data: dict  # contenido completo; las demás claves se validan en fases posteriores
    sigma: float = NUMERIC_KEYS["sigma"][0]
    canny_sigma: float = NUMERIC_KEYS["canny_sigma"][0]
    canny_high: float = NUMERIC_KEYS["canny_high"][0]
    canny_low_ratio: float = NUMERIC_KEYS["canny_low_ratio"][0]


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_number(key: str, value) -> float:
    """Valida una clave de NUMERIC_KEYS: número finito dentro de su rango."""
    _, lo, hi = NUMERIC_KEYS[key]
    if not _is_number(value):
        raise ConfigError(f"{key} no es un número: {value!r}")
    if not math.isfinite(value):
        raise ConfigError(f"{key} no es finito: {value!r}")
    if not lo <= value <= hi:
        raise ConfigError(f"{key} debe estar entre {lo:g} y {hi:g}; vale {value!r}")
    return float(value)


def parse_kernel(value) -> tuple[float, ...]:
    """Acepta una matriz 3x3 o una lista plana de 9 números finitos."""
    if not isinstance(value, list):
        raise ConfigError("custom_kernel debe ser una matriz 3x3 de números, p. ej. [[0, 0, 0], [0, 1, 0], [0, 0, 0]]")
    if all(isinstance(row, list) for row in value) and value:
        flat = [v for row in value for v in row]
        if len(value) != 3 or any(len(row) != 3 for row in value):
            lengths = ", ".join(str(len(row)) for row in value)
            raise ConfigError(
                f"custom_kernel debe tener 9 valores en 3 filas de 3; tiene {len(flat)} "
                f"en {len(value)} filas (longitudes: {lengths})"
            )
    else:
        flat = list(value)
    if len(flat) != 9:
        raise ConfigError(f"custom_kernel debe tener 9 valores (3x3); tiene {len(flat)}")
    for i, v in enumerate(flat):
        if not _is_number(v):
            raise ConfigError(f"custom_kernel[{i // 3}][{i % 3}] no es un número: {v!r}")
        if not math.isfinite(v):
            raise ConfigError(f"custom_kernel[{i // 3}][{i % 3}] no es finito: {v!r}")
    return tuple(float(v) for v in flat)


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    path = Path(path)
    try:
        # utf-8-sig: el Bloc de notas puede guardar con BOM.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"no se pudo leer {path}: {exc.strerror or exc}") from None
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path} no está en UTF-8: {exc}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"JSON inválido en {path}, línea {exc.lineno}, columna {exc.colno}: {exc.msg}",
            line=exc.lineno,
            column=exc.colno,
        ) from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} debe contener un objeto JSON {{...}}")
    if "custom_kernel" not in data:
        raise ConfigError(f"falta la clave custom_kernel en {path}")
    numbers = {key: parse_number(key, data[key]) for key in NUMERIC_KEYS if key in data}
    return Settings(custom_kernel=parse_kernel(data["custom_kernel"]), data=data, **numbers)


class SettingsWatcher:
    """Detecta cambios de settings.json comparando mtime y tamaño."""

    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = Path(path)
        self._stamp = self._read_stamp()

    def _read_stamp(self):
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def changed(self) -> bool:
        """True si el archivo cambió desde la última llamada (o desde la creación)."""
        stamp = self._read_stamp()
        if stamp == self._stamp:
            return False
        self._stamp = stamp
        return stamp is not None
