"""Comprueba que las 4 dependencias de ejecución se importan y que los pins
son compatibles con Python 3.10, el mínimo exigido (R1.4)."""

import importlib
import importlib.metadata
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = "3.10"


def _pins(filename):
    """Pares (nombre, versión) de las líneas `nombre==versión` del archivo."""
    pins = []
    for ln in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith(("#", "-r")):
            name, _, version = ln.partition("==")
            pins.append((name.strip(), version.strip()))
    return pins


@pytest.mark.parametrize("module", ["moderngl", "glfw", "PIL", "numpy"])
def test_runtime_dependency_imports(module):
    assert importlib.import_module(module) is not None


def test_requirements_pinned_exactly():
    lines = [
        ln.strip()
        for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    names = {ln.split("==")[0].lower() for ln in lines}
    assert names == {"moderngl", "glfw", "pillow", "numpy"}
    assert all("==" in ln for ln in lines)


@pytest.mark.parametrize(
    "name,version", _pins("requirements.txt") + _pins("requirements-dev.txt")
)
def test_pin_supports_min_python(name, version):
    # Se leen los metadatos de la distribución instalada; por eso primero se
    # exige que la versión instalada sea la fijada.
    assert importlib.metadata.version(name) == version
    requires = importlib.metadata.metadata(name).get("Requires-Python") or ""
    assert SpecifierSet(requires).contains(MIN_PYTHON), (
        f"{name}=={version} declara Requires-Python {requires!r}, que excluye Python {MIN_PYTHON}"
    )
