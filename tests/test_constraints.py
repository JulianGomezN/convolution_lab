"""Restricciones obligatorias del proyecto, verificadas sobre el código (R1, R2)."""

import re
from pathlib import Path

import numpy as np
import pytest

from src.render_pass import QUAD_MODE, QUAD_VERTICES

ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = ROOT / "shaders"
SHADERS = sorted(p for p in SHADER_DIR.iterdir() if p.suffix in (".vert", ".frag", ".glsl"))


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def test_shaders_exist():
    names = {p.name for p in SHADERS}
    assert {
        "fullscreen.vert", "common.glsl", "convolve3x3.frag", "display.frag", "blur_1d.frag",
        "grayscale.frag", "sobel.frag", "nms.frag", "threshold.frag", "blur_2d.frag", "unsharp.frag",
    } <= names


@pytest.mark.parametrize("path", SHADERS, ids=lambda p: p.name)
def test_no_matrix_types(path):
    # Se analiza el texto completo, comentarios incluidos (R2.3).
    assert not re.search(r"\bd?mat[234]", path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", SHADERS, ids=lambda p: p.name)
def test_no_transform_uniforms(path):
    code = _strip_comments(path.read_text(encoding="utf-8"))
    for decl in re.findall(r"\buniform\s+\w+\s+(\w+)", code):
        assert not re.search(r"model|view|proj|camera|transform|mvp|rotat|scale|translat", decl, re.I), decl


@pytest.mark.parametrize("path", SHADERS, ids=lambda p: p.name)
def test_only_texelfetch(path):
    code = _strip_comments(path.read_text(encoding="utf-8"))
    assert not re.search(r"\btexture(2D|Lod|Offset|Proj|Grad|Gather)?\s*\(", code)


@pytest.mark.parametrize("path", [p for p in SHADERS if p.suffix != ".glsl"], ids=lambda p: p.name)
def test_version_330_core(path):
    assert path.read_text(encoding="utf-8").splitlines()[0].strip() == "#version 330 core"


@pytest.mark.parametrize(
    # display.frag y status.frag muestran una textura 1:1 (el HUD en status.frag) y no
    # leen vecinos, así que están exceptuados del fetch común de modos de borde.
    "path",
    [p for p in SHADERS if p.suffix == ".frag" and p.name not in ("display.frag", "status.frag")],
    ids=lambda p: p.name,
)
def test_neighbor_reads_go_through_common_fetch(path):
    """Los modos de borde viven solo en common.glsl (R10.3): ningún .frag de proceso llama a texelFetch."""
    code = _strip_comments(path.read_text(encoding="utf-8"))
    assert "texelFetch" not in code
    assert re.search(r'^\s*#\s*include\s+"common\.glsl"', code, re.M)


def test_border_modes_defined_once():
    common = _strip_comments((SHADER_DIR / "common.glsl").read_text(encoding="utf-8"))
    for fn in ("int imod(", "int reflect_idx(", "vec4 fetch("):
        assert fn in common
    for path in SHADERS:
        if path.name != "common.glsl":
            code = _strip_comments(path.read_text(encoding="utf-8"))
            assert not re.search(r"\b(imod|reflect_idx|fetch)\s*\([^)]*\)\s*\{", code), path.name


def test_single_vertex_shader_without_uniforms():
    verts = [p for p in SHADERS if p.suffix == ".vert"]
    assert [p.name for p in verts] == ["fullscreen.vert"]
    code = _strip_comments(verts[0].read_text(encoding="utf-8"))
    assert "uniform" not in code
    assert re.search(r"gl_Position\s*=\s*vec4\(\s*in_pos\s*,\s*0\.0\s*,\s*1\.0\s*\)", code)


def test_quad_is_four_clip_space_vertices():
    import moderngl

    assert QUAD_VERTICES.shape == (4, 2)
    assert set(np.abs(QUAD_VERTICES).ravel()) == {1.0}
    assert {tuple(v) for v in QUAD_VERTICES} == {(-1, -1), (1, -1), (-1, 1), (1, 1)}
    assert QUAD_MODE == moderngl.TRIANGLE_STRIP


def test_quad_vbo_on_gpu(quad):
    data = np.frombuffer(quad.vbo.read(), dtype="f4").reshape(-1, 2)
    np.testing.assert_array_equal(data, QUAD_VERTICES)


def test_context_is_gl33(ctx):
    assert ctx.version_code >= 330
