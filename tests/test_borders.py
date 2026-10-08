"""Modos de borde de common.glsl frente a numpy.pad (R10.1, R10.3–R10.5).

Un shader de prueba lee fetch(u_src, p + u_offset) con un desplazamiento
uniforme, incluidos desplazamientos mayores que la imagen.
"""

import shutil

import numpy as np
import pytest

from src.framebuffers import RenderTarget
from src.render_pass import run_pass
from src.shader_loader import SHADER_DIR, load_program
from src.texture_loader import create_texture
from tests.reference import NP_MODES, shift

OFFSET_FRAG = """#version 330 core
#include "common.glsl"
uniform sampler2D u_src;
uniform ivec2 u_offset;
out vec4 frag_color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    frag_color = vec4(fetch(u_src, p + u_offset).rgb, 1.0);
}
"""

SIZES = [(1, 1), (2, 3), (7, 5)]  # (W, H)
OFFSETS = [
    (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1),
    (2, -3), (7, 5), (-7, -5), (13, -13), (-13, 13), (14, 10), (30, -30), (-30, 29),
]


@pytest.fixture(scope="module")
def offset_prog(ctx, tmp_path_factory):
    shader_dir = tmp_path_factory.mktemp("shaders")
    for name in ("common.glsl", "fullscreen.vert"):
        shutil.copy(SHADER_DIR / name, shader_dir / name)
    (shader_dir / "offset_test.frag").write_text(OFFSET_FRAG, encoding="utf-8")
    prog = load_program(ctx, "offset_test.frag", shader_dir=shader_dir)
    yield prog
    prog.release()


def test_np_modes_cover_four_borders():
    assert NP_MODES == {0: "constant", 1: "edge", 2: "reflect", 3: "wrap"}


@pytest.mark.parametrize("border", [0, 1, 2, 3], ids=lambda b: NP_MODES[b])
@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_offset_matches_np_pad(ctx, quad, offset_prog, size, border):
    w, h = size
    rng = np.random.default_rng(w * 100 + h)
    src = rng.uniform(-2.0, 3.0, size=(h, w, 4)).astype(np.float32)
    tex = create_texture(ctx, src)
    rt = RenderTarget.create(ctx, size)
    try:
        for ox, oy in OFFSETS:
            run_pass(quad, offset_prog, rt, {"u_src": tex}, {"u_offset": (ox, oy), "u_border": border})
            out = rt.read()
            expected = shift(src[..., :3], ox, oy, border)
            np.testing.assert_allclose(
                out[..., :3], expected, atol=1e-6, err_msg=f"offset ({ox}, {oy}), {w}x{h}, {NP_MODES[border]}"
            )
            assert np.all(out[..., 3] == 1.0)
    finally:
        rt.release()
        tex.release()


def test_exact_multiples_of_size(ctx, quad, offset_prog):
    """Desplazamientos múltiplos exactos del tamaño o del periodo (donde floor(a/n) es frágil)."""
    w, h = 7, 5
    rng = np.random.default_rng(99)
    src = rng.uniform(0.0, 1.0, size=(h, w, 4)).astype(np.float32)
    tex = create_texture(ctx, src)
    rt = RenderTarget.create(ctx, (w, h))
    offsets = [(k * w, k * h) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    offsets += [(k * (2 * w - 2), k * (2 * h - 2)) for k in (-2, -1, 1, 2)]
    try:
        for border in (2, 3):
            for ox, oy in offsets:
                run_pass(quad, offset_prog, rt, {"u_src": tex}, {"u_offset": (ox, oy), "u_border": border})
                np.testing.assert_allclose(
                    rt.read()[..., :3], shift(src[..., :3], ox, oy, border), atol=1e-6,
                    err_msg=f"offset ({ox}, {oy}), {NP_MODES[border]}",
                )
    finally:
        rt.release()
        tex.release()
