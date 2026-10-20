"""Quad a pantalla completa y ejecución de una pasada de render.

La única geometría es un quad de 4 vértices en (±1, ±1), ya en coordenadas de
recorte, dibujado como TRIANGLE_STRIP (R2.1). No hay matrices ni transformaciones.
"""

from __future__ import annotations

from typing import Mapping

import moderngl
import numpy as np

from src.framebuffers import RenderTarget

# (x, y) en coordenadas de recorte, orden de TRIANGLE_STRIP.
QUAD_VERTICES = np.array(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ],
    dtype="f4",
)
QUAD_MODE = moderngl.TRIANGLE_STRIP


class Quad:
    """VBO compartido + un VAO por programa (en moderngl el VAO va ligado al programa)."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.vbo = ctx.buffer(QUAD_VERTICES.tobytes())
        # id(prog) → (prog, vao); guardar prog evita que su id se reutilice.
        self._vaos: dict[int, tuple[moderngl.Program, moderngl.VertexArray]] = {}

    def vao(self, prog: moderngl.Program) -> moderngl.VertexArray:
        key = id(prog)
        if key not in self._vaos:
            self._vaos[key] = (prog, self.ctx.vertex_array(prog, [(self.vbo, "2f", "in_pos")]))
        return self._vaos[key][1]

    def render(self, prog: moderngl.Program) -> None:
        self.vao(prog).render(QUAD_MODE, vertices=len(QUAD_VERTICES))

    def release(self) -> None:
        for _, vao in self._vaos.values():
            vao.release()
        self._vaos.clear()
        self.vbo.release()


def set_uniform(prog: moderngl.Program, name: str, value) -> None:
    """Escribe un uniform; ignora los que el compilador eliminó (moderngl lanzaría KeyError)."""
    if name not in prog:
        return
    uniform = prog[name]
    if uniform.array_length > 1:
        # Arrays de floats: se rellenan con ceros hasta la longitud activa. El
        # compilador puede acortar el array si solo se usan los primeros elementos.
        flat = np.asarray(value, dtype="f4").ravel()
        n = uniform.array_length * uniform.dimension
        buf = np.zeros(n, dtype="f4")
        buf[: min(n, flat.size)] = flat[:n]
        uniform.write(buf.tobytes())
    else:
        uniform.value = value


def run_pass(
    quad: Quad,
    prog: moderngl.Program,
    target: RenderTarget,
    textures: Mapping[str, moderngl.Texture],
    uniforms: Mapping[str, object] | None = None,
    viewport: tuple[int, int, int, int] | None = None,
) -> None:
    """Dibuja el quad sobre `target`, opcionalmente dentro de un viewport."""
    if target.texture is not None and any(tex is target.texture for tex in textures.values()):
        # Leer y escribir la misma textura en una pasada es comportamiento indefinido (R12.3).
        raise AssertionError("run_pass: la textura de destino también es una entrada (feedback loop)")
    target.fbo.use()
    quad.ctx.viewport = viewport or (0, 0, *target.size)
    for unit, (name, tex) in enumerate(textures.items()):
        tex.use(location=unit)
        set_uniform(prog, name, unit)
    for name, value in (uniforms or {}).items():
        set_uniform(prog, name, value)
    quad.render(prog)
