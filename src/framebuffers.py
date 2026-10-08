"""Render targets W×H (textura + framebuffer) para las pasadas de procesamiento.

Intermedias en RGBA32F (`f4`) para conservar negativos y valores > 1 (R9.1);
la captura usa RGBA8 (`f1`). Todas miden exactamente W×H (R3.4).
"""

from __future__ import annotations

import moderngl
import numpy as np


class RenderTarget:
    """Textura de color RGBA + framebuffer, o la pantalla (`texture is None`)."""

    def __init__(self, fbo: moderngl.Framebuffer, size: tuple[int, int], texture=None, owned: bool = True):
        self.fbo = fbo
        self.size = (int(size[0]), int(size[1]))
        self.texture = texture
        self._owned = owned

    @classmethod
    def create(cls, ctx: moderngl.Context, size: tuple[int, int], dtype: str = "f4") -> "RenderTarget":
        if dtype not in ("f4", "f1"):
            raise ValueError(f"formato no compatible: {dtype}")
        tex = ctx.texture(size, 4, dtype=dtype)
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        fbo = ctx.framebuffer(color_attachments=[tex])
        return cls(fbo, size, tex)

    @classmethod
    def screen(cls, ctx: moderngl.Context, size: tuple[int, int]) -> "RenderTarget":
        """La ventana; el viewport se fija a W×H aunque el framebuffer difiera (R3.2)."""
        return cls(ctx.screen, size, None, owned=False)

    @property
    def dtype(self) -> str | None:
        return None if self.texture is None else self.texture.dtype

    def read(self) -> np.ndarray:
        """Lee el contenido como H×W×4 (fila 0 = texel y = 0, es decir, arriba; D2)."""
        w, h = self.size
        dtype = "f4" if self.dtype == "f4" else "f1"
        data = self.fbo.read(viewport=(0, 0, w, h), components=4, dtype=dtype)
        np_dtype = np.float32 if dtype == "f4" else np.uint8
        return np.frombuffer(data, dtype=np_dtype).reshape(h, w, 4).copy()

    def release(self) -> None:
        if self._owned:
            self.fbo.release()
            if self.texture is not None:
                self.texture.release()
            self._owned = False


class PingPong:
    """Dos render targets que alternan como origen y destino de una cadena de pasadas.

    Cada pasada lee `src` (o la entrada original) y escribe en `dst`; después,
    `swap()` convierte el resultado en el nuevo `src`. `src` y `dst` son siempre
    targets distintos, así que ninguna pasada lee la textura que escribe (R12.3).
    """

    def __init__(self, a: RenderTarget, b: RenderTarget):
        if a is b or a.texture is None or a.texture is b.texture:
            raise ValueError("PingPong necesita dos render targets con texturas distintas")
        self._targets = (a, b)
        self._i = 0

    @classmethod
    def create(cls, ctx: moderngl.Context, size: tuple[int, int], dtype: str = "f4") -> "PingPong":
        return cls(RenderTarget.create(ctx, size, dtype), RenderTarget.create(ctx, size, dtype))

    @property
    def src(self) -> RenderTarget:
        return self._targets[self._i]

    @property
    def dst(self) -> RenderTarget:
        return self._targets[1 - self._i]

    def swap(self) -> None:
        self._i = 1 - self._i

    def release(self) -> None:
        for target in self._targets:
            target.release()
