"""Processor: ejecuta las pasadas de cada modo sobre la textura de entrada.

No conoce GLFW (D1): recibe un moderngl.Context, así lo reutilizan la app, los
tests y, más adelante, el benchmark y los experimentos.

Todas las pasadas escriben en un esquema ping-pong de dos targets RGBA32F
(R9.1, R12.2): cada pasada lee la entrada original o `pp.src`, escribe en
`pp.dst` y después `swap()` deja el resultado en `pp.src`. La entrada original
es aparte y de solo lectura.
"""

from __future__ import annotations

from dataclasses import dataclass

import moderngl

from src import kernels
from src.framebuffers import PingPong
from src.render_pass import Quad, run_pass
from src.shader_loader import load_program
from src.state import MODE_BLUR, MODE_CANNY, MODE_KERNEL, MODE_UNSHARP, AppState, effective_kernel

# Canal que muestra display.frag (u_channel).
CH_RGB, CH_R, CH_G, CH_B, CH_DIRECTION = 0, 1, 2, 3, 4

# |G| de Sobel para un escalón negro → blanco ideal vale 4: se muestra y se
# umbraliza |G|·0.25 (D3, S7).
MAG_GAIN = 0.25


@dataclass(frozen=True)
class View:
    """Cómo mostrar la textura resultante en display.frag."""

    map: int = kernels.MAP_CLIP  # u_map
    gain: float = 1.0  # u_gain
    channel: int = CH_RGB  # u_channel

    def uniforms(self, flip_y: int) -> dict:
        """Uniforms de display.frag: flip_y = 1 en pantalla, 0 en la captura (D2)."""
        return {"u_flip_y": flip_y, "u_channel": self.channel, "u_map": self.map, "u_gain": self.gain}


# Pasadas de Canny en orden (R12.2) y cuántas hacen falta para ver cada etapa (R12.5),
# con el gaussiano separable. En 2D, "blur H" y "blur V" se sustituyen por una sola
# pasada "blur 2D", que ya da el desenfoque completo en la etapa 3.
CANNY_PASSES = ("gris", "blur H", "blur V", "sobel", "nms", "umbral")
STAGE_PASSES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4, 9: 5, 10: 6, 11: 6}
# Canal, mapeo y ganancia de cada etapa (tabla de design.md).
STAGE_VIEWS = {
    1: View(),
    2: View(channel=CH_R),
    3: View(channel=CH_R),
    4: View(channel=CH_R),
    5: View(map=kernels.MAP_ABS, gain=MAG_GAIN, channel=CH_R),  # Gx
    6: View(map=kernels.MAP_ABS, gain=MAG_GAIN, channel=CH_G),  # Gy
    7: View(gain=MAG_GAIN, channel=CH_B),  # |G|
    8: View(gain=MAG_GAIN, channel=CH_DIRECTION),  # tono = θ, brillo = |G|·0.25
    9: View(gain=MAG_GAIN, channel=CH_R),  # NMS
    10: View(channel=CH_G),  # clasificación 0 / 0.5 / 1
    11: View(channel=CH_R),  # bordes
}


class Processor:
    def __init__(self, ctx: moderngl.Context, quad: Quad, source: moderngl.Texture):
        self.ctx = ctx
        self.quad = quad
        self.source = source  # entrada de solo lectura (RGBA8 o RGBA32F)
        self.size = tuple(source.size)
        self._resources = []
        self.prog_convolve = self._keep(load_program(ctx, "convolve3x3.frag"))
        self.prog_blur_1d = self._keep(load_program(ctx, "blur_1d.frag"))
        self.prog_blur_2d = self._keep(load_program(ctx, "blur_2d.frag"))
        self.prog_unsharp = self._keep(load_program(ctx, "unsharp.frag"))
        self.prog_gray = self._keep(load_program(ctx, "grayscale.frag"))
        self.prog_sobel = self._keep(load_program(ctx, "sobel.frag"))
        self.prog_nms = self._keep(load_program(ctx, "nms.frag"))
        self.prog_threshold = self._keep(load_program(ctx, "threshold.frag"))
        self.pp = self._keep(PingPong.create(ctx, self.size, dtype="f4"))
        self.last_passes: list[str] = []  # nombres de las pasadas del último run()

    def _keep(self, obj):
        self._resources.append(obj)
        return obj

    def _pass(
        self, name: str, prog, src: moderngl.Texture, uniforms: dict, extra: dict | None = None
    ) -> moderngl.Texture:
        """Una pasada src (+ texturas extra) → pp.dst; tras el swap, el resultado es pp.src."""
        run_pass(self.quad, prog, self.pp.dst, {"u_src": src, **(extra or {})}, uniforms)
        self.pp.swap()
        self.last_passes.append(name)
        return self.pp.src.texture

    def blur(
        self,
        src: moderngl.Texture,
        weights,
        border: int,
        separable: bool = True,
        stop_after_h: bool = False,
    ) -> moderngl.Texture:
        """Desenfoque con pesos 1D: separable (blur_1d.frag, H y V) o 2D directo (blur_2d.frag) (R14.1).

        Son las únicas pasadas que mide el benchmark (R14.3).
        """
        r = (len(weights) - 1) // 2
        uniforms = {"u_weights": weights, "u_radius": r, "u_border": border}
        if not separable:
            return self._pass("blur 2D", self.prog_blur_2d, src, uniforms)
        tex = self._pass("blur H", self.prog_blur_1d, src, {**uniforms, "u_direction": (1, 0)})
        if stop_after_h:
            return tex
        return self._pass("blur V", self.prog_blur_1d, tex, {**uniforms, "u_direction": (0, 1)})

    def run(self, state: AppState) -> tuple[moderngl.Texture, View]:
        """Ejecuta el modo de `state` y devuelve (textura, vista)."""
        self.last_passes = []
        if state.mode == MODE_KERNEL:
            weights, _ = effective_kernel(state)
            tex = self._pass(
                "kernel", self.prog_convolve, self.source, {"u_kernel": weights, "u_border": state.border}
            )
            return tex, View(map=state.display_map)
        if state.mode == MODE_BLUR:
            weights = kernels.blur_weights(state.profile, state.sigma)
            return self.blur(self.source, weights, state.border, state.separable), View()
        if state.mode == MODE_UNSHARP:
            # orig + α·(orig − blur_σ(orig)) con el gaussiano (R13.1). El blur queda en
            # pp.src y unsharp escribe en pp.dst; la entrada original es aparte.
            blur = self.blur(self.source, kernels.gaussian_1d(state.sigma), state.border, state.separable)
            tex = self._pass(
                "unsharp",
                self.prog_unsharp,
                self.source,
                {"u_alpha": state.alpha, "u_border": state.border},
                extra={"u_blur": blur},
            )
            return tex, View()
        if state.mode == MODE_CANNY:
            return self._run_canny(state)
        raise ValueError(f"modo no implementado: {state.mode}")

    def _run_canny(self, state: AppState) -> tuple[moderngl.Texture, View]:
        """Ejecuta solo las pasadas necesarias hasta la etapa visible (R12.5)."""
        n = STAGE_PASSES[state.stage]
        view = STAGE_VIEWS[state.stage]
        border = {"u_border": state.border}
        if n == 0:
            return self.source, view
        tex = self._pass("gris", self.prog_gray, self.source, border)
        if n >= 2:
            # Gaussiano con el σ de Canny, siempre gaussiano (R12.7); separable o 2D según S.
            weights = kernels.gaussian_1d(state.canny_sigma)
            tex = self.blur(tex, weights, state.border, state.separable, stop_after_h=(n == 2))
        if n >= 4:
            tex = self._pass("sobel", self.prog_sobel, tex, border)
        if n >= 5:
            tex = self._pass("nms", self.prog_nms, tex, border)
        if n >= 6:
            tex = self._pass(
                "umbral",
                self.prog_threshold,
                tex,
                {**border, "u_low": state.low, "u_high": state.high, "u_mag_gain": MAG_GAIN},
            )
        return tex, view

    def release(self) -> None:
        for obj in reversed(self._resources):
            obj.release()
        self._resources.clear()
