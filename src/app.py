"""Ventana GLFW, contexto OpenGL 3.3 core y bucle principal.

Es el único módulo que importa glfw (D1). El procesado lo hace el Processor
(entrada RGBA8 → target RGBA32F) y display.frag lo lleva a la pantalla con el
mapeo de visualización. El teclado se traduce a las teclas de state.py.
También crea el contexto sin ventana visible de --benchmark y --experiments.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import glfw
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import benchmark, capture, experiments, kernels, state as st
from src import playback, video
from src import video as _video  # alias: en _headless el parámetro `video` tapa el módulo
from src.playback import EffectiveFps, PlaybackClock, Player, Timeline
from src.config import SETTINGS_PATH, ConfigError, SettingsWatcher, load_settings
from src.framebuffers import RenderTarget
from src.pipeline import Processor
from src.render_pass import Quad, run_pass
from src.shader_loader import load_program
from src.texture_loader import ImageLoadError, create_texture, load_image

APP_NAME = st.APP_NAME
# Panel lateral único a la izquierda de ancho fijo; la imagen va pegada a su
# derecha, sin reescalar (R3.x). PANEL_MIN_HEIGHT garantiza que todas las
# secciones del panel quepan aunque la imagen sea más baja.
PANEL_WIDTH = 340
# El contenido del panel (título + Imagen + Modo + Parámetros + Borde con la
# leyenda de 4 bordes + Lupa + Controles) mide ~570 px en el modo más alto (Canny,
# con 3 filas de parámetros). 640 lo cubre con ~70 px de holgura inferior para que
# la sección CONTROLES y el marco de la imagen nunca queden recortados.
PANEL_MIN_HEIGHT = 640
# Descripciones cortas de cada modo de borde para la leyenda del panel.
EDGE_DESCRIPTIONS = {
    st.BORDER_ZERO: "fuera=0",
    st.BORDER_REPLICATE: "repite el borde",
    st.BORDER_REFLECT: "espeja",
    st.BORDER_WRAP: "da la vuelta",
}
EDGE_COLORS = {
    st.BORDER_ZERO: (238, 92, 92, 255),
    st.BORDER_REPLICATE: (101, 210, 140, 255),
    st.BORDER_REFLECT: (93, 174, 232, 255),
    st.BORDER_WRAP: (205, 137, 238, 255),
}
# Con este tiempo de espera la vigilancia del mtime responde en < 1 s (R7.3).
WAIT_TIMEOUT_S = 0.25

# Códigos de GLFW → teclas de state.py. Fila de números y teclado numérico.
KEY_MAP = {
    **{getattr(glfw, f"KEY_{d}"): str(d) for d in range(10)},
    **{getattr(glfw, f"KEY_KP_{d}"): str(d) for d in range(10)},
    glfw.KEY_B: st.KEY_B,
    glfw.KEY_N: st.KEY_N,
    glfw.KEY_V: st.KEY_V,
    glfw.KEY_R: st.KEY_R,
    glfw.KEY_H: st.KEY_H,
    glfw.KEY_M: st.KEY_M,
    glfw.KEY_X: st.KEY_X,
    glfw.KEY_C: st.KEY_C,
    glfw.KEY_S: st.KEY_S,
    glfw.KEY_G: st.KEY_G,
    glfw.KEY_P: st.KEY_P,
    glfw.KEY_L: st.KEY_L,
    glfw.KEY_LEFT_BRACKET: st.KEY_LBRACKET,
    glfw.KEY_RIGHT_BRACKET: st.KEY_RBRACKET,
    glfw.KEY_UP: st.KEY_UP,
    glfw.KEY_DOWN: st.KEY_DOWN,
    glfw.KEY_LEFT: st.KEY_LEFT,
    glfw.KEY_RIGHT: st.KEY_RIGHT,
    glfw.KEY_PAGE_UP: st.KEY_PAGE_UP,
    glfw.KEY_PAGE_DOWN: st.KEY_PAGE_DOWN,
    glfw.KEY_ESCAPE: st.KEY_ESCAPE,
    # Reproducción de vídeo (V8): posición física US; punto y coma coinciden en ES.
    glfw.KEY_SPACE: st.KEY_SPACE,
    glfw.KEY_PERIOD: st.KEY_PERIOD,
    glfw.KEY_COMMA: st.KEY_COMMA,
    glfw.KEY_HOME: st.KEY_HOME,
    glfw.KEY_O: st.KEY_O,
}
# Teclas que responden a la autorrepetición: flechas, RePág/AvPág, punto y coma.
_REPEATABLE = st.REPEAT_KEYS | st.PLAYBACK_REPEAT_KEYS
# Tope de la espera mientras se reproduce: el FPS_Efectivo se refresca ≥ 1 vez/s (R8.3).
PLAYBACK_MAX_WAIT_S = 1.0

# En un teclado español la tecla física de [ y ] no siempre coincide con GLFW_KEY_
# LEFT/RIGHT_BRACKET; el teclado numérico - y + sirve de alias para el radio de la
# lupa. Se añade solo si GLFW define esas teclas en esta versión.
for _glfw_key, _lens_key in (("KEY_KP_SUBTRACT", st.KEY_LBRACKET), ("KEY_KP_ADD", st.KEY_RBRACKET)):
    _code = getattr(glfw, _glfw_key, None)
    if _code is not None:
        KEY_MAP[_code] = _lens_key

# Campos de Settings → campos de AppState que se toman de settings.json.
SETTINGS_TO_STATE = {
    "custom_kernel": "custom_kernel",
    "sigma": "sigma",
    "canny_sigma": "canny_sigma",
    "canny_high": "high",
    "canny_low_ratio": "low_ratio",
    "alpha": "alpha",
}

GL_CAUSES = (
    "Causas probables:\n"
    "  - Driver de la GPU ausente o desactualizado (instala el del fabricante: NVIDIA, AMD o Intel).\n"
    "  - Sesión de Escritorio Remoto (RDP), que puede ofrecer solo OpenGL 1.1.\n"
    "  - Se está usando el «Adaptador de pantalla básico de Microsoft» en lugar del driver real."
)


class GLContextError(Exception):
    """GLFW no se inicia, la ventana no se crea o OpenGL < 3.3 (código de salida 3)."""


def window_size(image_size: tuple[int, int]) -> tuple[int, int]:
    """Tamaño de ventana: panel lateral + imagen a su derecha, sin reescalar (R3.x).

    El alto es el de la imagen pero nunca menor que PANEL_MIN_HEIGHT, para que el
    panel de parámetros quepa completo aunque la imagen sea baja.
    """
    width, height = image_size
    return PANEL_WIDTH + width, max(height, PANEL_MIN_HEIGHT)


def image_viewport(window_size: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Rectángulo (x, y, w, h) donde se dibuja la imagen a su resolución exacta (R3.x).

    x = PANEL_WIDTH: la imagen va pegada a la derecha del panel. Si la ventana es
    más alta que la imagen, se centra verticalmente; nunca se reescala.
    """
    _, win_h = window_size
    image_w, image_h = image_size
    return (
        PANEL_WIDTH,
        max(0, (win_h - image_h) // 2),
        image_w,
        image_h,
    )


# Paleta del panel (RGBA). Fondo sólido, encabezados tenues y valores claros.
PANEL_BG = (24, 28, 40, 242)
PANEL_HEADER = (120, 134, 158, 255)
PANEL_VALUE = (226, 233, 242, 255)
PANEL_TITLE = (247, 203, 108, 255)
PANEL_DIM = (150, 160, 176, 255)


def _load_font(size: int):
    """Fuente TrueType del tamaño pedido; si no hay ninguna, la de PIL por defecto."""
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _warn(msg: str) -> None:
    print(f"Aviso: {msg}", file=sys.stderr)


class App:
    def __init__(
        self,
        image,
        image_path: Path,
        settings_path: Path = SETTINGS_PATH,
        source=None,
        now=time.perf_counter,
    ):
        # Con `source` (vídeo o animación), `image` es su frame 0.
        self.source = source
        self.player = None  # Player; se crea con el contexto si hay source
        self.now = now  # reloj de pared inyectable (tests)
        self._last_result = None  # (tex, view) del último Processor.run, para present()
        self._panel_dirty = False  # redibujar solo panel y título, sin Processor
        self.image = image
        self.image_path = Path(image_path)
        self.height, self.width = image.shape[:2]
        self.size = (self.width, self.height)
        self.window_size = window_size(self.size)
        self.window = None
        self.ctx = None
        self.last_glfw_error = None
        self._resources = []  # objetos GL a liberar en orden inverso
        self.timer = None  # GpuTimer, se crea con el contexto
        self.last_gpu_ms = None  # tiempo de GPU del último procesado (R14.5)
        self.results_dir = capture.RESULTS_DIR
        self.settings_path = Path(settings_path)
        self.watcher = SettingsWatcher(self.settings_path)
        try:
            self.settings = load_settings(self.settings_path)
        except ConfigError as exc:
            _warn(
                f"{exc}\nSe usa la identidad como kernel personalizado y los valores por defecto "
                "de σ y umbrales hasta que se corrija (tecla R)."
            )
            self.settings = None
            self.state = st.AppState(custom_kernel=kernels.IDENTITY)
        else:
            self.state = st.AppState(
                **{field: getattr(self.settings, key) for key, field in SETTINGS_TO_STATE.items()}
            )

    # --- inicialización -------------------------------------------------

    def _on_glfw_error(self, code, description):
        if isinstance(description, bytes):
            description = description.decode("utf-8", "replace")
        self.last_glfw_error = f"({code}) {description}"
        print(f"[GLFW] error {self.last_glfw_error}", file=sys.stderr)

    def _gl_error(self, what: str) -> GLContextError:
        detail = f"\nDetalle de GLFW: {self.last_glfw_error}" if self.last_glfw_error else ""
        return GLContextError(f"{what}{detail}\n{GL_CAUSES}")

    def init(self) -> None:
        glfw.set_error_callback(self._on_glfw_error)
        if not glfw.init():
            raise self._gl_error("No se pudo inicializar GLFW.")

        # Contexto 3.3 core forward-compatible (R1.1); ventana no redimensionable.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.SCALE_TO_MONITOR, False)  # sin reescalado por DPI
        glfw.window_hint(glfw.VISIBLE, False)  # se muestra tras validar tamaños

        self.window = glfw.create_window(*self.window_size, APP_NAME, None, None)
        if not self.window:
            raise self._gl_error("No se pudo crear la ventana con un contexto OpenGL 3.3 core.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        try:
            self.ctx = moderngl.create_context(require=330)
        except Exception as exc:  # noqa: BLE001 - moderngl no unifica sus excepciones
            raise self._gl_error(f"No se pudo crear el contexto moderngl con OpenGL 3.3: {exc}") from None
        if self.ctx.version_code < 330:
            raise self._gl_error(f"Se requiere OpenGL 3.3 y el contexto ofrece {self.ctx.version_code}.")

        print_gl_info(self.ctx)
        self._check_sizes(self.ctx.info["GL_MAX_TEXTURE_SIZE"])
        self._create_gl_objects()

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        # Sin esto, Windows deja la ventana en negro tras minimizarla o taparla.
        glfw.set_window_refresh_callback(self.window, self._on_refresh)
        glfw.set_window_title(self.window, self._title())
        glfw.show_window(self.window)
        print("Pulsa H para ver los controles.")
        print(st.kernel_report(self.state))

    def _check_sizes(self, max_texture_size: int) -> None:
        w, h = self.size
        if self.source is not None:
            kind = "el vídeo"
            check_texture_limit(
                self.image_path, self.size, max_texture_size, error_cls=video.VideoOpenError, kind=kind
            )  # R3.5
        else:
            kind = "la imagen"
            check_texture_limit(self.image_path, self.size, max_texture_size)
        monitor = glfw.get_primary_monitor()
        if monitor:
            # Se compara la ventana completa (panel + imagen) con el área de trabajo (R4.7).
            _, _, mw, mh = glfw.get_monitor_workarea(monitor)
            ww, wh = self.window_size
            if ww > mw or wh > mh:
                _warn(
                    f"la ventana ({ww}x{wh}) para {kind} ({w}x{h}) es mayor que el área de trabajo "
                    f"del monitor ({mw}x{mh}); no se reescala, parte de la ventana puede quedar "
                    "fuera de la pantalla."
                )
        fw, fh = glfw.get_framebuffer_size(self.window)
        print(f"imagen {w}x{h} | ventana {fw}x{fh}")
        if (fw, fh) != self.window_size:
            _warn(
                f"el framebuffer ({fw}x{fh}) no coincide con la ventana solicitada "
                f"({self.window_size[0]}x{self.window_size[1]}); "
                f"la imagen se dibuja a su resolución {w}x{h}, sin reescalar."
            )

    def _create_gl_objects(self) -> None:
        ctx = self.ctx
        self.quad = self._keep(Quad(ctx))
        self.prog_display = self._keep(load_program(ctx, "display.frag"))
        self.prog_status = self._keep(load_program(ctx, "status.frag"))
        self.input_tex = self._keep(create_texture(ctx, self.image))
        self.player = self._create_player() if self.source is not None else None
        self.processor = self._keep(Processor(ctx, self.quad, self.input_tex))
        self.screen = RenderTarget.screen(ctx, self.window_size)
        self.status_tex = self._keep(ctx.texture(self.window_size, 4, dtype="f1"))
        self.status_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.timer = benchmark.GpuTimer(ctx)
        self._init_fonts()
        self._update_status_texture()

    def _create_player(self) -> Player:
        """Reloj, medidor de FPS y Player sobre input_tex (que ya tiene el frame 0)."""
        source = self.source
        if source.durations is not None:
            timeline = Timeline.from_durations(source.durations)
        else:
            timeline = Timeline.constant(source.fps, source.n_frames)
        now = self.now()
        clock = PlaybackClock(timeline, now)
        # Los avisos del Player ya empiezan por "Aviso:": se imprimen tal cual.
        return Player(source, self.input_tex, clock, EffectiveFps(now), video.print_warning)

    def _init_fonts(self) -> None:
        """Fuentes del panel (R16.3). Se cargan una vez; load_default si truetype falla."""
        self.font_title = _load_font(18)
        self.font_header = _load_font(13)
        self.font_value = _load_font(15)
        self.font_small = _load_font(12)

    def _keep(self, obj):
        self._resources.append(obj)
        return obj

    # --- callbacks ------------------------------------------------------

    def _on_key(self, window, key, scancode, action, mods):
        mapped = KEY_MAP.get(key)
        if mapped is None:
            return  # tecla sin asignar (R16.5)
        # Flechas y RePág/AvPág también responden a la autorrepetición.
        if action == glfw.PRESS or (action == glfw.REPEAT and mapped in _REPEATABLE):
            self.apply_key(mapped)

    def _on_cursor_pos(self, window, x, y):
        """Mueve el centro de la lupa (R21). GLFW da el origen arriba a la izquierda en
        píxeles de ventana, que coincide con el sistema fila 0 arriba de display.frag.
        No dibuja: set_cursor marca dirty solo si la lupa está activa y el bucle redibuja."""
        w, h = self.size
        offset_x, offset_y, _, _ = image_viewport(self.window_size, self.size)
        cx = min(w - 1, max(0, int(x) - offset_x))
        cy = min(h - 1, max(0, int(y) - offset_y))
        old = self.state
        new = st.set_cursor(old, cx, cy)
        if new is not old:
            self.state = new
            glfw.set_window_title(self.window, self._title())

    def apply_key(self, key: str) -> None:
        """Aplica una tecla de state.py: nuevo estado, acción, consola y título."""
        if key == kernels.CUSTOM_KEY and self.watcher.changed():
            # El archivo cambió mientras el kernel personalizado no estaba activo.
            self.reload_settings()
        old = self.state
        new, action = st.handle_key(old, key)
        self._set_state(new, old)
        if action == st.ACTION_QUIT:
            glfw.set_window_should_close(self.window, True)
        elif action == st.ACTION_HELP:
            print(st.HELP_TEXT)
        elif action == st.ACTION_RELOAD:
            self.reload_settings()
        elif action == st.ACTION_CAPTURE:
            self.capture()
        elif action == st.ACTION_BENCHMARK:
            self.benchmark()
        elif action in st.PLAYBACK_ACTIONS and self.player is not None:
            # El reloj cambia ya; el frame nuevo lo sube player.update() en el bucle.
            self.player.handle(action, self.now())
            self._panel_dirty = True

    def capture(self) -> Path | None:
        """Guarda el resultado visible como PNG W×H en results/ (R15). Si falla, avisa y sigue."""
        tex, view = self.processor.run(self.state)
        # Con vídeo, el índice del frame visible va en el nombre; el reloj no se toca (R10.2, R10.4).
        frame_index = self.player.shown_index if self.player is not None else None
        name = capture.capture_name(self.state, datetime.now(), frame_index=frame_index)
        path = capture.unique_path(self.results_dir / name)
        try:
            rgb = capture.render_display(
                self.ctx, self.quad, self.prog_display, tex, view, getattr(self, "input_tex", None)
            )
            capture.save_png(rgb, path)
        except capture.CaptureError as exc:
            print(f"Error al guardar la captura: {exc}", file=sys.stderr)
            return None
        print(f"Captura guardada: {path}")
        return path

    def benchmark(self) -> Path | None:
        """Tecla P: benchmark con el borde activo; después se redibuja (R14.3)."""
        try:
            return benchmark.benchmark_and_save(
                self.ctx,
                self.processor,
                border=self.state.border,
                csv_path=benchmark.default_csv_path(results_dir=self.results_dir),
            )
        except OSError as exc:
            print(f"Error al guardar el CSV del benchmark: {exc}", file=sys.stderr)
            return None
        finally:
            # El benchmark reutiliza los targets del Processor: hay que volver a procesar.
            self.state = replace(self.state, dirty=True)

    def _set_state(self, new: st.AppState, old: st.AppState) -> None:
        if new is old:
            return
        self.state = new
        if (new.kernel_id, new.normalize) != (old.kernel_id, old.normalize) or (
            new.kernel_id == kernels.CUSTOM_KEY and new.custom_kernel != old.custom_kernel
        ):
            print(st.kernel_report(new))  # R6.3
        glfw.set_window_title(self.window, self._title())

    def reload_settings(self) -> bool:
        """Recarga settings.json. Si no es válido, conserva la configuración anterior (R7.4)."""
        self.watcher.changed()  # sincroniza el mtime para no recargar dos veces
        try:
            settings = load_settings(self.settings_path)
        except ConfigError as exc:
            print(f"Error en la configuración: {exc}\nSe conserva la configuración anterior.", file=sys.stderr)
            return False
        print(f"Configuración recargada de {self.settings_path}.")
        # Solo se aplican las claves que cambiaron en el archivo: así no se pierden
        # los ajustes hechos con el teclado (σ, umbral) al editar otra clave.
        prev, self.settings = self.settings, settings
        old = self.state
        changes = {}
        for key, field in SETTINGS_TO_STATE.items():
            value = getattr(settings, key)
            if (prev is None or value != getattr(prev, key)) and value != getattr(old, field):
                changes[field] = value
        if changes:
            self._set_state(replace(old, **changes, dirty=True), old)
        return True

    def poll_settings(self) -> None:
        """Con el kernel personalizado activo, recarga si cambió el archivo (R7.3)."""
        if self.state.kernel_id == kernels.CUSTOM_KEY and self.watcher.changed():
            self.reload_settings()

    def _on_refresh(self, window):
        self.state = replace(self.state, dirty=True)

    # --- render ---------------------------------------------------------

    def _title(self) -> str:
        title = st.title(self.state, self.image_path.name, self.size, self.last_gpu_ms)
        if self.player is not None:
            # Con vídeo: frame y estado del reproductor tras el tiempo de GPU (R8.5).
            title += " | " + playback.title_suffix(self.player, self.now())
        return title

    def _video_rows(self) -> list[str]:
        """Filas pequeñas de la sección VÍDEO del panel (R8.1–R8.4, R8.8)."""
        player = self.player
        clock = player.clock
        k = player.shown_index
        size = f"{self.size[0]} x {self.size[1]}"
        frame = playback.frame_label(k, clock.timeline.n_frames)
        moment = playback.time_label(clock.timeline.start(k))
        status = f"{playback.status_label(clock.playing)} · {playback.loop_label(clock.loop)}"
        effective = None if not clock.playing else player.fps.value(self.now())
        fps = (
            f"fps {playback.fps_label(player.source.fps)} nominal · "
            f"{playback.fps_label(effective)} efectivo"
        )
        return [f"{size} · {frame} · {moment}", status, fps]

    def _mode_summary(self) -> str:
        """Dato principal del modo activo, para la sección MODO del panel (R16.3)."""
        s = self.state
        if s.mode == st.MODE_KERNEL:
            return st.kernel_name(s)
        if s.mode == st.MODE_BLUR:
            return f"σ={s.sigma:g} {kernels.PROFILE_NAMES[s.profile]}"
        if s.mode == st.MODE_UNSHARP:
            return f"α={s.alpha:.2f} σ={s.sigma:g}"
        return f"k{s.stage}/{st.STAGE_COUNT}: {st.stage_label(s)}"

    def _param_rows(self) -> list[tuple[str, str]]:
        """Pares (etiqueta, valor) de la sección PARAMETROS según el modo (R16.3)."""
        s = self.state
        if s.mode == st.MODE_KERNEL:
            return [
                ("Normalización", st.normalize_label(s)),
                ("Vista", kernels.MAP_NAMES[s.display_map]),
            ]
        if s.mode == st.MODE_BLUR:
            return [
                ("Sigma", f"{s.sigma:g}"),
                ("Perfil", kernels.PROFILE_NAMES[s.profile]),
                ("Método", st.method_label(s)),
            ]
        if s.mode == st.MODE_UNSHARP:
            return [
                ("Alfa", f"{s.alpha:.2f}"),
                ("Sigma", f"{s.sigma:g}"),
                ("Método", st.method_label(s)),
            ]
        return [
            ("Sigma", f"{s.canny_sigma:g}"),
            ("Alto / bajo", f"{s.high:.2f} / {s.low:.2f}"),
            ("Método", st.method_label(s)),
        ]

    def _update_status_texture(self) -> None:
        """Dibuja el panel (_draw_panel) y lo sube a status_tex."""
        image, self._panel_content_height = self._draw_panel()
        self.status_tex.write(np.asarray(image, dtype=np.uint8).tobytes())

    def _draw_panel(self) -> tuple[Image.Image, int]:
        """Dibuja el panel lateral izquierdo y el marco de la imagen (R16.3, R8).

        El panel es una imagen RGBA sólida en x ∈ [0, PANEL_WIDTH); el resto queda
        transparente para que status.frag solo tape el área del panel. El marco del
        color del borde activo rodea la imagen sin invadir su interior. Devuelve la
        imagen y el alto ocupado por el contenido; no usa la GPU.
        """
        width, height = self.window_size
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        accent = EDGE_COLORS[self.state.border]

        # Fondo del panel y filo de acento en su borde derecho.
        panel_w = min(PANEL_WIDTH, width)
        draw.rectangle((0, 0, panel_w - 1, height - 1), fill=PANEL_BG)
        draw.rectangle((panel_w - 2, 0, panel_w - 1, height - 1), fill=accent)

        margin = 16
        y = 18

        def header(text: str) -> None:
            nonlocal y
            draw.text((margin, y), text.upper(), font=self.font_header, fill=PANEL_HEADER)
            y += 20

        def value(text: str, fill=PANEL_VALUE, font=None) -> None:
            nonlocal y
            draw.text((margin, y), text, font=font or self.font_value, fill=fill)
            y += 22

        def gap(px: int = 12) -> None:
            nonlocal y
            y += px

        # Título.
        draw.text((margin, y), "LABORATORIO DE", font=self.font_title, fill=PANEL_TITLE)
        y += 24
        draw.text((margin, y), "CONVOLUCIÓN", font=self.font_title, fill=PANEL_TITLE)
        y += 30

        if self.player is not None:
            # VÍDEO (sustituye a IMAGEN, R8.1–R8.4).
            header("Vídeo")
            value(self.image_path.name)
            for line in self._video_rows():
                draw.text((margin, y), line, font=self.font_small, fill=PANEL_DIM)
                y += 18
            gap()
        else:
            # IMAGEN.
            header("Imagen")
            value(self.image_path.name)
            value(f"{self.size[0]} x {self.size[1]}", fill=PANEL_DIM, font=self.font_small)
            gap()

        # MODO.
        header("Modo")
        value(st.MODE_NAMES.get(self.state.mode, self.state.mode))
        value(self._mode_summary(), fill=PANEL_DIM, font=self.font_small)
        gap()

        # PARAMETROS.
        header("Parámetros")
        for label, val in self._param_rows():
            draw.text((margin, y), label, font=self.font_small, fill=PANEL_HEADER)
            draw.text((margin + 150, y), val, font=self.font_small, fill=PANEL_VALUE)
            y += 20
        gap()

        # BORDE: modo activo y leyenda de los cuatro modos.
        header("Borde")
        value(st.BORDER_NAMES[self.state.border].upper(), fill=accent)
        for border in range(len(st.BORDER_NAMES)):
            color = EDGE_COLORS[border]
            active = border == self.state.border
            sw_top = y + 3
            draw.rectangle((margin, sw_top, margin + 11, sw_top + 11), fill=color)
            prefix = "> " if active else "  "
            name = st.BORDER_NAMES[border]
            desc = EDGE_DESCRIPTIONS[border]
            text_fill = PANEL_VALUE if active else PANEL_DIM
            draw.text((margin + 20, y), f"{prefix}{name}", font=self.font_small, fill=text_fill)
            draw.text((margin + 110, y), desc, font=self.font_small, fill=text_fill)
            y += 18
        gap()

        # LUPA.
        header("Lupa")
        value(st.lens_label(self.state), fill=PANEL_DIM, font=self.font_small)
        gap()

        # CONTROLES (resumen; H muestra la ayuda completa en consola).
        header("Controles")
        controls = [
            "1-8 kernel   B borde   M modo",
            "C canny   L lupa   G captura",
            "Esp pausa  . , frame  O bucle  H ayuda" if self.player is not None else "H ayuda completa",
        ]
        for line in controls:
            draw.text((margin, y), line, font=self.font_small, fill=PANEL_DIM)
            y += 18

        # Alto realmente ocupado por el contenido del panel; sirve para verificar
        # (en los tests) que PANEL_MIN_HEIGHT lo cubre con holgura inferior.
        content_height = y

        # Marco fino del color del borde alrededor del área de imagen (sin taparla).
        vx, vy, vw, vh = image_viewport(self.window_size, self.size)
        frame_w = 2
        draw.rectangle(
            (vx - frame_w, vy - frame_w, vx + vw + frame_w - 1, vy + vh + frame_w - 1),
            outline=accent,
            width=frame_w,
        )
        return image, content_height

    def render(self) -> bool:
        fw, fh = glfw.get_framebuffer_size(self.window)
        if fw == 0 or fh == 0:  # ventana minimizada: no se dibuja
            return False
        # Tiempo de GPU de las pasadas de procesado, sin la de pantalla (R14.5).
        (tex, view), self.last_gpu_ms = self.timer.measure(lambda: self.processor.run(self.state))
        self._last_result = (tex, view)
        self._draw_frame((fw, fh), tex, view)
        return True

    def present(self) -> bool:
        """Redibuja pantalla y panel con el último resultado, sin Processor.run (R5.9, R8.3).

        Sirve para refrescar estado de pausa, bucle o FPS_Efectivo sin volver a
        procesar. Si aún no hay resultado, hace un render() completo.
        """
        if self._last_result is None:
            return self.render()
        fw, fh = glfw.get_framebuffer_size(self.window)
        if fw == 0 or fh == 0:  # ventana minimizada: no se dibuja
            return False
        self._draw_frame((fw, fh), *self._last_result)
        return True

    def _draw_frame(self, fb_size: tuple[int, int], tex, view) -> None:
        """display.frag con (tex, view) + panel + swap + título."""
        fw, fh = fb_size
        # La imagen conserva su tamaño y va pegada a la derecha del panel; el resto
        # es un fondo gris muy oscuro.
        viewport = image_viewport((fw, fh), self.size)
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, fw, fh)
        self.ctx.clear(0.07, 0.08, 0.10, 1.0)
        # u_original = textura de entrada para la lupa (R21). No dispara la aserción
        # anti-feedback: el destino es self.screen.texture, distinto de input_tex.
        run_pass(
            self.quad,
            self.prog_display,
            self.screen,
            {"u_src": tex, "u_original": self.input_tex},
            view.uniforms(flip_y=1, origin=viewport[:2]),
            viewport=viewport,
        )
        self._update_status_texture()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        run_pass(self.quad, self.prog_status, self.screen, {"u_status": self.status_tex})
        self.ctx.disable(moderngl.BLEND)
        glfw.swap_buffers(self.window)
        glfw.set_window_title(self.window, self._title())

    def _wait_timeout(self) -> float:
        """Espera de la iteración (R7.9, R8.3, R9.1).

        Reproduciendo: hasta el próximo cambio de frame, con tope de 1 s. En pausa,
        con imagen fija o sin próximo cambio conocido: WAIT_TIMEOUT_S, que mantiene
        la vigilancia de settings.json por debajo de 1 s (R7.3).
        """
        player = self.player
        if player is not None and player.clock.playing:
            now = self.now()
            next_change = player.clock.next_change_time(now)
            if next_change is not None:
                return min(max(0.0, next_change - now), PLAYBACK_MAX_WAIT_S)
        return WAIT_TIMEOUT_S

    def _wait(self, timeout: float) -> None:
        """Procesa eventos: sin esperar si timeout ≤ 0; si no, espera como máximo timeout."""
        if timeout <= 0:
            glfw.poll_events()
        else:
            glfw.wait_events_timeout(timeout)

    def loop_once(self) -> None:
        """Una iteración del bucle principal (diagrama "Bucle principal" del diseño)."""
        # Render bajo demanda (S12): espera eventos sin gastar CPU ni GPU.
        self._wait(self._wait_timeout())
        self.poll_settings()
        changes = self.player.update(self.now()) if self.player is not None else None
        if (changes is not None and changes.frame) or self.state.dirty:
            if self.render():
                self.state = replace(self.state, dirty=False)
                self._panel_dirty = False
            elif changes is not None and changes.frame:
                # Minimizada: el frame nuevo ya está en la textura; se dibuja al volver.
                self.state = replace(self.state, dirty=True)
        elif (changes is not None and changes.status) or self._panel_dirty:
            if self.present():
                self._panel_dirty = False

    def loop(self) -> None:
        while not glfw.window_should_close(self.window):
            self.loop_once()

    # --- liberación -----------------------------------------------------

    def release(self) -> None:
        # Primero la fuente (termina ffmpeg / libera el archivo), después GL (R12.1).
        # safe_close avisa si close() lanza y no relanza (R12.7). Sus mensajes ya
        # empiezan por "Aviso:", así que se imprimen tal cual.
        video.safe_close(self.source, video.print_warning)
        self.source = None
        for obj in reversed(self._resources):
            try:
                obj.release()
            except Exception:  # noqa: BLE001 - liberar lo posible al salir
                pass
        self._resources.clear()
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()


def print_gl_info(ctx: moderngl.Context) -> None:
    """GL_VERSION, GL_RENDERER y GL_VENDOR en consola (R1.2)."""
    info = ctx.info
    print(f"GL_VERSION:  {info['GL_VERSION']}")
    print(f"GL_RENDERER: {info['GL_RENDERER']}")
    print(f"GL_VENDOR:   {info['GL_VENDOR']}")


def check_texture_limit(
    image_path,
    size: tuple[int, int],
    max_texture_size: int,
    error_cls: type[ImageLoadError] = ImageLoadError,
    kind: str = "la imagen",
) -> None:
    """Código 2 si la imagen o el vídeo supera GL_MAX_TEXTURE_SIZE (R3.3; vídeo: R3.5)."""
    w, h = size
    if w > max_texture_size or h > max_texture_size:
        raise error_cls(
            image_path,
            f"{kind} mide {w}x{h} y supera GL_MAX_TEXTURE_SIZE ({max_texture_size}) de esta GPU",
        )


def run(image_path) -> int:
    """Carga la imagen (antes de crear la ventana, R4.3), abre la ventana y ejecuta el bucle."""
    image = load_image(image_path)
    app = App(image, Path(image_path))
    try:
        app.init()
        app.loop()
    finally:
        app.release()
    return 0


def run_video(path) -> int:
    """Abre el vídeo y decodifica el frame 0 antes de crear la ventana; después el bucle.

    Los errores de apertura (R3.1–R3.4) salen como VideoOpenError (código 2) sin
    crear ventana. El `finally` externo cierra la fuente aunque fallen App(...) o
    init(), también con Ctrl+C (R12.2); repetir el cierre no hace nada (R12.6).
    """
    path = Path(path)
    source = video.open_source(path)
    try:
        frame0 = source.read(0)
        print(video.describe(source))  # R4.6, una vez antes del primer frame
        app = App(frame0, path, source=source)
        try:
            app.init()
            app.loop()
        finally:
            app.release()
    finally:
        video.safe_close(source)
    return 0


# --- modos sin ventana visible: --benchmark y --experiments (R14.7, R17.1) ------


@contextmanager
def offscreen_context():
    """Contexto OpenGL 3.3 sin ventana visible.

    Primero un contexto standalone de moderngl (WGL en Windows) y, si falla, una
    ventana GLFW oculta con contexto 3.3 core (D1). Lanza GLContextError si no hay
    ninguno. Las texturas y FBOs no dependen del tamaño de la ventana oculta.
    """
    errors = []
    ctx = window = None
    glfw_started = False
    try:
        try:
            ctx = moderngl.create_standalone_context(require=330)
        except Exception as exc:  # noqa: BLE001 - moderngl no unifica sus excepciones
            errors.append(f"contexto standalone: {exc}")
        if ctx is None:
            glfw_started = bool(glfw.init())
            if not glfw_started:
                errors.append("ventana oculta: no se pudo inicializar GLFW")
            else:
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
                glfw.window_hint(glfw.VISIBLE, False)
                window = glfw.create_window(16, 16, APP_NAME, None, None)
                if not window:
                    errors.append("ventana oculta: no se pudo crear con OpenGL 3.3 core")
                else:
                    glfw.make_context_current(window)
                    try:
                        ctx = moderngl.create_context(require=330)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"ventana oculta: {exc}")
        if ctx is not None and ctx.version_code < 330:
            errors.append(f"se requiere OpenGL 3.3 y el contexto ofrece {ctx.version_code}")
        if ctx is None or ctx.version_code < 330:
            raise GLContextError(
                "No se pudo crear un contexto OpenGL 3.3 sin ventana.\n"
                + "\n".join(f"  - {e}" for e in errors)
                + f"\n{GL_CAUSES}"
            )
        yield ctx
    finally:
        if ctx is not None:
            ctx.release()
        if window:
            glfw.destroy_window(window)
        if glfw_started:
            glfw.terminate()


def _headless(image_path, task, video: bool = False) -> int:
    """Carga la entrada, crea el contexto sin ventana y ejecuta `task(ctx, image, extra)`.

    Modo imagen: `load_image` (código 2 si falla) y `extra = None`. Con `video=True`
    se abre el vídeo, se decodifica el frame 0 y la fuente se cierra antes de crear
    el contexto (R11.3, R12.3); `extra = {"video": ruta, "frame": 0}` (R11.4, R11.5).
    Todos los errores de apertura ocurren antes de `task`: no se crea ningún CSV ni
    carpeta de resultados (R11.6).
    """
    path = Path(image_path)
    if video:
        with _video.open_source(path) as source:
            image = source.read(0)
        extra = {"video": str(path), "frame": 0}
    else:
        image = load_image(path)
        extra = None
    h, w = image.shape[:2]
    with offscreen_context() as ctx:
        print_gl_info(ctx)
        max_size = ctx.info["GL_MAX_TEXTURE_SIZE"]
        if video:
            check_texture_limit(path, (w, h), max_size, error_cls=_video.VideoOpenError, kind="el vídeo")  # R3.5
            print(f"vídeo: {path} | frame 0")
        else:
            check_texture_limit(path, (w, h), max_size)
        print(f"imagen {path.name} {w}x{h}")
        task(ctx, image, extra)
    return 0


def run_benchmark(image_path, quick: bool = False, video: bool = False) -> int:
    """--benchmark: tabla en consola y results/benchmark_<fecha>.csv; código 0 (R14.7, R11.1)."""

    def task(ctx, image, extra):
        quad = Quad(ctx)
        tex = create_texture(ctx, image)
        proc = Processor(ctx, quad, tex)
        try:
            benchmark.benchmark_and_save(
                ctx, proc, border=st.AppState().border, quick=quick, extra_columns=extra
            )
        finally:
            proc.release()
            tex.release()
            quad.release()

    return _headless(image_path, task, video=video)


def run_experiments(image_path, quick: bool = False, video: bool = False) -> int:
    """--experiments: E1–E5 y manifest.csv en results/experimentos_<fecha>/ (R17.1, R11.2)."""

    def task(ctx, image, extra):
        experiments.run_experiments(ctx, image, experiments.default_out_dir(), quick, extra_columns=extra)

    return _headless(image_path, task, video=video)
