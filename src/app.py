"""Ventana GLFW, contexto OpenGL 3.3 core y bucle principal.

Es el único módulo que importa glfw (D1). El procesado lo hace el Processor
(entrada RGBA8 → target RGBA32F) y display.frag lo lleva a la pantalla con el
mapeo de visualización. El teclado se traduce a las teclas de state.py.
También crea el contexto sin ventana visible de --benchmark y --experiments.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import glfw
import moderngl

from src import benchmark, capture, experiments, kernels, state as st
from src.config import SETTINGS_PATH, ConfigError, SettingsWatcher, load_settings
from src.framebuffers import RenderTarget
from src.pipeline import Processor
from src.render_pass import Quad, run_pass
from src.shader_loader import load_program
from src.texture_loader import ImageLoadError, create_texture, load_image

APP_NAME = st.APP_NAME
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
}

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


def _warn(msg: str) -> None:
    print(f"Aviso: {msg}", file=sys.stderr)


class App:
    def __init__(self, image, image_path: Path, settings_path: Path = SETTINGS_PATH):
        self.image = image
        self.image_path = Path(image_path)
        self.height, self.width = image.shape[:2]
        self.size = (self.width, self.height)
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

        # Contexto 3.3 core forward-compatible (R1.1); ventana fija W×H (R3.1).
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.SCALE_TO_MONITOR, False)  # sin reescalado por DPI
        glfw.window_hint(glfw.VISIBLE, False)  # se muestra tras validar tamaños

        self.window = glfw.create_window(self.width, self.height, APP_NAME, None, None)
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
        check_texture_limit(self.image_path, self.size, max_texture_size)
        monitor = glfw.get_primary_monitor()
        if monitor:
            _, _, mw, mh = glfw.get_monitor_workarea(monitor)
            if w > mw or h > mh:
                _warn(
                    f"la imagen ({w}x{h}) es mayor que el área de trabajo del monitor ({mw}x{mh}); "
                    "no se reescala, parte de la ventana puede quedar fuera de la pantalla."
                )
        fw, fh = glfw.get_framebuffer_size(self.window)
        print(f"imagen {w}x{h} | framebuffer {fw}x{fh}")
        if (fw, fh) != (w, h):
            _warn(
                f"el framebuffer ({fw}x{fh}) no coincide con la imagen ({w}x{h}); "
                f"se sigue dibujando con viewport {w}x{h}, sin reescalar."
            )

    def _create_gl_objects(self) -> None:
        ctx = self.ctx
        self.quad = self._keep(Quad(ctx))
        self.prog_display = self._keep(load_program(ctx, "display.frag"))
        self.input_tex = self._keep(create_texture(ctx, self.image))
        self.processor = self._keep(Processor(ctx, self.quad, self.input_tex))
        self.screen = RenderTarget.screen(ctx, self.size)
        self.timer = benchmark.GpuTimer(ctx)

    def _keep(self, obj):
        self._resources.append(obj)
        return obj

    # --- callbacks ------------------------------------------------------

    def _on_key(self, window, key, scancode, action, mods):
        mapped = KEY_MAP.get(key)
        if mapped is None:
            return  # tecla sin asignar (R16.5)
        # Flechas y RePág/AvPág también responden a la autorrepetición.
        if action == glfw.PRESS or (action == glfw.REPEAT and mapped in st.REPEAT_KEYS):
            self.apply_key(mapped)

    def _on_cursor_pos(self, window, x, y):
        """Mueve el centro de la lupa (R21). GLFW da el origen arriba a la izquierda en
        píxeles de ventana, que coincide con el sistema fila 0 arriba de display.frag.
        No dibuja: set_cursor marca dirty solo si la lupa está activa y el bucle redibuja."""
        w, h = self.size
        cx = min(w - 1, max(0, int(x)))
        cy = min(h - 1, max(0, int(y)))
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

    def capture(self) -> Path | None:
        """Guarda el resultado visible como PNG W×H en results/ (R15). Si falla, avisa y sigue."""
        tex, view = self.processor.run(self.state)
        path = capture.unique_path(self.results_dir / capture.capture_name(self.state, datetime.now()))
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
        return st.title(self.state, self.image_path.name, self.size, self.last_gpu_ms)

    def render(self) -> bool:
        fw, fh = glfw.get_framebuffer_size(self.window)
        if fw == 0 or fh == 0:  # ventana minimizada: no se dibuja
            return False
        # Tiempo de GPU de las pasadas de procesado, sin la de pantalla (R14.5).
        (tex, view), self.last_gpu_ms = self.timer.measure(lambda: self.processor.run(self.state))
        # Se limpia todo el framebuffer por si es mayor que W×H; luego viewport W×H.
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, fw, fh)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        # u_original = textura de entrada para la lupa (R21). No dispara la aserción
        # anti-feedback: el destino es self.screen.texture, distinto de input_tex.
        run_pass(
            self.quad,
            self.prog_display,
            self.screen,
            {"u_src": tex, "u_original": self.input_tex},
            view.uniforms(flip_y=1),
        )
        glfw.swap_buffers(self.window)
        glfw.set_window_title(self.window, self._title())
        return True

    def loop(self) -> None:
        while not glfw.window_should_close(self.window):
            # Render bajo demanda (S12): espera eventos sin gastar CPU ni GPU.
            glfw.wait_events_timeout(WAIT_TIMEOUT_S)
            self.poll_settings()
            if self.state.dirty and self.render():
                self.state = replace(self.state, dirty=False)

    # --- liberación -----------------------------------------------------

    def release(self) -> None:
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


def check_texture_limit(image_path, size: tuple[int, int], max_texture_size: int) -> None:
    """Código 2 si la imagen supera GL_MAX_TEXTURE_SIZE (R3.3)."""
    w, h = size
    if w > max_texture_size or h > max_texture_size:
        raise ImageLoadError(
            image_path,
            f"la imagen mide {w}x{h} y supera GL_MAX_TEXTURE_SIZE ({max_texture_size}) de esta GPU",
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


def _headless(image_path, task) -> int:
    """Carga la imagen (código 2 si falla), crea el contexto sin ventana y ejecuta `task(ctx, image)`."""
    image = load_image(image_path)
    h, w = image.shape[:2]
    with offscreen_context() as ctx:
        print_gl_info(ctx)
        check_texture_limit(image_path, (w, h), ctx.info["GL_MAX_TEXTURE_SIZE"])
        print(f"imagen {Path(image_path).name} {w}x{h}")
        task(ctx, image)
    return 0


def run_benchmark(image_path, quick: bool = False) -> int:
    """--benchmark: tabla en consola y results/benchmark_<fecha>.csv; código 0 (R14.7)."""

    def task(ctx, image):
        quad = Quad(ctx)
        tex = create_texture(ctx, image)
        proc = Processor(ctx, quad, tex)
        try:
            benchmark.benchmark_and_save(ctx, proc, border=st.AppState().border, quick=quick)
        finally:
            proc.release()
            tex.release()
            quad.release()

    return _headless(image_path, task)


def run_experiments(image_path, quick: bool = False) -> int:
    """--experiments: E1–E5 y manifest.csv en results/experimentos_<fecha>/ (R17.1)."""
    return _headless(
        image_path, lambda ctx, image: experiments.run_experiments(ctx, image, experiments.default_out_dir(), quick)
    )
