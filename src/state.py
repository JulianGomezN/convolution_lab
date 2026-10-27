"""Estado de la aplicación y lógica de teclado, sin GLFW (D1).

`handle_key(state, key)` es pura: devuelve un estado nuevo y una acción para
`app.py`. Las teclas se representan con constantes propias; `app.py` traduce
los códigos de GLFW a estas constantes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src import kernels

# --- teclas -------------------------------------------------------------
KEY_0, KEY_1, KEY_2, KEY_3, KEY_4 = "0", "1", "2", "3", "4"
KEY_5, KEY_6, KEY_7, KEY_8 = "5", "6", "7", "8"
KEY_B = "B"
KEY_N = "N"
KEY_V = "V"
KEY_R = "R"
KEY_H = "H"
KEY_M = "M"
KEY_X = "X"
KEY_C = "C"
KEY_S = "S"
KEY_G = "G"
KEY_P = "P"
KEY_L = "L"
KEY_LBRACKET = "LBRACKET"
KEY_RBRACKET = "RBRACKET"
KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_LEFT = "LEFT"
KEY_RIGHT = "RIGHT"
KEY_PAGE_UP = "PAGE_UP"
KEY_PAGE_DOWN = "PAGE_DOWN"
KEY_ESCAPE = "ESCAPE"
# Reproducción de vídeo (V8): solo tienen efecto con --video.
KEY_SPACE = "SPACE"
KEY_PERIOD = "PERIOD"
KEY_COMMA = "COMMA"
KEY_HOME = "HOME"
KEY_O = "O"
# Teclas que también responden a la autorrepetición del teclado.
REPEAT_KEYS = frozenset(
    {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_LBRACKET, KEY_RBRACKET}
)
# Punto y coma también se repiten para recorrer frames manteniendo la tecla.
PLAYBACK_REPEAT_KEYS = frozenset({KEY_PERIOD, KEY_COMMA})

# --- acciones -----------------------------------------------------------
ACTION_RELOAD = "reload"
ACTION_HELP = "help"
ACTION_QUIT = "quit"
ACTION_CAPTURE = "capture"
ACTION_BENCHMARK = "benchmark"
# Reproducción de vídeo: las ejecuta el Player; con imagen fija se ignoran (R6.11).
ACTION_PLAY_PAUSE = "play_pause"
ACTION_NEXT_FRAME = "next_frame"
ACTION_PREV_FRAME = "prev_frame"
ACTION_FIRST_FRAME = "first_frame"
ACTION_TOGGLE_LOOP = "toggle_loop"
PLAYBACK_ACTIONS = frozenset(
    {ACTION_PLAY_PAUSE, ACTION_NEXT_FRAME, ACTION_PREV_FRAME, ACTION_FIRST_FRAME, ACTION_TOGGLE_LOOP}
)
_PLAYBACK_KEY_ACTIONS = {
    KEY_SPACE: ACTION_PLAY_PAUSE,
    KEY_PERIOD: ACTION_NEXT_FRAME,
    KEY_COMMA: ACTION_PREV_FRAME,
    KEY_HOME: ACTION_FIRST_FRAME,
    KEY_O: ACTION_TOGGLE_LOOP,
}

# --- modos --------------------------------------------------------------
MODE_KERNEL = "kernel"
MODE_BLUR = "blur"
MODE_UNSHARP = "unsharp"
MODE_CANNY = "canny"
MODE_NAMES = {MODE_KERNEL: "Kernel", MODE_BLUR: "Desenfoque", MODE_UNSHARP: "Unsharp", MODE_CANNY: "Canny"}
# Ciclo de la tecla M (R11.1).
MODE_CYCLE = (MODE_KERNEL, MODE_BLUR, MODE_UNSHARP)
# Modos que desenfocan: responden a Arriba/Abajo (σ) y a S (2D/separable) (R11.2, R14.1).
BLUR_MODES = frozenset({MODE_BLUR, MODE_UNSHARP, MODE_CANNY})
# Gaussiano 2D directo o separable (R14.1), para el título.
METHOD_SEPARABLE, METHOD_2D = "separable", "2D"

# --- unsharp (R13.2) ------------------------------------------------------
ALPHA_STEP, ALPHA_MIN, ALPHA_MAX = 0.25, 0.0, 5.0

# --- bordes (u_border, R10.1) --------------------------------------------
BORDER_ZERO, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP = 0, 1, 2, 3
BORDER_NAMES = ("zero", "replicate", "reflect", "wrap")

# --- Canny (R12.4) ------------------------------------------------------
STAGE_NAMES = (
    "Original", "Gris", "Blur H", "Blur", "Gx", "Gy",
    "Magnitud", "Dirección", "NMS", "Clasificación", "Bordes",
)
STAGE_COUNT = len(STAGE_NAMES)  # etapas 1..11
# Umbral alto sobre |G|/4 (S7): paso de RePág/AvPág y límites (R12.11).
HIGH_STEP, HIGH_MIN, HIGH_MAX = 0.05, 0.05, 1.0

# --- lupa (R21) ----------------------------------------------------------
# Radio del círculo en píxeles: límites y paso de [ y ]. LENS_EDGE es el ancho
# del borde suave (smoothstep) de display.frag, fijo.
LENS_RADIUS_MIN, LENS_RADIUS_MAX, LENS_RADIUS_STEP = 10.0, 600.0, 10.0
LENS_EDGE = 12.0

APP_NAME = "Laboratorio de convolución"

HELP_TEXT = """\
Controles:
  1-8      kernel preestablecido: 1 identidad, 2 box, 3 gaussiano, 4 sharpen,
           5 Sobel Gx, 6 Sobel Gy, 7 Laplaciano, 8 emboss
  0        kernel personalizado (custom_kernel de config/settings.json)
  B        modo de borde: zero -> replicate -> reflect -> wrap
  N        activar/desactivar la normalización (dividir por la suma)
  V        mapeo de visualización (modo Kernel): recorte -> |v| -> offset 0.5+0.5v
  M        modo: Kernel -> Desenfoque -> Unsharp
  X        perfil de desenfoque: gaussiano / box de varianza equivalente
  Arriba/Abajo    sigma del desenfoque (en Canny, sigma del gaussiano de Canny)
  S        desenfoque 2D directo / separable (Desenfoque, Unsharp y Canny)
  C        activar/desactivar el pipeline Canny (al salir vuelve al modo anterior)
  Izquierda/Derecha  etapa de Canny: Original, Gris, Blur H, Blur, Gx, Gy,
           Magnitud, Dirección, NMS, Clasificación, Bordes
  RePág/AvPág     alfa de Unsharp (+/- 0.25) o umbral alto de Canny
                  (+/- 0.05; bajo = ratio · alto)
  L        lupa: aplicar el efecto solo en el círculo bajo el cursor
  [ ]      radio de la lupa (-/+ 10 px)
  G        guardar captura del resultado en results/
  P        benchmark: gaussiano 2D directo frente a separable (CSV en results/)
  R        recargar config/settings.json
  H        mostrar esta ayuda
  Esc      salir
Vídeo (--video):
  Espacio  pausa / reproducir
  .        punto: frame siguiente (pausa)
  ,        coma: frame anterior (pausa)
  Inicio   volver al frame 0
  O        bucle activado / desactivado
Con el kernel personalizado activo, guardar settings.json lo recarga solo."""


@dataclass(frozen=True)
class AppState:
    mode: str = MODE_KERNEL
    kernel_id: str = "1"  # "1".."8" presets, "0" personalizado
    custom_kernel: tuple[float, ...] = kernels.IDENTITY
    normalize: bool = True  # activa por defecto (R8.3)
    display_map: int = kernels.MAP_CLIP
    border: int = BORDER_ZERO
    sigma: float = 2.0  # modo Desenfoque (R11.2)
    canny_sigma: float = 1.4  # gaussiano de Canny (R12.7)
    profile: str = kernels.PROFILE_GAUSS  # perfil del modo Desenfoque (R11.3)
    separable: bool = True  # False: desenfoque 2D directo con blur_2d.frag (R14.1)
    alpha: float = 1.0  # unsharp: orig + α·(orig − blur) (R13.2)
    prev_mode: str = MODE_KERNEL  # modo al que vuelve C (R12.1)
    stage: int = 1  # etapa visible de Canny, 1..11; se conserva al salir y volver
    high: float = 0.3  # umbral alto sobre |G|/4 (S7: tras el gaussiano σ=1.4 un escalón ideal da ≈ 0.51)
    low_ratio: float = 0.4  # bajo = low_ratio · alto
    lens_on: bool = False  # lupa activa (R21): el efecto solo se ve bajo el cursor
    lens_radius: float = 80.0  # radio del círculo de la lupa en píxeles
    lens_cx: float = 0.0  # centro de la lupa en píxeles, columna (fila 0 arriba)
    lens_cy: float = 0.0  # centro de la lupa en píxeles, fila (fila 0 arriba)
    dirty: bool = True

    @property
    def canny_on(self) -> bool:
        return self.mode == MODE_CANNY

    @property
    def low(self) -> float:
        return self.low_ratio * self.high


def kernel_name(state: AppState) -> str:
    if state.kernel_id == kernels.CUSTOM_KEY:
        return kernels.CUSTOM_NAME
    return kernels.PRESETS[state.kernel_id].name


def raw_kernel(state: AppState) -> tuple[float, ...]:
    """Pesos sin normalizar del kernel seleccionado."""
    if state.kernel_id == kernels.CUSTOM_KEY:
        return tuple(state.custom_kernel)
    return kernels.PRESETS[state.kernel_id].weights


def effective_kernel(state: AppState) -> tuple[tuple[float, ...], bool]:
    """(pesos que se envían al shader, suma 0 con normalización activa)."""
    raw = raw_kernel(state)
    if not state.normalize:
        return raw, False
    return kernels.normalize(raw)


def _select_kernel(state: AppState, kernel_id: str) -> AppState:
    default_map = (
        kernels.MAP_CLIP if kernel_id == kernels.CUSTOM_KEY else kernels.PRESETS[kernel_id].default_map
    )
    return replace(state, mode=MODE_KERNEL, kernel_id=kernel_id, display_map=default_map, dirty=True)


def _step_sigma(sigma: float, delta: float) -> float:
    """σ ± 0.5 dentro de [0.5, 10] (R11.2); se redondea para no acumular error de coma flotante."""
    return round(min(kernels.SIGMA_MAX, max(kernels.SIGMA_MIN, sigma + delta)), 4)


def _next_mode(state: AppState) -> str:
    """Siguiente modo del ciclo de M. Desde Canny se avanza a partir del modo previo."""
    current = state.prev_mode if state.mode == MODE_CANNY else state.mode
    return MODE_CYCLE[(MODE_CYCLE.index(current) + 1) % len(MODE_CYCLE)]


def handle_key(state: AppState, key) -> tuple[AppState, str | None]:
    """Aplica una pulsación. Las teclas sin asignar devuelven el mismo estado (R16.5)."""
    if key in kernels.PRESETS or key == kernels.CUSTOM_KEY:
        return _select_kernel(state, key), None
    if key == KEY_B:
        return replace(state, border=(state.border + 1) % len(BORDER_NAMES), dirty=True), None
    if key == KEY_N:
        return replace(state, normalize=not state.normalize, dirty=True), None
    if key == KEY_V:
        return replace(state, display_map=(state.display_map + 1) % len(kernels.MAP_NAMES), dirty=True), None
    if key == KEY_M:
        return replace(state, mode=_next_mode(state), dirty=True), None
    if key == KEY_X:
        profile = kernels.PROFILE_BOX if state.profile == kernels.PROFILE_GAUSS else kernels.PROFILE_GAUSS
        return replace(state, profile=profile, dirty=True), None
    if key in (KEY_UP, KEY_DOWN):
        delta = kernels.SIGMA_STEP if key == KEY_UP else -kernels.SIGMA_STEP
        if state.mode == MODE_CANNY:
            new = replace(state, canny_sigma=_step_sigma(state.canny_sigma, delta))
        elif state.mode in (MODE_BLUR, MODE_UNSHARP):
            new = replace(state, sigma=_step_sigma(state.sigma, delta))
        else:
            return state, None  # el modo Kernel no desenfoca
        return (state, None) if new == state else (replace(new, dirty=True), None)
    if key == KEY_S:
        if state.mode not in BLUR_MODES:
            return state, None  # el modo Kernel no desenfoca
        return replace(state, separable=not state.separable, dirty=True), None
    if key == KEY_C:
        if state.mode == MODE_CANNY:
            return replace(state, mode=state.prev_mode, dirty=True), None
        return replace(state, mode=MODE_CANNY, prev_mode=state.mode, dirty=True), None
    if key in (KEY_LEFT, KEY_RIGHT):
        if state.mode != MODE_CANNY:
            return state, None
        stage = min(STAGE_COUNT, max(1, state.stage + (1 if key == KEY_RIGHT else -1)))
        return (state, None) if stage == state.stage else (replace(state, stage=stage, dirty=True), None)
    if key in (KEY_PAGE_UP, KEY_PAGE_DOWN):
        sign = 1 if key == KEY_PAGE_UP else -1
        if state.mode == MODE_UNSHARP:
            alpha = round(min(ALPHA_MAX, max(ALPHA_MIN, state.alpha + sign * ALPHA_STEP)), 4)
            return (state, None) if alpha == state.alpha else (replace(state, alpha=alpha, dirty=True), None)
        if state.mode != MODE_CANNY:
            return state, None
        high = round(min(HIGH_MAX, max(HIGH_MIN, state.high + sign * HIGH_STEP)), 4)
        return (state, None) if high == state.high else (replace(state, high=high, dirty=True), None)
    if key == KEY_L:
        # La lupa funciona en cualquier modo (R21).
        return replace(state, lens_on=not state.lens_on, dirty=True), None
    if key in (KEY_LBRACKET, KEY_RBRACKET):
        # Radio ± paso dentro de [MIN, MAX], redondeado como σ para no acumular error.
        delta = LENS_RADIUS_STEP if key == KEY_RBRACKET else -LENS_RADIUS_STEP
        radius = round(min(LENS_RADIUS_MAX, max(LENS_RADIUS_MIN, state.lens_radius + delta)), 4)
        return (state, None) if radius == state.lens_radius else (replace(state, lens_radius=radius, dirty=True), None)
    if key == KEY_G:
        return state, ACTION_CAPTURE
    if key == KEY_P:
        return state, ACTION_BENCHMARK
    if key == KEY_R:
        return state, ACTION_RELOAD
    if key == KEY_H:
        return state, ACTION_HELP
    if key == KEY_ESCAPE:
        return state, ACTION_QUIT
    if key in _PLAYBACK_KEY_ACTIONS:
        # Mismo estado, sin dirty: la reproducción la gestiona el Player (R6.11).
        return state, _PLAYBACK_KEY_ACTIONS[key]
    return state, None


def set_cursor(state: AppState, x: float, y: float) -> AppState:
    """Mueve el centro de la lupa a (x, y) en píxeles (R21).

    Función pura: si la lupa está activa devuelve un estado nuevo con el centro
    actualizado y dirty=True (hay que redibujar); si no, devuelve el mismo estado
    sin marcar dirty, así mover el ratón con la lupa apagada no gasta GPU. (x, y)
    vienen del callback del cursor con el origen arriba a la izquierda, que es el
    mismo sistema fila 0 arriba que usa u_lens_center en display.frag.
    """
    if not state.lens_on:
        return state
    return replace(state, lens_cx=x, lens_cy=y, dirty=True)


def lens_label(state: AppState) -> str:
    """Texto de la lupa para el título: «Lupa: on r=80» o «Lupa: off»."""
    if state.lens_on:
        return f"Lupa: on r={state.lens_radius:g}"
    return "Lupa: off"


def normalize_label(state: AppState) -> str:
    if not state.normalize:
        return "no"
    _, zero_sum = effective_kernel(state)
    return "sí (suma 0: sin normalizar)" if zero_sum else "sí"


def blur_label(state: AppState) -> str:
    r = kernels.blur_radius(state.profile, state.sigma)
    return f"{kernels.PROFILE_NAMES[state.profile]} σ={state.sigma:g} r={r}"


def method_label(state: AppState) -> str:
    return METHOD_SEPARABLE if state.separable else METHOD_2D


def stage_label(state: AppState) -> str:
    name = STAGE_NAMES[state.stage - 1]
    if state.stage == 3 and not state.separable:
        # En 2D no hay pasada horizontal aparte: se ve el desenfoque completo.
        name += " (n/a en 2D)"
    return name


def title(state: AppState, image_name: str, size: tuple[int, int], gpu_ms: float | None = None) -> str:
    """Título de la ventana con el estado visible (R6.3, R9.4, R12.4, R14.5, R16.3)."""
    w, h = size
    parts = [APP_NAME, f"{image_name} {w}x{h}"]
    border = f"Borde: {BORDER_NAMES[state.border]}"
    if state.mode == MODE_KERNEL:
        parts += [
            f"{MODE_NAMES[state.mode]}: {kernel_name(state)}",
            border,
            f"Norm: {normalize_label(state)}",
            f"Vista: {kernels.MAP_NAMES[state.display_map]}",
        ]
    elif state.mode == MODE_BLUR:
        parts += [f"{MODE_NAMES[state.mode]}: {blur_label(state)}", border, method_label(state)]
    elif state.mode == MODE_UNSHARP:
        r = kernels.gaussian_radius(state.sigma)
        parts += [
            f"{MODE_NAMES[state.mode]}: α={state.alpha:.2f} gaussiano σ={state.sigma:g} r={r}",
            border,
            method_label(state),
        ]
    else:
        r = kernels.gaussian_radius(state.canny_sigma)
        parts += [
            f"Canny {state.stage}/{STAGE_COUNT}: {stage_label(state)}",
            border,
            f"σ={state.canny_sigma:g} r={r}",
            f"alto={state.high:.2f} bajo={state.low:.2f}",
            method_label(state),
        ]
    if state.lens_on:
        parts.append(lens_label(state))  # «Lupa: on r=…»; apagada no se muestra
    if gpu_ms is not None:
        parts.append(f"GPU: {gpu_ms:.2f} ms")
    return " | ".join(parts)


def kernel_report(state: AppState) -> str:
    """Matriz efectiva (ya normalizada) y su suma, para la consola (R6.3)."""
    weights, zero_sum = effective_kernel(state)
    raw_sum = sum(raw_kernel(state))
    lines = [f"Kernel {state.kernel_id} ({kernel_name(state)}), normalización {normalize_label(state)}:"]
    lines.append(kernels.format_matrix(weights))
    line = f"  suma: {sum(weights):.4g}"
    if state.normalize and not zero_sum and abs(raw_sum - 1.0) > 1e-9:
        line += f" (original: {raw_sum:.4g})"
    lines.append(line)
    return "\n".join(lines)
