"""Lógica pura de reproducción de vídeo y animaciones (spec soporte-video).

Sin GPU, sin GLFW y sin imageio: todo se prueba en CPU con tiempo inyectado.
Contiene las constantes, el saneado del FPS declarado (R7.6) y de las
duraciones de frame de las animaciones (R7.7), y la línea de tiempo
(`Timeline`, `frame_index`) que traduce t en índice de frame (R7.1, R7.8).
"""

from __future__ import annotations

import math
import numbers
from bisect import bisect_right
from dataclasses import dataclass

from src import state as st

FPS_FALLBACK = 25.0  # FPS que se usa si el declarado no es válido (R7.6)
FPS_MAX = 240.0  # FPS declarado máximo aceptado (R7.6)
DEFAULT_FRAME_MS = 100.0  # duración de un frame sin duración válida (R7.7)
FPS_WINDOW_S = 1.0  # ventana del medidor de FPS_Efectivo


def _finite_real(value) -> float | None:
    """Devuelve value como float si es un número real finito; si no, None.

    Los bool se rechazan aunque sean `numbers.Real`; los escalares de NumPy se
    aceptan.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def sanitize_fps(value) -> tuple[float, str | None]:
    """Sanea el FPS_Nominal declarado por el backend (R7.6).

    Devuelve (fps, None) si value es finito y 0 < value ≤ 240; en otro caso
    (no numérico, no finito, ≤ 0 o > 240) devuelve (25.0, aviso) con un aviso
    en español que incluye el valor recibido y "25 fps".
    """
    v = _finite_real(value)
    if v is not None and 0.0 < v <= FPS_MAX:
        return v, None
    return FPS_FALLBACK, f"Aviso: FPS declarado inválido ({value!r}); se usan {FPS_FALLBACK:g} fps."


def sanitize_durations_ms(values) -> tuple[float, ...]:
    """Convierte las duraciones declaradas (ms) a segundos (R7.7).

    Cada valor None, no numérico, no finito o ≤ 0 se sustituye por 100 ms; el
    resto se mantiene sin cambios. La tupla tiene la misma longitud que values.
    """
    out = []
    for d in values:
        v = _finite_real(d)
        out.append((v if v is not None and v > 0.0 else DEFAULT_FRAME_MS) / 1000.0)
    return tuple(out)


# ---------------------------------------------------------------------------
# Timeline: inicios de frame e índice del frame que contiene t (R7.1, R7.5, R7.8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Timeline:
    """Línea de tiempo de una fuente de frames.

    Animación: `starts` = (0, d0, d0+d1, …) con N valores, `total` = D = Σ dk,
    `frame_dt` = None. Vídeo: `starts` = None, `frame_dt` = 1/fps, `n_frames`
    y `total` (= N·dt) son None si el backend no informa N.
    """

    starts: tuple[float, ...] | None
    total: float | None
    n_frames: int | None
    frame_dt: float | None

    @classmethod
    def from_durations(cls, durations_s) -> "Timeline":
        """Construye la línea de tiempo de una animación (duraciones en s, > 0)."""
        durations = tuple(float(d) for d in durations_s)
        if not durations:
            raise ValueError("una animación necesita al menos 1 frame")
        if any(not (math.isfinite(d) and d > 0.0) for d in durations):
            raise ValueError("las duraciones deben ser finitas y > 0")
        starts = []
        acc = 0.0
        for d in durations:
            starts.append(acc)
            acc += d
        # total es la misma suma acumulada, así start(N) == total exactamente
        return cls(starts=tuple(starts), total=acc, n_frames=len(durations), frame_dt=None)

    @classmethod
    def constant(cls, fps: float, n_frames: int | None) -> "Timeline":
        """Línea de tiempo de un vídeo con duración 1/fps por frame (R7.5)."""
        fps = float(fps)
        if not (math.isfinite(fps) and fps > 0.0):
            raise ValueError("fps debe ser finito y > 0")
        if n_frames is not None and n_frames < 1:
            raise ValueError("n_frames debe ser ≥ 1 o None")
        dt = 1.0 / fps
        total = None if n_frames is None else n_frames * dt
        return cls(starts=None, total=total, n_frames=n_frames, frame_dt=dt)

    def with_frame_count(self, n: int) -> "Timeline":
        """Fija N = n: trunca (R3.6) o fija el total descubierto en el EOF."""
        if n < 1:
            raise ValueError("n debe ser ≥ 1")
        if self.starts is not None:
            n = min(n, len(self.starts))
            return Timeline(
                starts=self.starts[:n], total=self.start(n), n_frames=n, frame_dt=None
            )
        return Timeline(starts=None, total=n * self.frame_dt, n_frames=n, frame_dt=self.frame_dt)

    def start(self, k: int) -> float:
        """inicio_k; con N conocido, start(N) == total."""
        if self.starts is not None:
            if k >= len(self.starts):
                return self.total
            return self.starts[k] if k > 0 else 0.0
        return k * self.frame_dt if k > 0 else 0.0

    def index_of(self, t: float) -> int:
        """k con start(k) ≤ t < start(k+1); N−1 si t ≥ D; 0 si t < 0."""
        if not t > 0.0:  # t ≤ 0 o NaN
            return 0
        n = self.n_frames
        if self.total is not None and t >= self.total:
            return n - 1
        if self.starts is not None:
            return bisect_right(self.starts, t) - 1
        k = int(math.floor(t / self.frame_dt))
        # corrige el redondeo para que el índice sea coherente con start()
        while self.start(k + 1) <= t:
            k += 1
        while k > 0 and self.start(k) > t:
            k -= 1
        if n is not None:
            k = min(k, n - 1)
        return k


def frame_index(timeline: Timeline, t: float, loop: bool) -> int:
    """Índice del frame en el tiempo t (R7.1, R7.8).

    Con bucle y D conocido usa t mod D; si no, index_of(t) (que satura en N−1).
    """
    if loop and timeline.total is not None and t > 0.0:
        t = math.fmod(t, timeline.total)  # exacto para t ≥ 0: resultado en [0, D)
    return timeline.index_of(t)


# ---------------------------------------------------------------------------
# PlaybackClock: reloj de reproducción con tiempo inyectado (R6, R7.3, R7.4, R7.9)
# ---------------------------------------------------------------------------

_NEXT_CHANGE_MAX_ULPS = 64  # pasos máximos de ajuste fino en next_change_time


class PlaybackClock:
    """Reloj de reproducción puro: todos los métodos reciben `now` (s de pared).

    Estado: `playing`, `loop`, `t0` (tiempo de reproducción en el ancla) y
    `anchor` (tiempo de pared del ancla). Mientras reproduce,
    t = t0 + (now − anchor); en pausa, t = t0. Arranca con t = 0,
    reproduciendo y con bucle (R7.11).
    """

    def __init__(self, timeline: Timeline, now: float, playing: bool = True, loop: bool = True):
        self.timeline = timeline
        self.playing = bool(playing)
        self.loop = bool(loop)
        self.t0 = 0.0
        self.anchor = float(now)

    # -- tiempo e índice ----------------------------------------------------

    def _normalize(self, t: float) -> float:
        """Envuelve t con mod D si hay bucle; sin bucle, t ≥ D pasa a start(N−1)."""
        tl = self.timeline
        if tl.total is not None:
            if self.loop:
                if t > 0.0:
                    t = math.fmod(t, tl.total)
            elif t >= tl.total:
                t = tl.start(tl.n_frames - 1)
        return t if t > 0.0 else 0.0

    def _rebase(self, now: float) -> None:
        """t0 = position(now) normalizada y anchor = now (R7.3, R7.4, R6.5)."""
        self.t0 = self._normalize(self.position(now))
        self.anchor = float(now)

    def position(self, now: float) -> float:
        """t sin envolver: t0 + (now − anchor) si reproduce, t0 en pausa."""
        if self.playing:
            return self.t0 + (now - self.anchor)
        return self.t0

    def index(self, now: float) -> int:
        """Índice del Frame_Actual en el instante now (R7.1, R7.8)."""
        return frame_index(self.timeline, self.position(now), self.loop)

    def frame_start(self, now: float) -> float:
        """inicio_k del índice actual (panel, R8.2)."""
        return self.timeline.start(self.index(now))

    # -- controles ----------------------------------------------------------

    def update(self, now: float) -> bool:
        """Sin bucle y t ≥ D: pausa en N−1 (R6.7). True si cambió `playing`."""
        tl = self.timeline
        if (
            self.playing
            and not self.loop
            and tl.total is not None
            and self.position(now) >= tl.total
        ):
            self.playing = False
            self.t0 = tl.start(tl.n_frames - 1)
            self.anchor = float(now)
            return True
        return False

    def toggle_play(self, now: float) -> None:
        """Alterna reproducción y pausa (R6.1, R7.4); R6.14 en el último frame."""
        self._rebase(now)
        if self.playing:
            self.playing = False
            return
        n = self.timeline.n_frames
        if not self.loop and n is not None and self.index(now) == n - 1:
            self.t0 = 0.0  # R6.14: reanuda desde el frame 0
        self.playing = True

    def step(self, delta: int, now: float) -> None:
        """Pausa y avanza (delta = +1) o retrocede (delta = −1) un frame."""
        self._rebase(now)
        k = self.index(now)
        self.playing = False
        n = self.timeline.n_frames
        if delta > 0:
            k2 = k + 1
            if n is not None and k2 > n - 1:
                k2 = 0 if self.loop else n - 1  # R6.10 / R6.8
        elif delta < 0:
            k2 = k - 1
            if k2 < 0:
                k2 = n - 1 if (self.loop and n is not None) else 0  # R6.13 / R6.9
        else:
            k2 = k
        self.t0 = self.timeline.start(k2)

    def home(self, now: float) -> None:
        """t = 0 conservando `playing` y `loop` (R6.4)."""
        self._rebase(now)
        self.t0 = 0.0

    def toggle_loop(self, now: float) -> None:
        """Alterna el bucle sin cambiar el índice ni `playing` (R6.5)."""
        self._rebase(now)
        self.loop = not self.loop

    def set_frame_count(self, n: int, now: float) -> None:
        """Fija N (truncado R3.6 o EOF descubierto por ffmpeg)."""
        p = self.position(now)
        self.timeline = self.timeline.with_frame_count(n)
        self.t0 = self._normalize(p)
        self.anchor = float(now)

    # -- espera del bucle de la app -------------------------------------------

    def _changed(self, s: float, k: int) -> bool:
        """True si en s el índice ya no es k o, sin bucle, se alcanzó D."""
        if self.index(s) != k:
            return True
        tl = self.timeline
        return not self.loop and tl.total is not None and self.position(s) >= tl.total

    def next_change_time(self, now: float) -> float | None:
        """Instante de pared del próximo cambio de índice (R7.9); None si no aplica.

        None en pausa, con N = 1 o sin forma de conocer el siguiente límite.
        El resultado se ajusta en ulps para que [now, n) mantenga el índice
        actual y en n ya haya cambiado (o, sin bucle, toque pausar).
        """
        tl = self.timeline
        if not self.playing or tl.n_frames == 1:
            return None
        if tl.total is None and tl.frame_dt is None:
            return None
        k = self.index(now)
        if self._changed(now, k):
            return float(now)
        tw = self.position(now)
        if self.loop and tl.total is not None and tw > 0.0:
            tw = math.fmod(tw, tl.total)
        n_frames = tl.n_frames
        limit = tl.total if (n_frames is not None and k >= n_frames - 1) else tl.start(k + 1)
        n = max(float(now), now + (limit - tw))
        # ajuste fino por redondeo: primer instante representable con cambio
        for _ in range(_NEXT_CHANGE_MAX_ULPS):
            if self._changed(n, k):
                break
            n = math.nextafter(n, math.inf)
        for _ in range(_NEXT_CHANGE_MAX_ULPS):
            prev = math.nextafter(n, -math.inf)
            if prev <= now or not self._changed(prev, k):
                break
            n = prev
        return n


# ---------------------------------------------------------------------------
# EffectiveFps: frames subidos en el último segundo (R8.3, R8.8)
# ---------------------------------------------------------------------------


class EffectiveFps:
    """Medidor del FPS_Efectivo con tiempo inyectado.

    `record(now)` se llama una vez por frame subido a la GPU; `value(now)` es
    el nº de registros en (now − 1, now], o None si desde el último `reset`
    no ha pasado 1 s (R8.8).
    """

    def __init__(self, now: float):
        self._times: list[float] = []
        self._reset_at = float(now)

    def reset(self, now: float) -> None:
        """Reinicia la medida (al cargar y al reanudar, R8.8)."""
        self._times.clear()
        self._reset_at = float(now)

    def record(self, now: float) -> None:
        """Registra un frame subido y descarta los registros fuera de la ventana."""
        now = float(now)
        times = self._times
        # mantiene la lista ordenada aunque llegue un instante anterior
        if times and now < times[-1]:
            times.insert(bisect_right(times, now), now)
        else:
            times.append(now)
        cut = bisect_right(times, times[-1] - FPS_WINDOW_S)  # registros ≤ now − 1
        if cut:
            del times[:cut]

    def value(self, now: float) -> float | None:
        """Nº de registros en (now − 1, now] como float; None si now − reset < 1 s."""
        if now - self._reset_at < FPS_WINDOW_S:
            return None
        times = self._times
        hi = bisect_right(times, now)
        lo = bisect_right(times, now - FPS_WINDOW_S)
        return float(max(0, hi - lo))


# ---------------------------------------------------------------------------
# Etiquetas de texto del panel y del título (R8.1–R8.4, R8.8)
# ---------------------------------------------------------------------------


def frame_label(k: int, n: int | None) -> str:
    """Índice y total: "frame 12/240" o "frame 12/?" si n es None (R8.1)."""
    return f"frame {k}/{'?' if n is None else n}"


def time_label(seconds: float) -> str:
    """Instante de inicio del frame con 2 decimales: "0.48 s" (R8.2)."""
    return f"{seconds + 0.0:.2f} s"  # + 0.0 evita "-0.00"


def status_label(playing: bool) -> str:
    """Estado del reproductor: "Reproduciendo" / "En pausa" (R8.4)."""
    return "Reproduciendo" if playing else "En pausa"


def loop_label(loop: bool) -> str:
    """Estado del bucle: "bucle: sí" / "bucle: no" (R8.4)."""
    return "bucle: sí" if loop else "bucle: no"


def fps_label(value: float | None) -> str:
    """FPS con 1 decimal ("24.8") o "—" si no hay valor (R8.3, R8.8)."""
    return "—" if value is None else f"{value + 0.0:.1f}"


# ---------------------------------------------------------------------------
# Player: combina fuente, reloj y medidor de FPS (R3.6, R3.7, R5, R7.2, R8.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Changes:
    """Resultado de `Player.update` / `Player.handle`."""

    frame: bool  # el Frame_Actual de la textura cambió (hay que ejecutar el Processor)
    status: bool  # cambió playing/loop o el total de frames (basta present())


class Player:
    """Sube a la Textura_de_Entrada solo el frame que pide el reloj (VD4).

    De la textura solo usa `write(bytes_like)`. `shown_index` es el índice cuyo
    contenido está en la textura (Frame_Actual); al crear vale 0 porque la
    textura se creó con el frame 0. Con un único frame (`enabled` False) los
    controles de reproducción no hacen nada (R2.6).
    """

    def __init__(self, source, texture, clock: PlaybackClock, fps, warn):
        self.source = source
        self.texture = texture
        self.clock = clock
        self.fps = fps
        self.warn = warn
        self.shown_index = 0
        self.enabled = source.n_frames != 1
        self._warned_read = False  # aviso único de frame ilegible / EOF anticipado (R3.6)
        self._warned_size = False  # aviso único de frame de otro tamaño (R3.7)

    def update(self, now: float) -> Changes:
        """Sincroniza la textura con el índice del reloj en `now`."""
        from src.video import FrameReadError  # import tardío: src.video importa playback

        status = self.clock.update(now)
        target = self.clock.index(now)
        if target == self.shown_index:
            return Changes(False, status)  # R5.9: sin tocar la GPU
        known_total = self.clock.timeline.n_frames is not None
        try:
            frame = self.source.read(target)
            error = None
        except FrameReadError as exc:
            frame, error = None, exc
        if frame is None:
            # EOF antes de lo declarado o frame ilegible: ese índice pasa a ser el final.
            self.clock.set_frame_count(max(1, target), now)
            if (error is not None or known_total) and not self._warned_read:
                self._warned_read = True
                reason = f": {error}" if error is not None else " (fin del archivo antes de lo declarado)"
                self.warn(
                    f"Aviso: no se pudo leer el frame {target} de {self.source.path}{reason}; "
                    f"el vídeo termina en el frame {max(1, target) - 1}."
                )
            return Changes(False, True)
        w, h = self.source.size
        shape = tuple(getattr(frame, "shape", ()))
        if shape != (h, w, 4):
            # R3.7: no se sube; el reloj sigue y el siguiente frame válido se sube.
            if not self._warned_size:
                self._warned_size = True
                got = f"{shape[1]}×{shape[0]}" if len(shape) >= 2 else str(shape)
                self.warn(
                    f"Aviso: el frame {target} mide {got} y se esperaba {w}×{h}; "
                    "se descarta sin subirlo."
                )
            return Changes(False, status)
        self.texture.write(frame)  # una sola llamada (R5.1, R7.2)
        self.shown_index = target
        self.fps.record(now)
        return Changes(True, status)

    def handle(self, action: str, now: float) -> Changes:
        """Ejecuta una acción de reproducción de state.py; el frame lo sube update()."""
        if not self.enabled or action not in st.PLAYBACK_ACTIONS:
            return Changes(False, False)
        clock = self.clock
        if action == st.ACTION_PLAY_PAUSE:
            was_playing = clock.playing
            clock.toggle_play(now)
            if clock.playing and not was_playing:
                self.fps.reset(now)  # R8.8: el FPS_Efectivo se mide desde la reanudación
        elif action == st.ACTION_NEXT_FRAME:
            clock.step(+1, now)
        elif action == st.ACTION_PREV_FRAME:
            clock.step(-1, now)
        elif action == st.ACTION_FIRST_FRAME:
            clock.home(now)
        else:  # st.ACTION_TOGGLE_LOOP
            clock.toggle_loop(now)
        return Changes(False, True)


def title_suffix(player: Player, now: float) -> str:
    """Sufijo del título con vídeo (R8.5): "frame 12/240 | reproduciendo" o "… | pausa"."""
    n = player.clock.timeline.n_frames
    estado = "reproduciendo" if player.clock.playing else "pausa"
    return f"{frame_label(player.shown_index, n)} | {estado}"
