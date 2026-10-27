"""Tests de src/playback.py (lógica pura, CPU)."""

import math

import pytest

from src import playback as pb


# --- sanitize_fps (R7.6) ---

@pytest.mark.parametrize("value", [1e-3, 1, 24.0, 29.97, 240.0])
def test_sanitize_fps_valid(value):
    assert pb.sanitize_fps(value) == (float(value), None)


@pytest.mark.parametrize("value", [0, -5.0, 240.01, math.inf, -math.inf, math.nan, None, "30", True])
def test_sanitize_fps_invalid(value):
    fps, warning = pb.sanitize_fps(value)
    assert fps == 25.0
    assert warning is not None
    assert repr(value) in warning
    assert "25 fps" in warning


# --- sanitize_durations_ms (R7.7) ---

def test_sanitize_durations_ms():
    values = [40, 0, -10, None, math.nan, math.inf, "50", 100.5]
    assert pb.sanitize_durations_ms(values) == (0.04, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1005)


def test_sanitize_durations_ms_empty():
    assert pb.sanitize_durations_ms([]) == ()


# --- Timeline y frame_index (R7.1, R7.5, R7.8) ---

def test_timeline_from_durations_intervals():
    tl = pb.Timeline.from_durations([0.1, 0.2, 0.3])
    assert tl.starts == (0.0, 0.1, pytest.approx(0.3))
    assert tl.n_frames == 3 and tl.frame_dt is None
    assert tl.total == pytest.approx(0.6)
    assert tl.start(3) == tl.total
    assert [tl.index_of(t) for t in (0.0, 0.0999, 0.1, 0.29, tl.start(2), 0.59)] == [0, 0, 1, 1, 2, 2]
    assert tl.index_of(0.6) == 2 and tl.index_of(100.0) == 2  # t ≥ D → N−1
    assert tl.index_of(-1.0) == 0


def test_timeline_from_durations_empty_raises():
    with pytest.raises(ValueError):
        pb.Timeline.from_durations([])


def test_timeline_constant_known_and_unknown():
    tl = pb.Timeline.constant(25.0, 10)
    assert tl.frame_dt == 0.04 and tl.total == pytest.approx(0.4) and tl.starts is None
    assert tl.index_of(0.0) == 0
    assert tl.index_of(tl.start(3)) == 3
    assert tl.index_of(0.12 - 1e-9) == 2
    assert tl.index_of(1.0) == 9
    unknown = pb.Timeline.constant(30.0, None)
    assert unknown.total is None and unknown.n_frames is None
    assert unknown.index_of(1000.0) == 30000


def test_timeline_constant_index_consistent_with_start():
    # 1/fps no exacto en binario: el índice debe coincidir con start() siempre
    tl = pb.Timeline.constant(29.97, None)
    for k in range(0, 2000, 7):
        t = tl.start(k)
        assert tl.index_of(t) == k
        assert tl.start(tl.index_of(t)) <= t < tl.start(tl.index_of(t) + 1)


def test_timeline_with_frame_count():
    anim = pb.Timeline.from_durations([0.1, 0.2, 0.3]).with_frame_count(2)
    assert anim.n_frames == 2 and anim.starts == (0.0, 0.1)
    assert anim.total == pytest.approx(0.3)
    assert anim.index_of(0.5) == 1
    vid = pb.Timeline.constant(10.0, None).with_frame_count(5)
    assert vid.n_frames == 5 and vid.total == pytest.approx(0.5)
    assert vid.index_of(2.0) == 4


def test_frame_index_loop_and_no_loop():
    tl = pb.Timeline.from_durations([0.1, 0.2, 0.3])
    assert pb.frame_index(tl, 0.65, loop=True) == tl.index_of(math.fmod(0.65, tl.total))
    assert pb.frame_index(tl, 0.65, loop=True) == 0
    assert pb.frame_index(tl, 0.65, loop=False) == 2
    assert pb.frame_index(tl, -0.1, loop=True) == 0
    unknown = pb.Timeline.constant(10.0, None)
    assert pb.frame_index(unknown, 5.05, loop=True) == 50  # D desconocido: sin envolver


# Feature: soporte-video, Property 12: Saneado del FPS nominal
# **Validates: Requirements 7.6**

from hypothesis import given, strategies as st  # noqa: E402

_fps_values = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.floats(min_value=-1.0, max_value=300.0),  # concentra casos cerca de 0 y 240
    st.integers(min_value=-1000, max_value=1000),
    st.sampled_from([0.0, -0.0, 240.0, math.nextafter(240.0, math.inf), 5e-324]),
    st.none(),
    st.booleans(),
    st.text(max_size=5),
    st.lists(st.integers(), max_size=2),
)


@given(_fps_values)
def test_property_sanitize_fps(value):
    fps, warning = pb.sanitize_fps(value)
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if is_number and math.isfinite(value) and 0 < value <= 240:
        assert fps == float(value)
        assert warning is None
    else:
        assert fps == 25.0
        assert warning is not None
        assert repr(value) in warning
        assert "25 fps" in warning


# Feature: soporte-video, Property 1: El índice corresponde al intervalo semiabierto que contiene t
# **Validates: Requirements 7.1, 7.5**

_durations = st.lists(
    st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=30,
)
_fraction = st.floats(min_value=0.0, max_value=1.0, exclude_max=True)
_fps = st.floats(min_value=1e-3, max_value=240.0, allow_nan=False, allow_infinity=False)


@given(_durations, _fraction)
def test_property_index_in_half_open_interval_animation(durations, fraction):
    tl = pb.Timeline.from_durations(durations)
    t = fraction * tl.total
    if t >= tl.total:  # el producto puede redondear hasta D
        t = math.nextafter(tl.total, 0.0)
    k = tl.index_of(t)
    assert 0 <= k < len(durations)
    # start(0) = 0 y start(k) = suma acumulada de las duraciones 0..k−1
    acc = 0.0
    for d in durations[:k]:
        acc += d
    assert tl.start(k) == acc
    assert tl.start(k) <= t < tl.start(k) + durations[k]


@given(_fps, st.one_of(st.integers(min_value=1, max_value=100_000), st.none()), _fraction)
def test_property_index_in_half_open_interval_video(fps, n_frames, fraction):
    tl = pb.Timeline.constant(fps, n_frames)
    dt = 1.0 / fps
    # con N desconocido se explora un horizonte acotado (hasta 10⁵ frames)
    horizon = tl.total if n_frames is not None else 100_000 * dt
    t = fraction * horizon
    if t >= horizon:
        t = math.nextafter(horizon, 0.0)
    k = tl.index_of(t)
    assert k >= 0
    if n_frames is not None:
        assert k < n_frames
    assert tl.start(k) == (k * dt if k > 0 else 0.0)
    # intervalo semiabierto [k·dt, (k+1)·dt) de duración 1/fps
    assert tl.start(k) <= t < tl.start(k + 1)
    assert tl.start(k + 1) - tl.start(k) == pytest.approx(dt)


# Feature: soporte-video, Property 13: Saneado de duraciones de animación
# **Validates: Requirements 7.7**

_duration_values = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.sampled_from([math.nan, math.inf, -math.inf, 0.0, -0.0, 5e-324]),
    st.integers(min_value=-10_000, max_value=1_000_000),
    st.none(),
    st.booleans(),
    st.text(max_size=5),
)


@given(st.lists(_duration_values, max_size=30))
def test_property_sanitize_durations_ms(values):
    result = pb.sanitize_durations_ms(values)
    assert isinstance(result, tuple)
    assert len(result) == len(values)
    for d, out in zip(values, result):
        is_number = isinstance(d, (int, float)) and not isinstance(d, bool)
        if is_number and math.isfinite(d) and d > 0:
            assert out == float(d) / 1000
        else:
            assert out == 0.1


# Feature: soporte-video, Property 2: Rango, monotonía, fin y bucle del índice
# **Validates: Requirements 7.8, 6.6**

_timelines = st.one_of(
    st.builds(pb.Timeline.from_durations, _durations),
    st.builds(pb.Timeline.constant, _fps, st.integers(min_value=1, max_value=10_000)),
)
_t_factor = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)


@given(_timelines, _t_factor, _t_factor)
def test_property_frame_index_range_monotony_end_loop(tl, f1, f2):
    n, d = tl.n_frames, tl.total
    t1, t2 = sorted((f1 * d, f2 * d))  # t ≥ 0 hasta 5·D, cubre t ≥ D y varias vueltas
    for t in (t1, t2):
        for loop in (False, True):
            assert 0 <= pb.frame_index(tl, t, loop) <= n - 1
        # con bucle: igual al índice sin bucle de t mod D
        assert pb.frame_index(tl, t, True) == pb.frame_index(tl, math.fmod(t, d), False)
        # sin bucle: N−1 para todo t ≥ D
        if t >= d:
            assert pb.frame_index(tl, t, False) == n - 1
    # sin bucle: monótono no decreciente
    assert pb.frame_index(tl, t1, False) <= pb.frame_index(tl, t2, False)


# Feature: soporte-video, Property 3: La pausa congela el tiempo y reanudar continúa sin saltos
# **Validates: Requirements 6.1, 7.3, 7.4**

from hypothesis import assume  # noqa: E402

_wall = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)
_pause_len = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_offsets = st.lists(_fraction, min_size=1, max_size=10)


@given(_timelines, st.booleans(), _wall, _fraction, _pause_len, _offsets, _fraction)
def test_property_pause_freezes_and_resume_continues(tl, loop, c, f, w, offsets, fd):
    n, d = tl.n_frames, tl.total
    assume(n >= 2)
    clock = pb.PlaybackClock(tl, c, playing=True, loop=loop)
    # con bucle se pausa tras varias vueltas; sin bucle, antes del último frame
    # (la pausa automática al final sin bucle queda fuera de la propiedad)
    tp = f * (3.0 * d if loop else tl.start(n - 1))
    p = c + tp
    before = clock.position(p)
    k_pause = clock.index(p)
    if not loop:
        assume(k_pause < n - 1)
    clock.toggle_play(p)
    assert not clock.playing
    frozen = clock.position(p)
    # el valor en pausa es la posición previa, envuelta con mod D si hay bucle
    expected = math.fmod(before, d) if loop and before > 0.0 else before
    assert frozen == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert clock.index(p) == k_pause
    # en pausa, position e index son constantes en todo [p, p + w]
    for o in offsets:
        assert clock.position(p + o * w) == frozen
        assert clock.index(p + o * w) == k_pause
    assert clock.position(p + w) == frozen
    # reanudar en p + w: mismo índice y sin sumar la duración de la pausa
    r = p + w
    clock.toggle_play(r)
    assert clock.playing
    assert clock.index(r) == k_pause
    assert clock.position(r) == frozen
    delta = fd * 2.0 * d
    assert clock.position(r + delta) == pytest.approx(frozen + delta, rel=1e-9, abs=1e-6)


# Feature: soporte-video, Property 4: Paso de frame contra un modelo
# **Validates: Requirements 6.2, 6.3, 6.8, 6.9, 6.10, 6.13, 7.4**

_step_ops = st.lists(st.sampled_from(["+1", "-1", "loop"]), min_size=1, max_size=30)


def _step_model(k, delta, n, loop):
    """Modelo de referencia del paso de frame (R6.2, R6.3, R6.8–R6.10, R6.13)."""
    if delta > 0:
        return k + 1 if k + 1 <= n - 1 else (0 if loop else n - 1)
    return k - 1 if k - 1 >= 0 else (n - 1 if loop else 0)


@given(_timelines, st.booleans(), st.booleans(), _wall, _t_factor, _step_ops, _fraction)
def test_property_step_against_model(tl, playing, loop, c, f, ops, fw):
    n, d = tl.n_frames, tl.total
    clock = pb.PlaybackClock(tl, c, playing=playing, loop=loop)
    # estado inicial alcanzable: avanza el tiempo (reproduciendo) y aplica R6.7
    now = c + f * d
    clock.update(now)
    k = clock.index(now)
    assert 0 <= k <= n - 1
    for op in ops:
        now += fw * d  # el tiempo de pared avanza (solo cuenta si reproduce)
        clock.update(now)
        k = clock.index(now)
        if op == "loop":
            clock.toggle_loop(now)
            assert clock.index(now) == k  # R6.5: no cambia el índice
            continue
        delta = 1 if op == "+1" else -1
        expected = _step_model(k, delta, n, clock.loop)
        clock.step(delta, now)
        assert not clock.playing
        k = clock.index(now)
        assert k == expected
        # al reanudar se parte de inicio_k (R7.4)
        assert clock.position(now) == tl.start(k)


# Feature: soporte-video, Property 5: Inicio vuelve al frame 0 sin tocar pausa ni bucle
# **Validates: Requirements 6.4**


@given(_timelines, st.booleans(), st.booleans(), _wall, _t_factor, st.booleans())
def test_property_home_returns_to_frame_zero(tl, playing, loop, c, f, toggle):
    clock = pb.PlaybackClock(tl, c, playing=playing, loop=loop)
    # estado alcanzable: avanza el tiempo, aplica R6.7 y opcionalmente alterna pausa
    now = c + f * tl.total
    clock.update(now)
    if toggle:
        clock.toggle_play(now)
        clock.update(now)
    was_playing, was_loop = clock.playing, clock.loop
    clock.home(now)
    assert clock.index(now) == 0
    assert clock.position(now) == 0
    assert clock.playing == was_playing
    assert clock.loop == was_loop


# Feature: soporte-video, Property 6: Alternar el bucle no cambia el frame ni la pausa
# **Validates: Requirements 6.5**


@given(_timelines, st.booleans(), st.booleans(), _wall, _t_factor, st.booleans(),
       st.integers(min_value=1, max_value=6))
def test_property_toggle_loop_keeps_frame_and_pause(tl, playing, loop, c, f, toggle, reps):
    clock = pb.PlaybackClock(tl, c, playing=playing, loop=loop)
    # estado alcanzable: avanza el tiempo, aplica R6.7 y opcionalmente alterna pausa
    now = c + f * tl.total
    clock.update(now)
    if toggle:
        clock.toggle_play(now)
        clock.update(now)
    for _ in range(reps):
        k, was_playing, was_loop = clock.index(now), clock.playing, clock.loop
        clock.toggle_loop(now)
        assert clock.loop == (not was_loop)
        assert clock.index(now) == k
        assert clock.playing == was_playing


# Feature: soporte-video, Property 7: Sin bucle la reproducción se pausa en el último frame
# **Validates: Requirements 6.7, 6.6**


@given(_timelines, _wall, _t_factor, st.lists(_t_factor, min_size=1, max_size=5))
def test_property_no_loop_pauses_at_last_frame(tl, c, f, later):
    n, d = tl.n_frames, tl.total
    # sin bucle, reproduciendo desde t = 0: cualquier now con position ≥ D
    clock = pb.PlaybackClock(tl, c, playing=True, loop=False)
    now = c + d * (1.0 + f)
    while clock.position(now) < d:  # c + t puede redondear por debajo de D
        now = math.nextafter(now, math.inf)
    assert clock.update(now) is True
    assert not clock.playing
    assert clock.index(now) == n - 1
    # instantes posteriores: sigue en pausa en N−1 y update no cambia nada
    s = now
    for g in later:
        s += g * d
        assert clock.update(s) is False
        assert not clock.playing
        assert clock.index(s) == n - 1
    # con bucle: update nunca pausa y el índice envuelve con mod D (R6.6)
    looped = pb.PlaybackClock(tl, c, playing=True, loop=True)
    s = c
    for g in [f, *later]:
        s += g * d
        assert looped.update(s) is False
        assert looped.playing
        t = looped.position(s)
        expected = tl.index_of(math.fmod(t, d)) if t > 0.0 else 0
        assert looped.index(s) == expected
        assert 0 <= looped.index(s) <= n - 1


# Feature: soporte-video, Property 8: El siguiente cambio de frame se predice exactamente
# **Validates: Requirements 7.9, 2.6**


def _changed_at(clock, s, k):
    """En s el índice ya no es k o, sin bucle, toca pausar (position ≥ D)."""
    tl = clock.timeline
    return clock.index(s) != k or (not clock.loop and clock.position(s) >= tl.total)


@given(_timelines, st.booleans(), _wall, _t_factor, st.lists(_fraction, min_size=1, max_size=8))
def test_property_next_change_time_exact(tl, loop, c, f, samples):
    assume(tl.n_frames >= 2)
    clock = pb.PlaybackClock(tl, c, playing=True, loop=loop)
    now = c + f * tl.total  # reproduciendo; t hasta 5·D (varias vueltas o t ≥ D)
    k = clock.index(now)
    n = clock.next_change_time(now)
    assert n is not None
    assert n >= now
    # en [now, n) el índice se mantiene (y, sin bucle, aún no toca pausar)
    points = [now + g * (n - now) for g in samples]
    prev = math.nextafter(n, -math.inf)
    if prev > now:
        points.append(prev)
    if n > now:
        points.append(now)
    for s in points:
        if now <= s < n:
            assert not _changed_at(clock, s, k)
    # en n el índice ya cambió o, sin bucle, update pausa el reloj
    assert _changed_at(clock, n, k)
    if not loop and clock.index(n) == k:
        assert clock.update(n) is True
        assert not clock.playing


@given(_timelines, st.booleans(), _wall, _t_factor)
def test_property_next_change_time_none_when_paused(tl, loop, c, f):
    clock = pb.PlaybackClock(tl, c, playing=False, loop=loop)
    assert clock.next_change_time(c + f * tl.total) is None
    playing = pb.PlaybackClock(tl, c, playing=True, loop=loop)
    now = c + f * tl.total
    playing.toggle_play(now)  # pausa tras reproducir
    assert playing.next_change_time(now) is None


_single_frame = st.one_of(
    st.builds(lambda d: pb.Timeline.from_durations([d]),
              st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False)),
    st.builds(pb.Timeline.constant, _fps, st.just(1)),
)


@given(_single_frame, st.booleans(), st.booleans(), _wall, _t_factor)
def test_property_next_change_time_none_single_frame(tl, playing, loop, c, f):
    # R2.6: imagen fija (N = 1) → sin espera por tiempo
    clock = pb.PlaybackClock(tl, c, playing=playing, loop=loop)
    assert clock.next_change_time(c + f * tl.total) is None


# ---------------------------------------------------------------------------
# Tests unitarios deterministas del reloj (R13.3, R13.4, R6.14, R7.11, R2.6)
# Tiempo de pared controlado por el test (sin esperas reales). Duraciones
# diádicas (potencias de 2) para que la aritmética en coma flotante sea exacta.
# ---------------------------------------------------------------------------

# animación de 3 frames: inicios 0, 0.125, 0.375; D = 0.875
_ANIM = (0.125, 0.25, 0.5)
_C = 100.0  # instante de pared inicial


def _anim_clock(loop=True, playing=True):
    return pb.PlaybackClock(pb.Timeline.from_durations(_ANIM), _C, playing=playing, loop=loop)


def test_clock_estado_inicial():
    # R7.11: t = 0, frame 0, reproduciendo y con bucle
    clock = pb.PlaybackClock(pb.Timeline.from_durations(_ANIM), _C)
    assert clock.playing is True
    assert clock.loop is True
    assert clock.position(_C) == 0.0
    assert clock.index(_C) == 0


def test_clock_avance():
    clock = _anim_clock()
    assert clock.index(_C + 0.0625) == 0
    assert clock.index(_C + 0.125) == 1  # límite inferior incluido
    assert clock.index(_C + 0.25) == 1
    assert clock.index(_C + 0.375) == 2
    assert clock.index(_C + 0.75) == 2


def test_clock_salto_de_frames():
    # vídeo a 4 fps (dt = 0.25 s) con 8 frames: de 0 se salta directamente a 5
    clock = pb.PlaybackClock(pb.Timeline.constant(4.0, 8), _C)
    assert clock.index(_C) == 0
    assert clock.index(_C + 1.375) == 5


def test_clock_pausa():
    clock = _anim_clock()
    clock.toggle_play(_C + 0.25)
    assert clock.playing is False
    for dt in (0.25, 1.0, 10.0, 1000.0):
        assert clock.position(_C + dt) == 0.25
        assert clock.index(_C + dt) == 1


def test_clock_reanudacion():
    clock = _anim_clock()
    clock.toggle_play(_C + 0.25)  # pausa en t = 0.25 (frame 1)
    clock.toggle_play(_C + 10.0)  # reanuda 9.75 s después
    assert clock.playing is True
    assert clock.position(_C + 10.0) == 0.25  # sin sumar la pausa
    assert clock.index(_C + 10.0) == 1
    assert clock.index(_C + 10.125) == 2  # t = 0.375


def test_clock_reanudacion_tras_paso():
    # R7.4: si el frame cambió en pausa, se reanuda desde inicio_k
    clock = _anim_clock()
    clock.toggle_play(_C + 0.25)  # pausa en frame 1, t = 0.25
    clock.step(+1, _C + 5.0)  # frame 2
    clock.toggle_play(_C + 6.0)
    assert clock.position(_C + 6.0) == 0.375
    assert clock.index(_C + 6.0) == 2


def test_clock_bucle():
    clock = _anim_clock(loop=True)
    now = _C + 1.0  # t = 1.0 → 1.0 mod 0.875 = 0.125 → frame 1
    assert clock.update(now) is False
    assert clock.playing is True
    assert clock.index(now) == 1
    assert clock.index(_C + 0.875) == 0  # justo en D vuelve al frame 0
    assert clock.index(_C + 2.25) == 2  # 2.25 mod 0.875 = 0.5 → frame 2 (segunda vuelta)


def test_clock_fin_sin_bucle():
    # R6.7: sin bucle, al llegar a D se pausa en el último frame
    clock = _anim_clock(loop=False)
    assert clock.update(_C + 0.75) is False
    assert clock.index(_C + 0.75) == 2
    now = _C + 1.0
    assert clock.update(now) is True
    assert clock.playing is False
    assert clock.index(now) == 2
    assert clock.position(now) == 0.375  # inicio del último frame
    assert clock.update(_C + 50.0) is False
    assert clock.index(_C + 50.0) == 2


def test_clock_espacio_en_ultimo_frame_sin_bucle_reinicia():
    # R6.14: en pausa en el último frame sin bucle, Espacio reanuda desde el frame 0
    clock = _anim_clock(loop=False)
    clock.update(_C + 1.0)
    clock.toggle_play(_C + 2.0)
    assert clock.playing is True
    assert clock.position(_C + 2.0) == 0.0
    assert clock.index(_C + 2.0) == 0
    assert clock.index(_C + 2.125) == 1


def test_clock_paso_adelante():
    clock = _anim_clock(loop=True)
    clock.step(+1, _C)
    assert clock.playing is False  # el paso pausa
    assert clock.index(_C) == 1
    clock.step(+1, _C + 1.0)
    assert clock.index(_C + 1.0) == 2
    clock.step(+1, _C + 2.0)  # R6.10: con bucle, del último al 0
    assert clock.index(_C + 2.0) == 0
    no_loop = _anim_clock(loop=False, playing=False)
    no_loop.step(+1, _C)
    no_loop.step(+1, _C)
    no_loop.step(+1, _C)  # R6.8: sin bucle se queda en el último
    assert no_loop.index(_C) == 2
    assert no_loop.playing is False


def test_clock_paso_atras():
    clock = _anim_clock(loop=True)
    clock.step(-1, _C)  # R6.13: con bucle, del 0 al último
    assert clock.playing is False
    assert clock.index(_C) == 2
    clock.step(-1, _C + 1.0)
    assert clock.index(_C + 1.0) == 1
    no_loop = _anim_clock(loop=False, playing=False)
    no_loop.step(-1, _C)  # R6.9: sin bucle se queda en el 0
    assert no_loop.index(_C) == 0
    assert no_loop.playing is False


def test_clock_inicio_y_alternar_bucle():
    clock = _anim_clock()
    clock.home(_C + 0.5)  # R6.4: vuelve al frame 0 sin pausar
    assert clock.index(_C + 0.5) == 0 and clock.playing is True
    clock.toggle_loop(_C + 0.625)  # R6.5: el frame no cambia (t = 0.125)
    assert clock.loop is False
    assert clock.index(_C + 0.625) == 1


def test_clock_duraciones_invalidas():
    # R7.7: 0 ms y None pasan a 100 ms → inicios 0, 0.1, 0.2; D = 0.45
    durations = pb.sanitize_durations_ms([0, None, 250])
    assert durations == (0.1, 0.1, 0.25)
    clock = pb.PlaybackClock(pb.Timeline.from_durations(durations), 0.0)
    assert clock.index(0.05) == 0
    assert clock.index(0.15) == 1
    assert clock.index(0.3) == 2
    assert clock.index(0.5) == 0  # bucle: 0.5 mod 0.45 = 0.05


def test_clock_fps_invalido():
    # R7.6: FPS inválido → 25 fps (dt = 0.04 s)
    fps, warning = pb.sanitize_fps(0)
    assert fps == 25.0 and warning is not None
    clock = pb.PlaybackClock(pb.Timeline.constant(fps, 100), 0.0)
    assert clock.index(0.1) == 2
    assert clock.index(1.02) == 25


def test_clock_next_change_time_exacto():
    # R7.9: el próximo cambio coincide con el inicio del frame siguiente
    clock = _anim_clock()
    assert clock.next_change_time(_C) == _C + 0.125
    assert clock.next_change_time(_C + 0.25) == _C + 0.375
    clock.toggle_play(_C + 0.25)
    assert clock.next_change_time(_C + 0.25) is None  # en pausa


@pytest.mark.parametrize(
    "timeline",
    [pb.Timeline.from_durations([0.5]), pb.Timeline.constant(25.0, 1)],
    ids=["animacion", "video"],
)
def test_clock_un_solo_frame(timeline):
    # R2.6: con N = 1 el índice es siempre 0 y no hay espera por tiempo
    clock = pb.PlaybackClock(timeline, _C)
    for dt in (0.0, 0.25, 1.0, 100.0):
        assert clock.index(_C + dt) == 0
    assert clock.next_change_time(_C) is None
    clock.step(+1, _C + 1.0)
    assert clock.index(_C + 1.0) == 0
    clock.step(-1, _C + 2.0)
    assert clock.index(_C + 2.0) == 0
    clock.toggle_play(_C + 3.0)
    assert clock.index(_C + 4.0) == 0


# Feature: soporte-video, Property 14: Medidor de FPS efectivo
# **Validates: Requirements 8.3, 8.8**

_gap = st.floats(min_value=0.0, max_value=0.6, allow_nan=False, allow_infinity=False)


@given(
    _wall,
    st.lists(_gap, max_size=20),  # registros previos al reset (deben olvidarse)
    _gap,
    st.lists(_gap, max_size=80),  # incrementos tras el reset (secuencia creciente)
    st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
def test_property_effective_fps(c, before, reset_gap, gaps, tail):
    meter = pb.EffectiveFps(c)
    t = c
    for g in before:
        t += g
        meter.record(t)
    r = t + reset_gap
    meter.reset(r)
    records = []
    t = r
    for g in gaps:
        t += g
        records.append(t)
        meter.record(t)
    now = t + tail  # consulta en o tras el último registro
    expected = None if now - r < 1.0 else float(sum(1 for x in records if now - 1.0 < x <= now))
    assert meter.value(now) == expected


# Feature: soporte-video, Property 15: Formato de las etiquetas del panel
# **Validates: Requirements 8.1, 8.2, 8.3**

import re  # noqa: E402

_TIME_RE = re.compile(r"^\d+\.\d{2} s$")
_FPS_RE = re.compile(r"^\d+\.\d$")
_nonneg = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)


@given(
    st.integers(min_value=0, max_value=10**9),
    st.one_of(st.none(), st.integers(min_value=1, max_value=10**9)),
    _nonneg,
    _nonneg,
)
def test_property_panel_labels(k, n, t, f):
    expected_frame = f"frame {k}/?" if n is None else f"frame {k}/{n}"
    assert pb.frame_label(k, n) == expected_frame
    assert _TIME_RE.fullmatch(pb.time_label(t))
    assert _FPS_RE.fullmatch(pb.fps_label(f))
    assert pb.fps_label(None) == "—"


# ---------------------------------------------------------------------------
# Player: fuente y textura falsas (CPU, sin GL), reutilizables por otros tests
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

from src import state as app_state  # noqa: E402


class _FakeSource:
    """FrameSource falsa con N frames RGBA (H, W, 4) distintos entre sí.

    `frames[k]` está relleno con el valor k (uint8), así cada frame es
    distinguible. `overrides` permite sustituir lo que devuelve read(k) por
    otro valor (None, array de otro tamaño) o por una excepción a lanzar.
    """

    backend = "fake"

    def __init__(self, n, size=(4, 3), n_declared="same", fps=25.0, durations=None, overrides=None):
        w, h = size
        self.path = Path("fake/video.gif")
        self.size = (w, h)
        self.n_frames = n if n_declared == "same" else n_declared
        self.fps = fps
        self.durations = durations
        self.frames = [np.full((h, w, 4), k % 256, dtype=np.uint8) for k in range(n)]
        self.overrides = dict(overrides or {})
        self.reads: list[int] = []

    def open(self):
        pass

    def read(self, index):
        index = int(index)
        self.reads.append(index)
        if index in self.overrides:
            value = self.overrides[index]
            if isinstance(value, BaseException):
                raise value
            return value
        return self.frames[index] if 0 <= index < len(self.frames) else None

    def close(self):
        pass


class _FakeTexture:
    """Textura falsa: solo registra cada llamada a write()."""

    def __init__(self):
        self.writes: list = []

    def write(self, data):
        self.writes.append(data)


# Feature: soporte-video, Property 9: El Player sube a la GPU exactamente los frames objetivo
# **Validates: Requirements 1.4, 5.1, 5.8, 5.9, 7.2**

_player_ops = st.lists(
    st.one_of(
        st.tuples(st.just("tick"), st.floats(min_value=0.0, max_value=1.5, allow_nan=False)),
        st.tuples(st.just("action"), st.sampled_from(sorted(app_state.PLAYBACK_ACTIONS))),
    ),
    min_size=1,
    max_size=40,
)
_player_kind = st.one_of(
    st.tuples(st.just("anim"), st.lists(
        st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=12)),
    st.tuples(st.just("video"), st.tuples(_fps, st.integers(min_value=1, max_value=12))),
)


@given(_player_kind, st.booleans(), st.booleans(), _wall, _player_ops)
def test_property_player_uploads_exactly_target_frames(kind, playing, loop, c, ops):
    tag, params = kind
    if tag == "anim":
        tl = pb.Timeline.from_durations(params)
        source = _FakeSource(len(params), durations=tuple(params))
    else:
        fps, n = params
        tl = pb.Timeline.constant(fps, n)
        source = _FakeSource(n, fps=fps)
    texture = _FakeTexture()
    warnings: list[str] = []
    clock = pb.PlaybackClock(tl, c, playing=playing, loop=loop)
    player = pb.Player(source, texture, clock, pb.EffectiveFps(c), warnings.append)
    d = tl.total
    now = c
    expected_writes = 0
    for op, value in ops:
        if op == "action":
            before = len(texture.writes)
            player.handle(value, now)
            assert len(texture.writes) == before  # las acciones no suben nada
            continue
        now += value * d  # instantes de tick crecientes (gap 0 incluido)
        shown_before = player.shown_index
        before = len(texture.writes)
        changes = player.update(now)
        target = clock.index(now)
        if target != shown_before:
            # una sola subida y con el frame objetivo exacto, nunca un intermedio
            expected_writes += 1
            assert len(texture.writes) == before + 1
            assert texture.writes[-1] is source.frames[target]
            assert changes.frame is True
            assert player.shown_index == target
        else:
            # sin cambio de índice: ninguna llamada a la GPU (R5.9)
            assert len(texture.writes) == before
            assert changes.frame is False
            assert player.shown_index == shown_before
    assert len(texture.writes) == expected_writes
    assert warnings == []


# Feature: soporte-video, Property 10: Un frame ilegible se convierte en el final del vídeo
# **Validates: Requirements 3.6**

from src.video import FrameReadError  # noqa: E402

_failing_timeline = st.one_of(
    st.tuples(st.just("anim"), st.lists(
        st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=12)),
    st.tuples(st.just("video"), st.tuples(_fps, st.integers(min_value=2, max_value=12))),
)
# avance por tick como fracción de la duración mínima de frame (< 1): el índice
# sube como mucho 1 por tick, así el primer índice ≥ j que se pide es j
_tick_fraction = st.floats(min_value=0.0, max_value=0.9, allow_nan=False)


@given(
    _failing_timeline,
    st.data(),
    st.sampled_from(["none", "error"]),
    st.booleans(),
    _wall,
    st.lists(_tick_fraction, max_size=30),
    st.lists(_tick_fraction, max_size=30),
)
def test_property_unreadable_frame_becomes_end(kind, data, mode, loop, c, before_ticks, after_ticks):
    tag, params = kind
    if tag == "anim":
        tl = pb.Timeline.from_durations(params)
        n, min_dt = len(params), min(params)
        durations = tuple(params)
        fps = 25.0
    else:
        fps, n = params
        tl = pb.Timeline.constant(fps, n)
        min_dt, durations = 1.0 / fps, None
    j = data.draw(st.integers(min_value=1, max_value=n - 1), label="j")
    failure = None if mode == "none" else FrameReadError(j, "datos corruptos")
    source = _FakeSource(n, fps=fps, durations=durations, overrides={j: failure})
    texture = _FakeTexture()
    warnings: list[str] = []
    clock = pb.PlaybackClock(tl, c, playing=True, loop=loop)
    player = pb.Player(source, texture, clock, pb.EffectiveFps(c), warnings.append)

    now = c

    def tick(fraction):
        nonlocal now
        now += fraction * min_dt
        player.update(now)

    for f in before_ticks:
        tick(f)
    # completa la secuencia hasta que el reloj alcance j: cada tick cae dentro
    # del frame siguiente (inicio_{k+1} + ¼ de la duración mínima)
    guard = 0
    while j not in source.reads:
        k = clock.index(now)
        now += tl.start(k + 1) - clock.position(now) + 0.25 * min_dt
        player.update(now)
        guard += 1
        assert guard <= n
    # el primer índice ≥ j pedido a la fuente es exactamente j
    assert min(k for k in source.reads if k >= j) == j
    for f in after_ticks:
        tick(f)

    assert clock.timeline.n_frames == j
    assert player.shown_index <= j - 1
    # ninguna subida de un frame ≥ j (los frames de la fuente están rellenos con k)
    assert all(int(w.flat[0]) < j for w in texture.writes)
    assert all(k < j for k in source.reads[source.reads.index(j) + 1:])
    assert len(warnings) == 1
    assert str(source.path) in warnings[0]
    assert f"frame {j} " in warnings[0]


# ---------------------------------------------------------------------------
# Tests unitarios del Player (R3.7, R2.6, R6.1, R8.5, R8.8)
# ---------------------------------------------------------------------------

_PC = 100.0  # instante de pared inicial de los tests del Player


def _make_player(n=4, fps=4.0, playing=True, loop=True, overrides=None, n_declared="same"):
    """Player de vídeo a 4 fps (dt = 0.25 s) con fuente y textura falsas."""
    source = _FakeSource(n, fps=fps, overrides=overrides, n_declared=n_declared)
    declared = source.n_frames
    tl = pb.Timeline.constant(fps, declared)
    clock = pb.PlaybackClock(tl, _PC, playing=playing, loop=loop)
    texture = _FakeTexture()
    warnings: list[str] = []
    meter = pb.EffectiveFps(_PC)
    player = pb.Player(source, texture, clock, meter, warnings.append)
    return player, source, texture, warnings


def test_player_frame_de_otro_tamano_no_se_sube():
    # R3.7: tamaño fuente 4×3 (W×H); los frames 1 y 2 llegan con otro tamaño
    wrong1 = np.zeros((5, 6, 4), dtype=np.uint8)  # 6×5
    wrong2 = np.zeros((3, 4, 3), dtype=np.uint8)  # mismo W×H pero 3 canales
    player, source, texture, warnings = _make_player(overrides={1: wrong1, 2: wrong2})

    changes = player.update(_PC + 0.3)  # índice 1
    assert changes.frame is False
    assert texture.writes == []
    assert player.shown_index == 0
    assert len(warnings) == 1
    assert "frame 1" in warnings[0] and "6×5" in warnings[0] and "4×3" in warnings[0]

    player.update(_PC + 0.35)  # sigue en 1: se vuelve a pedir, sin nuevo aviso
    player.update(_PC + 0.6)  # índice 2, también inválido
    assert texture.writes == []
    assert len(warnings) == 1  # aviso único

    changes = player.update(_PC + 0.8)  # índice 3, válido: se sube
    assert changes.frame is True
    assert len(texture.writes) == 1 and texture.writes[0] is source.frames[3]
    assert player.shown_index == 3
    assert player.clock.playing is True  # el reloj sigue
    assert len(warnings) == 1


def test_player_handle_play_pause():
    player, _, texture, _ = _make_player(playing=True)
    assert player.handle(app_state.ACTION_PLAY_PAUSE, _PC + 0.3) == pb.Changes(False, True)
    assert player.clock.playing is False
    assert player.clock.index(_PC + 5.0) == 1  # en pausa el índice no avanza
    assert player.handle(app_state.ACTION_PLAY_PAUSE, _PC + 5.0) == pb.Changes(False, True)
    assert player.clock.playing is True
    assert texture.writes == []  # las acciones no suben nada


def test_player_handle_next_prev_first_toggle_loop():
    player, source, texture, _ = _make_player(playing=False, loop=True)
    now = _PC + 1.0

    assert player.handle(app_state.ACTION_NEXT_FRAME, now) == pb.Changes(False, True)
    assert player.clock.index(now) == 1 and player.clock.playing is False
    assert texture.writes == []
    assert player.update(now).frame is True  # el frame lo sube update()
    assert texture.writes[-1] is source.frames[1]

    player.handle(app_state.ACTION_PREV_FRAME, now)
    assert player.clock.index(now) == 0
    player.handle(app_state.ACTION_PREV_FRAME, now)  # con bucle: del 0 al último
    assert player.clock.index(now) == 3

    player.handle(app_state.ACTION_FIRST_FRAME, now)
    assert player.clock.index(now) == 0 and player.clock.playing is False

    assert player.handle(app_state.ACTION_TOGGLE_LOOP, now) == pb.Changes(False, True)
    assert player.clock.loop is False
    player.handle(app_state.ACTION_TOGGLE_LOOP, now)
    assert player.clock.loop is True


def test_player_handle_accion_desconocida():
    player, _, _, _ = _make_player()
    assert player.handle(app_state.ACTION_RELOAD, _PC) == pb.Changes(False, False)
    assert player.clock.playing is True


def test_player_un_solo_frame_deshabilitado():
    # R2.6: con 1 frame los controles de reproducción no hacen nada
    player, _, texture, _ = _make_player(n=1)
    assert player.enabled is False
    for action in sorted(app_state.PLAYBACK_ACTIONS):
        assert player.handle(action, _PC + 0.5) == pb.Changes(False, False)
    assert player.clock.playing is True and player.clock.loop is True
    assert player.clock.index(_PC + 0.5) == 0
    assert player.update(_PC + 2.0) == pb.Changes(False, False)
    assert texture.writes == []


def test_player_con_n_desconocido_esta_habilitado():
    player, _, _, _ = _make_player(n_declared=None)
    assert player.enabled is True


def test_player_reanudar_reinicia_fps():
    # R8.8: el FPS_Efectivo se mide desde la reanudación
    player, _, _, _ = _make_player()
    for k in range(1, 4):
        player.update(_PC + 0.25 * k + 0.01)  # sube frames 1..3
    assert player.fps.value(_PC + 1.0) == 3.0

    player.handle(app_state.ACTION_PLAY_PAUSE, _PC + 1.0)  # pausa: no reinicia
    assert player.fps.value(_PC + 1.2) is not None

    resume = _PC + 5.0
    player.handle(app_state.ACTION_PLAY_PAUSE, resume)  # reanuda: reinicia
    assert player.fps.value(resume + 0.5) is None
    assert player.fps.value(resume + 1.0) == 0.0  # registros previos descartados


def test_title_suffix():
    # R8.5: "frame k/n | reproduciendo" / "… | pausa"
    player, _, _, _ = _make_player()
    assert pb.title_suffix(player, _PC) == "frame 0/4 | reproduciendo"
    player.update(_PC + 0.6)
    assert pb.title_suffix(player, _PC + 0.6) == "frame 2/4 | reproduciendo"
    player.handle(app_state.ACTION_PLAY_PAUSE, _PC + 0.6)
    assert pb.title_suffix(player, _PC + 0.6) == "frame 2/4 | pausa"

    unknown, _, _, _ = _make_player(n_declared=None)
    assert pb.title_suffix(unknown, _PC) == "frame 0/? | reproduciendo"
