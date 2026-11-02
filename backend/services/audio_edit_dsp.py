"""Pure DSP for Audio Edit: splicing, crossfades, loudness, remix.

Arrays are float32 shaped ``(channels, samples)``. Nothing here touches disk,
models or the network, so every function is unit-testable. The splice
contract (see :func:`splice`) is what keeps unedited audio byte-identical.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

#: Loudness reported for digital silence (BS.1770 is -inf there).
SILENCE_LUFS = -120.0
#: Largest correction :func:`match_loudness` will apply, either way.
MAX_GAIN_DB = 12.0
#: Seconds of audio either side of a span used as the loudness reference.
CONTEXT_S = 2.0
#: Below this, loudness is too unreliable to match against.
_MATCH_FLOOR_LUFS = -60.0


def as_2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {a.shape}")
    return a


def match_channels(x, channels: int) -> np.ndarray:
    a = as_2d(x)
    if a.shape[0] == channels:
        return a
    if channels == 1:
        return a.mean(axis=0, keepdims=True).astype(np.float32)
    if a.shape[0] == 1:
        return np.repeat(a, channels, axis=0)
    raise ValueError(f"cannot map {a.shape[0]} channels to {channels}")


def ms_to_samples(ms: float, sr: int) -> int:
    return max(0, int(round(float(ms) * sr / 1000.0)))


def equal_power(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(fade_out, fade_in)`` of length *n* with ``out**2 + in**2 == 1``."""
    if n <= 0:
        empty = np.zeros(0, np.float32)
        return empty, empty
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    return (np.cos(t * math.pi / 2).astype(np.float32),
            np.sin(t * math.pi / 2).astype(np.float32))


def fit_length(x, n: int) -> np.ndarray:
    a = as_2d(x)
    if a.shape[1] >= n:
        return a[:, :n].copy()
    return np.pad(a, ((0, 0), (0, n - a.shape[1])))


def splice(base, sr: int, start_s: float, end_s: float,
           insert: Optional[np.ndarray], crossfade_ms: float) -> np.ndarray:
    """Replace ``base[a:b]`` (``a``/``b`` = *start_s*/*end_s* in samples).

    *insert* carries ``x`` samples of lead-in, the body, and ``x`` samples
    of lead-out (``x`` = *crossfade_ms* in samples). The margins
    crossfade with ``base[a-x:a]`` and ``base[b:b+x]``; at a file edge the
    margin shrinks to the room available. ``insert=None`` deletes the range
    with one crossfade at the seam (the output is ``b - a + x`` shorter).
    Samples outside ``[a-x, b+x)`` are copied unchanged.
    """
    base = as_2d(base)
    n = base.shape[1]
    a, b = int(round(start_s * sr)), int(round(end_s * sr))
    if not 0 <= a < b <= n:
        raise ValueError(f"span {start_s:.3f}-{end_s:.3f}s is outside the audio")
    x = ms_to_samples(crossfade_ms, sr)
    if insert is None:
        x = min(x, a, n - b)
        fade_out, fade_in = equal_power(x)
        seam = base[:, a - x:a] * fade_out + base[:, b:b + x] * fade_in
        return np.concatenate([base[:, :a - x], seam, base[:, b + x:]], axis=1).astype(np.float32)
    ins = match_channels(insert, base.shape[0])
    if ins.shape[1] < 2 * x + 1:
        raise ValueError("insert is shorter than its crossfade margins")
    lead, tail = min(x, a), min(x, n - b)
    body = ins[:, x - lead: ins.shape[1] - (x - tail)]
    out_l, in_l = equal_power(lead)
    out_t, in_t = equal_power(tail)
    head = base[:, a - lead:a] * out_l + body[:, :lead] * in_l
    end = body.shape[1]
    tail_mix = body[:, end - tail:] * out_t + base[:, b:b + tail] * in_t
    middle = body[:, lead:end - tail]
    return np.concatenate(
        [base[:, :a - lead], head, middle, tail_mix, base[:, b + tail:]], axis=1,
    ).astype(np.float32)


def apply_splices(base, sr: int,
                  edits: Sequence[tuple[float, float, Optional[np.ndarray]]],
                  crossfade_ms: float) -> np.ndarray:
    """Apply non-overlapping *edits* ``(start_s, end_s, insert|None)``.

    Runs from the latest edit back, so earlier offsets stay valid when an
    insert changes the length.
    """
    ordered = sorted(edits, key=lambda e: e[0])
    for (s0, e0, _), (s1, _e1, _) in zip(ordered, ordered[1:]):
        if s1 < e0:
            raise ValueError(f"edits overlap at {s1:.3f}s")
    out = as_2d(base)
    for start_s, end_s, insert in reversed(ordered):
        out = splice(out, sr, start_s, end_s, insert, crossfade_ms)
    return out


def loudness_lufs(x, sr: int) -> float:
    """Integrated BS.1770 loudness; short clips are tiled to 400 ms."""
    import torch
    import torchaudio

    a = as_2d(x)
    if a.size == 0 or float(np.max(np.abs(a))) < 1e-6:
        return SILENCE_LUFS
    need = int(math.ceil(0.4 * sr))
    if a.shape[1] < need:
        a = np.tile(a, (1, int(math.ceil(need / a.shape[1]))))
    value = float(torchaudio.functional.loudness(torch.from_numpy(np.ascontiguousarray(a)), sr))
    return value if math.isfinite(value) else SILENCE_LUFS


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def match_loudness(insert, context, sr: int) -> np.ndarray:
    """Scale *insert* to *context*'s loudness, by at most ±MAX_GAIN_DB."""
    ins = as_2d(insert)
    target = loudness_lufs(context, sr)
    current = loudness_lufs(ins, sr)
    if target < _MATCH_FLOOR_LUFS or current < _MATCH_FLOOR_LUFS:
        return ins
    gain_db = float(np.clip(target - current, -MAX_GAIN_DB, MAX_GAIN_DB))
    return (ins * db_to_gain(gain_db)).astype(np.float32)


def context_around(audio, sr: int, start_s: float, end_s: float,
                   context_s: float = CONTEXT_S) -> np.ndarray:
    """The *context_s* seconds before and after a span, span excluded."""
    a = as_2d(audio)
    n = a.shape[1]
    s, e = int(round(start_s * sr)), int(round(end_s * sr))
    c = int(round(context_s * sr))
    return np.concatenate([a[:, max(0, s - c):s], a[:, e:min(n, e + c)]], axis=1)


def remix(vocals, instrumental, vocal_gain_db: float) -> np.ndarray:
    """Stereo ``vocals * gain + instrumental``, peak-limited to 0.999."""
    n = max(as_2d(vocals).shape[1], as_2d(instrumental).shape[1])
    v = fit_length(match_channels(vocals, 2), n)
    i = fit_length(match_channels(instrumental, 2), n)
    mix = v * db_to_gain(vocal_gain_db) + i
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.999:
        mix = mix * (0.999 / peak)
    return mix.astype(np.float32)


def resample(x, sr_from: int, sr_to: int) -> np.ndarray:
    a = as_2d(x)
    if sr_from == sr_to:
        return a
    import torch
    import torchaudio

    out = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(a)), sr_from, sr_to)
    return out.numpy().astype(np.float32)


def estimate_duration(original_text: str, replacement_text: str, original_s: float) -> float:
    """Seconds the replacement should take: the UTF-8 length ratio of the
    texts times the original span, clamped to ``[0.25, 3 * original + 1]``."""
    o = max(1, len(original_text.strip().encode("utf-8")))
    r = len(replacement_text.strip().encode("utf-8"))
    estimate = original_s * r / o
    return float(min(max(estimate, 0.25), 3.0 * original_s + 1.0))
