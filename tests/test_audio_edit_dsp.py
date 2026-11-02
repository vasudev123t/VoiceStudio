"""Audio Edit DSP: untouched audio must stay byte-identical."""
from __future__ import annotations

import numpy as np
import pytest

from services import audio_edit_dsp as dsp

SR = 1000
LSR = 16000  # loudness needs a real sample rate for the K-weighting filters


def _noise(n: int, ch: int = 1, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal((ch, n)) * 0.1).astype(np.float32)


def _sine(seconds: float, amp: float, sr: int = LSR, ch: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return np.tile((amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32), (ch, 1))


def test_equal_power_sums_to_unity():
    out, fin = dsp.equal_power(64)
    np.testing.assert_allclose(out ** 2 + fin ** 2, 1.0, atol=1e-6)
    assert out[0] > 0.99 and fin[-1] > 0.99
    assert dsp.equal_power(0)[0].shape == (0,)


def test_as_2d_and_match_channels():
    assert dsp.as_2d(np.zeros(5)).shape == (1, 5)
    stereo = dsp.match_channels(np.ones((1, 4)), 2)
    assert stereo.shape == (2, 4)
    np.testing.assert_allclose(dsp.match_channels(np.array([[1.0, 1.0], [0.0, 0.0]]), 1), [[0.5, 0.5]])
    with pytest.raises(ValueError):
        dsp.as_2d(np.zeros((1, 2, 3)))


def test_splice_replaces_body_and_keeps_outside_identical():
    base = _noise(5000)
    x = dsp.ms_to_samples(30, SR)
    insert = np.full((1, 700 + 2 * x), 0.25, np.float32)
    out = dsp.splice(base, SR, 2.0, 2.5, insert, 30)
    assert out.shape == (1, 5000 - 500 + 700)
    assert np.array_equal(out[:, : 2000 - x], base[:, : 2000 - x])
    assert np.array_equal(out[:, 2700 + x:], base[:, 2500 + x:])
    np.testing.assert_allclose(out[:, 2000:2700], 0.25)


def test_splice_at_file_edges_shrinks_crossfade():
    base = _noise(1000)
    x = dsp.ms_to_samples(30, SR)
    insert = np.full((1, 100 + 2 * x), 0.5, np.float32)
    out = dsp.splice(base, SR, 0.0, 0.1, insert, 30)
    assert out.shape == (1, 1000)
    np.testing.assert_allclose(out[:, :100], 0.5)
    assert np.array_equal(out[:, 100 + x:], base[:, 100 + x:])
    tail = dsp.splice(base, SR, 0.9, 1.0, insert, 30)
    np.testing.assert_allclose(tail[:, 900:], 0.5)
    assert np.array_equal(tail[:, : 900 - x], base[:, : 900 - x])


def test_splice_delete_single_crossfade():
    base = _noise(3000)
    x = dsp.ms_to_samples(20, SR)
    out = dsp.splice(base, SR, 1.0, 1.5, None, 20)
    assert out.shape == (1, 3000 - 500 - x)
    assert np.array_equal(out[:, : 1000 - x], base[:, : 1000 - x])
    assert np.array_equal(out[:, 1000:], base[:, 1500 + x:])


def test_splice_rejects_bad_ranges_and_short_inserts():
    base = _noise(1000)
    with pytest.raises(ValueError):
        dsp.splice(base, SR, 0.8, 1.2, None, 10)
    with pytest.raises(ValueError):
        dsp.splice(base, SR, 0.5, 0.4, None, 10)
    with pytest.raises(ValueError):
        dsp.splice(base, SR, 0.2, 0.3, np.zeros((1, 10), np.float32), 30)


def test_apply_splices_runs_latest_first_and_rejects_overlap():
    base = _noise(6000)
    x = dsp.ms_to_samples(10, SR)
    a = np.full((1, 300 + 2 * x), 0.2, np.float32)
    b = np.full((1, 900 + 2 * x), -0.2, np.float32)
    expected = dsp.splice(dsp.splice(base, SR, 4.0, 4.5, b, 10), SR, 1.0, 1.5, a, 10)
    out = dsp.apply_splices(base, SR, [(1.0, 1.5, a), (4.0, 4.5, b)], 10)
    assert np.array_equal(out, expected)
    assert out.shape[1] == 6000 - 500 + 300 - 500 + 900
    with pytest.raises(ValueError):
        dsp.apply_splices(base, SR, [(1.0, 2.0, None), (1.5, 2.5, None)], 10)


def test_fit_length_pads_and_trims():
    assert dsp.fit_length(np.ones((1, 5)), 8).shape == (1, 8)
    assert dsp.fit_length(np.ones((2, 9)), 4).shape == (2, 4)


def test_loudness_tracks_gain_and_floors_silence():
    quiet = dsp.loudness_lufs(_sine(1.0, 0.1), LSR)
    loud = dsp.loudness_lufs(_sine(1.0, 0.4), LSR)
    assert loud - quiet == pytest.approx(12.04, abs=0.3)
    assert dsp.loudness_lufs(np.zeros((1, LSR), np.float32), LSR) == dsp.SILENCE_LUFS
    short = dsp.loudness_lufs(_sine(0.1, 0.4), LSR)
    assert short == pytest.approx(loud, abs=1.0)


def test_match_loudness_matches_clamps_and_skips_silence():
    context = _sine(2.0, 0.4)
    matched = dsp.match_loudness(_sine(0.5, 0.1), context, LSR)
    assert dsp.loudness_lufs(matched, LSR) == pytest.approx(dsp.loudness_lufs(context, LSR), abs=0.5)
    tiny = dsp.match_loudness(_sine(0.5, 0.01), context, LSR)
    gain_db = 20 * np.log10(np.max(np.abs(tiny)) / 0.01)
    assert gain_db == pytest.approx(dsp.MAX_GAIN_DB, abs=0.1)
    ins = _sine(0.5, 0.1)
    assert np.array_equal(dsp.match_loudness(ins, np.zeros((1, LSR), np.float32), LSR), ins)


def test_context_around_excludes_the_span():
    audio = np.arange(10_000, dtype=np.float32)[None, :]
    ctx = dsp.context_around(audio, SR, 4.0, 5.0, 2.0)
    assert ctx.shape == (1, 4000)
    assert ctx[0, 0] == 2000 and ctx[0, 1999] == 3999 and ctx[0, 2000] == 5000


def test_remix_applies_gain_and_limits_peak():
    vocals = np.full((1, 100), 0.5, np.float32)
    inst = np.full((2, 120), 0.3, np.float32)
    mixed = dsp.remix(vocals, inst, 0.0)
    assert mixed.shape == (2, 120)
    assert np.max(np.abs(mixed)) <= 0.999 + 1e-6
    soft = dsp.remix(vocals, np.zeros((2, 100), np.float32), -6.0)
    assert soft[0, 0] == pytest.approx(0.5 * dsp.db_to_gain(-6.0), rel=1e-5)


def test_resample_changes_length():
    out = dsp.resample(np.zeros((2, 24000), np.float32), 24000, 44100)
    assert out.shape == (2, 44100)
    same = np.ones((1, 10), np.float32)
    assert dsp.resample(same, 16000, 16000) is not None


def test_estimate_duration_scales_and_clamps():
    assert dsp.estimate_duration("hello", "hello", 1.0) == pytest.approx(1.0)
    assert dsp.estimate_duration("hi", "hello there", 1.0) == pytest.approx(3.0 + 1.0)
    assert dsp.estimate_duration("hello there", "a", 1.0) == pytest.approx(0.25)
    assert dsp.estimate_duration("ab", "abcd", 0.5) == pytest.approx(1.0)
