"""A/B take scoring (review item 5) — hermetic."""
import numpy as np
import pytest
import soundfile as sf

from services import take_scoring as ts


def test_error_rates():
    assert ts.error_rate("the cat sat", "the cat sat", unit="wer") == 0.0
    assert ts.error_rate("the cat sat", "the bat sat down", unit="wer") == pytest.approx(2 / 3, 1e-3)
    assert ts.error_rate("Hello, world!", "hello world", unit="wer") == 0.0
    # Kannada keeps its vowel signs (combining marks) when normalizing.
    assert ts.normalize_for_wer("ನನಗೆ ಎರಡು!") == "ನನಗೆ ಎರಡು"
    assert ts.error_rate("ನನಗೆ ಎರಡು", "ನನಗೆ ಎರಡು", unit="cer") == 0.0
    assert ts.error_rate("[laugh] hi there", "hi there", unit="wer") == 0.0
    assert ts.error_rate("", "", unit="wer") == 0.0


def test_metric_unit_by_language():
    assert ts.metric_unit("Kannada") == "cer"
    assert ts.metric_unit("kn") == "cer"
    assert ts.metric_unit("English") == "wer"
    assert ts.metric_unit(None) == "wer"
    assert ts.asr_language("Kannada") == "kn"
    assert ts.asr_language("Auto") is None


def test_overall_prefers_better_takes():
    good = {"secs": {"value": 0.95, "method": "mfcc-stats"}, "wer": {"value": 0.0},
            "mos": {"value": 4.2}}
    bad = {"secs": {"value": 0.6, "method": "mfcc-stats"}, "wer": {"value": 0.5},
           "mos": {"value": 2.0}}
    assert ts.overall(good) > ts.overall(bad)
    assert ts.overall({}) == 0.0


class _FakeScorer:
    def __init__(self):
        self.closed = False

    def secs(self, x, ref):
        return {"value": 0.9 if x.std() > 0.05 else 0.5, "method": "fake"}

    def wer(self, x, text, lang):
        return {"value": 0.1, "unit": "wer", "method": "fake"}

    def mos(self, x):
        return {"value": 4.0, "method": "fake"}

    def close(self):
        self.closed = True


def test_score_takes_ranks_and_isolates_errors(tmp_path):
    sr = 16000
    loud = (0.3 * np.sin(np.linspace(0, 2000, sr))).astype("float32")
    quiet = (0.01 * np.sin(np.linspace(0, 2000, sr))).astype("float32")
    for name, x in (("a.wav", quiet), ("b.wav", loud), ("ref.wav", loud)):
        sf.write(tmp_path / name, x, sr)
    scorer = _FakeScorer()
    takes = [{"id": "a", "path": str(tmp_path / "a.wav"), "text": "x"},
             {"id": "b", "path": str(tmp_path / "b.wav"), "text": "x"},
             {"id": "c", "path": str(tmp_path / "missing.wav"), "text": "x"}]
    out = ts.score_takes(takes, reference=str(tmp_path / "ref.wav"), scorer=scorer)
    by = {r["id"]: r for r in out}
    assert by["b"]["rank"] == 1 and by["a"]["rank"] == 2
    assert by["c"]["error"] and "rank" not in by["c"]
    assert scorer.closed


def test_real_fallback_scorer_runs_offline(tmp_path, monkeypatch):
    """The on-disk fallbacks (MFCC SECS + DSP MOS) work without any model."""
    monkeypatch.setattr(ts, "capabilities", lambda: {
        "secs": {"method": "mfcc-stats"}, "wer": {"available": False},
        "mos": {"method": "dsp-proxy"}})
    sr = 16000
    t = np.arange(sr * 2) / sr
    voice = (0.3 * np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype("float32")
    sf.write(tmp_path / "take.wav", voice, sr)
    sf.write(tmp_path / "ref.wav", voice, sr)
    scorer = ts.Scorer(device="cpu")
    out = ts.score_takes([{"id": "t", "path": str(tmp_path / "take.wav"), "text": "hi"}],
                         reference=str(tmp_path / "ref.wav"), scorer=scorer)
    row = out[0]
    assert row["error"] is None
    assert row["secs"]["method"] == "mfcc-stats" and row["secs"]["value"] > 0.99
    assert row["mos"]["method"] == "dsp-proxy" and 1.0 <= row["mos"]["value"] <= 5.0
    assert row["wer"] is None


def test_capabilities_report_missing_eval_models(monkeypatch):
    monkeypatch.setattr(ts, "eval_models_dir", lambda: None)
    caps = ts.capabilities()
    assert caps["secs"]["research_grade"] is False and "TTS_eval_models" in caps["secs"]["missing"]
    assert caps["mos"]["method"] == "dsp-proxy"
