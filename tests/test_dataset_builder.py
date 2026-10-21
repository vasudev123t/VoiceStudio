"""Voice Dataset Builder (review item 4) — hermetic: no models needed."""
import json
import time

import numpy as np
import pytest
import soundfile as sf

from services import dataset_builder as db


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_DATASETS_DIR", str(tmp_path / "datasets"))
    db._JOBS.clear()
    yield tmp_path / "datasets"
    db._JOBS.clear()


def _voice(sr, seconds, f0=140.0, seed=0, amp=0.3):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    syll = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)  # 4 Hz syllables
    return (amp * tone / 3 * syll + 0.001 * rng.standard_normal(len(t))).astype("float32")


def test_merge_spans_merges_splits_and_drops():
    spans = [(0.0, 1.0), (1.2, 2.0), (5.0, 5.4), (10.0, 40.0)]
    out = db.merge_spans(spans, min_s=1.5, max_s=12.0)
    assert out[0] == (0.0, 2.0)  # merged across a 0.2 s gap
    assert all(b - a >= 1.5 for a, b in out)  # 0.4 s blip dropped
    assert all(b - a <= 12.0 + 1e-6 for a, b in out)  # 30 s split
    assert len([s for s in out if s[0] >= 10.0]) == 3


def test_speaker_keep_rules():
    assert db.speaker_keep([0.9, 0.2, 0.6], 0.5) == [True, False, True]
    sims = [0.95, 0.94, 0.96, 0.93, 0.95, 0.2]
    assert db.speaker_keep(sims, None) == [True] * 5 + [False]


def test_mfcc_speaker_filter_rejects_a_different_voice():
    sr = db.ANALYSIS_SR
    emb = db._MfccEmbedder()
    same = [emb.embed(_voice(sr, 2.0, 130 + i, seed=i)) for i in range(6)]
    other = emb.embed(np.random.default_rng(9).standard_normal(sr * 2).astype("float32") * 0.2)
    sims = db.speaker_scores(same + [other], None, centered=True)
    keep = db.speaker_keep(sims, None)
    assert keep[:6] == [True] * 6 and keep[6] is False


def test_quality_score_prefers_clean_audio():
    sr = 44100
    clean = _voice(sr, 2.0)
    noisy = clean + 0.2 * np.random.default_rng(1).standard_normal(len(clean)).astype("float32")
    qc = db.quality_score(db.quality_metrics(clean, sr), 0.9)
    qn = db.quality_score(db.quality_metrics(noisy, sr), 0.9)
    assert qc > qn
    clipped = np.clip(clean * 10, -1, 1)
    assert db.quality_metrics(clipped, sr)["clip_ratio"] > 0.01


def test_laugh_markers_and_breaths():
    sr = 16000
    assert db.laugh_likelihood(np.zeros(sr, "float32"), sr, "(laughs) okay", 1.0) >= 0.9
    assert db.laugh_likelihood(np.zeros(sr, "float32"), sr, "ಹಹಹ", 1.0) >= 0.9
    x = np.zeros(sr * 4, "float32")
    x[: sr] = _voice(sr, 1.0)
    x[3 * sr:] = _voice(sr, 1.0)
    rng = np.random.default_rng(2)
    x[int(1.6 * sr):int(2.0 * sr)] = 0.01 * rng.standard_normal(int(0.4 * sr))  # breath
    x += 0.0002 * rng.standard_normal(len(x)).astype("float32")
    events = db.find_breaths(x, sr, [(0.0, 1.0), (3.0, 4.0)], noise_db=-70.0)
    assert len(events) == 1 and 1.5 <= events[0][0] <= 1.7


def test_ids_and_files_are_confined(root):
    for bad in ("../x", "A B", "", "x" * 80, "..", "a/b"):
        with pytest.raises(KeyError):
            db.dataset_dir(bad)
    d = root / "abc-123"
    (d / "clips").mkdir(parents=True)
    (d / "dataset.json").write_text("{}")
    (d / "clips" / "a.wav").write_bytes(b"RIFF")
    assert db.dataset_file("abc-123", "clips/a.wav").name == "a.wav"
    for bad in ("../dataset.json", "clips/../dataset.json", "other/a.wav", "clips/a.txt",
                "clips/sub/a.wav"):
        with pytest.raises(KeyError):
            db.dataset_file("abc-123", bad)


def _fake_caps(asr_model=None):
    return {"separation": {"available": False}, "transcription": {"available": False},
            "speaker": {"method": "mfcc-stats"}}


def test_end_to_end_build_without_models(root, tmp_path, monkeypatch):
    sr = 44100
    parts = []
    for i in range(4):
        parts += [_voice(sr, 2.5, 140 + 3 * i, seed=i), np.zeros(int(sr * 0.8), "float32")]
    src = tmp_path / "talk.wav"
    sf.write(src, np.concatenate(parts), sr)
    monkeypatch.setattr(db, "capabilities", _fake_caps)
    spans = [(i * 3.3, i * 3.3 + 2.5) for i in range(4)]
    monkeypatch.setattr(db, "vad_segments", lambda x16, min_s, max_s: spans)
    opts = db.BuildOptions(name="Test voice", separate=True, transcribe=True, free_gpu=False,
                           device="cpu", min_quality=0.0)
    job = db.start_build([src], opts)
    for _ in range(300):
        if job.status not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert job.status == "succeeded", job.log
    d = root / job.dataset_id
    summary = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    assert summary["stages"]["separation"] == "skipped"
    assert summary["stages"]["transcription"] == "skipped"
    assert any("separation skipped" in w.lower() for w in summary["warnings"])
    assert summary["stats"]["segments"] == 4 and summary["stats"]["accepted"] == 4
    rows = [json.loads(line) for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["accepted"] and r["file"] for r in rows)
    clips = sorted((d / "clips").glob("*.wav"))
    assert len(clips) == 4 and sf.info(clips[0]).samplerate == sr
    assert len((d / "metadata.csv").read_text(encoding="utf-8").splitlines()) == 4
    assert db.dataset_clips_dir(job.dataset_id) == d / "clips"
    assert db.list_datasets()[0]["id"] == job.dataset_id
    assert not (d / ".work").exists()
    pub = job.public()
    assert pub["status"] == "succeeded" and "cancel" not in pub and "log_tail" in pub


def test_api_validates_and_starts(root, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import voice_lab

    app = FastAPI()
    app.include_router(voice_lab.router)
    client = TestClient(app)
    seen = {}

    def fake_start(sources, opts, reference=None, runner=None):
        seen.update(sources=sources, opts=opts, reference=reference)
        return db.BuildJob(id="j1", dataset_id="x-1", name=opts.name)

    monkeypatch.setattr(db, "start_build", fake_start)
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(1600, "float32"), 16000)
    r = client.post("/voice-lab/datasets", data={"name": "Me", "language": "KN"},
                    files=[("files", ("a.wav", wav.read_bytes(), "audio/wav")),
                           ("reference", ("r.wav", wav.read_bytes(), "audio/wav"))])
    assert r.status_code == 200, r.text
    assert seen["opts"].language == "kn" and seen["reference"].name == "reference.wav"
    r = client.post("/voice-lab/datasets", files=[("files", ("a.exe", b"MZ", "x/y"))])
    assert r.status_code == 415
    r = client.post("/voice-lab/datasets", data={"min_clip_s": "5", "max_clip_s": "2"},
                    files=[("files", ("a.wav", wav.read_bytes(), "audio/wav"))])
    assert r.status_code == 400
    assert client.get("/voice-lab/datasets/nope-nope").status_code == 404
    assert client.get("/voice-lab/capabilities").json()["vad"]["model"].startswith("silero")
