"""Audio Edit service: sessions, spans, windows, jobs and render pipelines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from services import audio_edit as ae
from services import audio_edit_dsp as dsp

SR = ae.SAMPLE_RATE


def _fake_decode(src: Path, dst: Path, channels: int) -> float:
    data, sr = sf.read(str(src), always_2d=True, dtype="float32")
    audio = dsp.match_channels(data.T, channels)
    sf.write(str(dst), audio.T, sr, subtype="FLOAT")
    return audio.shape[1] / sr


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_AUDIO_EDIT_DIR", str(tmp_path / "audio_edit"))
    monkeypatch.setattr(ae, "decode_to_wav", _fake_decode)
    monkeypatch.setattr(ae, "RUN_INLINE", True)
    ae._JOBS.clear()
    yield tmp_path
    ae._JOBS.clear()


def _upload(tmp_path: Path, seconds: float = 4.0, channels: int = 1, name: str = "take.wav") -> Path:
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((int(seconds * SR), channels)) * 0.1).astype(np.float32)
    path = tmp_path / name
    sf.write(str(path), data, SR, subtype="FLOAT")
    return path


WORDS = [
    {"i": 0, "word": "the", "start": 0.10, "end": 0.30},
    {"i": 1, "word": "quick", "start": 0.35, "end": 0.70},
    {"i": 2, "word": "brown", "start": 0.80, "end": 1.20},
    {"i": 3, "word": "fox", "start": 1.30, "end": 1.60},
    {"i": 4, "word": "jumps", "start": 1.70, "end": 2.10},
    {"i": 5, "word": "high", "start": 2.20, "end": 2.60},
]


def _ready(session_id: str) -> dict:
    return ae.update_session(
        session_id, stage="ready",
        transcript={"language": "en", "text": " ".join(w["word"] for w in WORDS), "words": WORDS},
    )


def test_create_speech_session_writes_source_and_json(store):
    session = ae.create_session("speech", "", _upload(store), "Interview Take.wav")
    d = ae.session_dir(session["id"])
    assert (d / "source.wav").is_file()
    assert session["stage"] == "uploaded" and session["mode"] == "speech"
    assert session["name"] == "Interview Take"
    assert session["duration_s"] == pytest.approx(4.0, abs=1e-3)
    assert session["channels"] == 1 and session["renders"] == []
    assert json.loads((d / "session.json").read_text(encoding="utf-8"))["id"] == session["id"]


def test_create_song_session_is_stereo(store):
    session = ae.create_session("song", "Demo", _upload(store, channels=2), "demo.wav")
    audio, sr = ae.read_audio(ae.session_dir(session["id"]) / "source.wav")
    assert audio.shape[0] == 2 and sr == SR and session["name"] == "Demo"


def test_create_rejects_bad_mode_and_too_long(store, monkeypatch):
    with pytest.raises(ae.AudioEditError) as bad:
        ae.create_session("podcast", "", _upload(store), "a.wav")
    assert bad.value.code == "invalid_mode" and bad.value.status == 422
    monkeypatch.setitem(ae.MAX_DURATION_S, "speech", 1.0)
    with pytest.raises(ae.AudioEditError) as long:
        ae.create_session("speech", "", _upload(store), "a.wav")
    assert long.value.code == "too_long" and long.value.status == 400
    assert list(ae.root().iterdir()) == []


def test_ids_are_validated(store):
    for bad in ("../etc", "ABC", "0" * 31, ""):
        with pytest.raises(ae.AudioEditError) as err:
            ae.session_dir(bad)
        assert err.value.status == 404


def test_list_and_delete_sessions(store):
    first = ae.create_session("speech", "one", _upload(store), "a.wav")
    second = ae.create_session("speech", "two", _upload(store), "b.wav")
    ae.update_session(first["id"], created_at=1.0)
    assert [s["name"] for s in ae.list_sessions()] == ["two", "one"]
    ae.delete_session(first["id"])
    assert [s["id"] for s in ae.list_sessions()] == [second["id"]]
    with pytest.raises(ae.AudioEditError):
        ae.load_session(first["id"])


def test_delete_refuses_busy_session(store):
    session = ae.create_session("speech", "", _upload(store), "a.wav")
    job = ae.EditJob(id="j" * 32, session_id=session["id"], kind="transcribe", status="running")
    ae._JOBS[job.id] = job
    with pytest.raises(ae.AudioEditError) as err:
        ae.delete_session(session["id"])
    assert err.value.code == "busy" and err.value.status == 409


def test_interrupted_stage_is_swept(store):
    session = ae.create_session("song", "", _upload(store, channels=2), "a.wav")
    ae.update_session(session["id"], stage="transcribing")
    swept = ae.load_session(session["id"])
    assert swept["stage"] == "separated" and swept["error"] == "interrupted"


def _span(sid="s1", start=0.35, end=0.70, original="quick", replacement="slow"):
    return {"id": sid, "start_s": start, "end_s": end, "original_text": original, "replacement_text": replacement}


def test_validate_spans_sorts_and_normalises():
    spans = ae.validate_spans([_span("b", 1.3, 1.6, "fox", "cat"), _span("a")], 4.0)
    assert [s["id"] for s in spans] == ["a", "b"]
    assert spans[0] == {"id": "a", "start_s": 0.35, "end_s": 0.7, "original_text": "quick", "replacement_text": "slow"}
    assert ae.validate_spans([_span(replacement="")], 4.0)[0]["replacement_text"] == ""


@pytest.mark.parametrize("raw", [
    [_span(), _span("s2", 0.5, 0.9)],                       # overlap
    [_span(start=3.9, end=4.5)],                            # past the end
    [_span(start=0.7, end=0.35)],                           # reversed
    [_span(replacement="quick")],                           # unchanged
    [_span(), _span()],                                     # duplicate id
    [_span(sid="bad id!")],                                 # id charset
    [_span(replacement="x" * 501)],                         # too long
    [_span(f"s{i}", i * 0.05, i * 0.05 + 0.01) for i in range(51)],  # too many
    "nope",
])
def test_validate_spans_rejects(raw):
    with pytest.raises(ae.AudioEditError) as err:
        ae.validate_spans(raw, 4.0)
    assert err.value.code == "invalid_spans" and err.value.status == 400


def test_set_spans_requires_ready(store):
    session = ae.create_session("speech", "", _upload(store), "a.wav")
    with pytest.raises(ae.AudioEditError) as err:
        ae.set_spans(session["id"], [_span()])
    assert err.value.code == "not_ready" and err.value.status == 409
    _ready(session["id"])
    assert ae.set_spans(session["id"], [_span()])["spans"][0]["replacement_text"] == "slow"


def test_words_from_asr_reads_segments_and_top_level():
    segs = {"segments": [{"words": [{"word": " Hi", "start": 0.0, "end": 0.2},
                                    {"word": "there", "start": 0.25}]},
                         {"words": [{"word": "you", "start": 0.5, "end": 0.7, "score": 0.9}]}]}
    assert ae.words_from_asr(segs) == [
        {"i": 0, "word": "Hi", "start": 0.0, "end": 0.2},
        {"i": 1, "word": "you", "start": 0.5, "end": 0.7},
    ]
    top = {"words": [{"word": "a", "start": 1, "end": 2}]}
    assert ae.words_from_asr(top)[0] == {"i": 0, "word": "a", "start": 1.0, "end": 2.0}
    with pytest.raises(ae.AudioEditError) as err:
        ae.words_from_asr({"segments": [{"text": "no words"}]})
    assert err.value.code == "no_word_timestamps" and err.value.status == 422


def test_span_window_adds_context_words():
    win = ae.span_window(_span("s", 0.80, 1.20, "brown", "red"), WORDS, 4.0, 0.6)
    assert (win.start_s, win.end_s) == (0.35, 1.6)
    assert win.raw_text == "quick brown fox"
    assert win.target_text == "quick red fox"


def test_span_window_respects_bounds_and_edges():
    spans = [_span("a", 0.35, 0.70, "quick", "slow"), _span("b", 1.30, 1.60, "fox", "cat")]
    lo, hi = ae.window_bounds(spans, 1, 4.0)
    assert (lo, hi) == (pytest.approx(1.0), 4.0)
    win = ae.span_window(spans[1], WORDS, 4.0, 2.0, lo, hi)
    assert win.start_s == pytest.approx(1.05) and win.raw_text == "fox jumps high"
    first = ae.span_window(_span("c", 0.10, 0.30, "the", "a"), WORDS, 4.0, 2.0)
    assert first.raw_text.startswith("the")
    lonely = ae.span_window(_span("d", 3.0, 3.5, "", "hello"), WORDS, 4.0, 2.0, 2.7, 4.0)
    assert (lonely.start_s, lonely.end_s) == (2.75, 3.75) and lonely.target_text == "hello"


def test_get_job_unknown_is_404(store):
    with pytest.raises(ae.AudioEditError) as err:
        ae.get_job("0" * 32)
    assert err.value.status == 404


# ── Task 7: licences and commercial use ─────────────────────────────────

@pytest.fixture
def acks(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(ae, "_read_acks", lambda: dict(saved))
    monkeypatch.setattr(ae, "_write_acks", lambda value: (saved.clear(), saved.update(value)))
    return saved


def test_ack_and_revoke(acks):
    assert ae.licence_acks() == {}
    with pytest.raises(ae.AudioEditError) as missing:
        ae.require_licence("f5tts")
    assert missing.value.code == "licence_ack_required" and missing.value.status == 403
    out = ae.acknowledge_licence("f5tts")
    assert out["required"] is True and acks["f5tts"] == out["accepted_at"]
    ae.require_licence("f5tts")
    assert ae.revoke_licence("f5tts")["accepted_at"] is None and "f5tts" not in acks
    assert ae.acknowledge_licence("ace_step") == {"engine": "ace_step", "required": False, "accepted_at": None}
    ae.require_licence("ace_step")
    with pytest.raises(ae.AudioEditError) as bad:
        ae.acknowledge_licence("nope")
    assert bad.value.code == "invalid_engine" and bad.value.status == 422


def test_licence_acks_ignore_junk(acks):
    acks.update({"f5tts": 5, "ace_step": 3, "vevo2": "yes", "other": 1, "yingmusic": True})
    assert ae.licence_acks() == {"f5tts": 5.0}


def test_commercial_use_is_the_and_of_every_record():
    cleared = {"commercial_use": True, "review_status": "cleared"}
    records = {"ACE-Step/Ace-Step1.5": dict(cleared), ae.HTDEMUCS_RECORD: dict(cleared),
               ae.RVC_RECORD: {"commercial_use": False, "review_status": "unreviewed"}}
    assert ae.render_commercial_use("ace_step", rvc=False, demucs=True, records=records) is True
    assert ae.render_commercial_use("ace_step", rvc=True, demucs=True, records=records) is False
    records["ACE-Step/Ace-Step1.5"]["review_status"] = "unreviewed"
    assert ae.render_commercial_use("ace_step", rvc=False, demucs=True, records=records) is False
    assert ae.render_commercial_use("f5tts", rvc=False, demucs=False, records={}) is False


def test_real_registry_badges_every_engine_non_commercial():
    records = ae._registry_records()
    assert ae.HTDEMUCS_RECORD in records and ae.RVC_RECORD in records
    for engine in ae.ENGINES:
        assert ae.render_commercial_use(engine, rvc=False, demucs=False, records=records) is False


def test_engines_overview(acks, monkeypatch):
    from services import audio_edit_engines as aee

    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": True, "install_id": e, "needs_gpu": False, "available": True, "reason": None})
    monkeypatch.setattr(aee, "has_gpu", lambda: False)
    monkeypatch.setattr(ae, "separation_status", lambda: {
        "available": False, "weights": None, "missing": "adefossez/HTDemucs (~84 MB)"})
    acks["vevo2"] = 9.0
    data = ae.engines_overview()
    by_id = {e["id"]: e for e in data["engines"]}
    assert list(by_id) == list(ae.ENGINES)
    assert by_id["f5tts"]["mode"] == "speech" and by_id["vevo2"]["mode"] == "song"
    assert by_id["f5tts"]["display_name"] == "F5-TTS" and by_id["f5tts"]["licence"]
    assert by_id["vevo2"]["acked_at"] == 9.0 and by_id["vevo2"]["needs_ack"] is True
    assert by_id["ace_step"]["needs_ack"] is False and by_id["ace_step"]["acked_at"] is None
    assert all(e["commercial_use"] is False and e["available"] for e in data["engines"])
    assert data["separation"]["available"] is False and data["gpu"] is False


# ── Task 8: runner, separation, transcription ───────────────────────────

@pytest.fixture(autouse=True)
def _no_vram_calls(monkeypatch):
    monkeypatch.setattr(ae, "_free_vram", lambda: None)


def test_transcribe_speech_session(store, monkeypatch):
    calls = []
    monkeypatch.setattr(ae, "_release_gpu", lambda: calls.append("release"))
    monkeypatch.setattr(ae, "_asr_transcribe", lambda path: calls.append(Path(path).name) or {
        "language": "en", "segments": [{"words": [{"word": "hi", "start": 0.1, "end": 0.4}]}]})
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    job = ae.start_transcription(sid)
    assert job.status == "succeeded" and job.kind == "transcribe" and job.progress == 1.0
    loaded = ae.load_session(sid)
    assert loaded["stage"] == "ready" and loaded["error"] is None
    assert loaded["transcript"] == {"language": "en", "text": "hi",
                                    "words": [{"i": 0, "word": "hi", "start": 0.1, "end": 0.4}]}
    assert calls == ["release", "source.wav"]
    assert (ae.session_dir(sid) / "transcript.json").is_file()


def test_transcribe_without_word_timings_reverts_stage(store, monkeypatch):
    monkeypatch.setattr(ae, "_release_gpu", lambda: None)
    monkeypatch.setattr(ae, "_asr_transcribe", lambda path: {"segments": [{"text": "hello"}]})
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    job = ae.start_transcription(sid)
    assert job.status == "failed" and job.error == "no_word_timestamps"
    loaded = ae.load_session(sid)
    assert loaded["stage"] == "uploaded" and loaded["error"] == "no_word_timestamps"


def test_song_separation_then_transcription(store, monkeypatch):
    monkeypatch.setattr(ae, "_release_gpu", lambda: None)

    def fake_separate(path):
        audio, _ = ae.read_audio(path)
        return audio.mean(axis=0, keepdims=True) * 0.5, audio * 0.5

    seen = []
    monkeypatch.setattr(ae, "separate_stems", fake_separate)
    monkeypatch.setattr(ae, "_asr_transcribe", lambda path: seen.append(Path(path).name) or {"words": WORDS})
    sid = ae.create_session("song", "", _upload(store, channels=2), "s.wav")["id"]
    with pytest.raises(ae.AudioEditError) as early:
        ae.start_transcription(sid)
    assert early.value.code == "not_ready" and early.value.status == 409
    assert ae.start_separation(sid).status == "succeeded"
    assert ae.load_session(sid)["stage"] == "separated"
    assert ae.read_audio(ae.audio_path(sid, "vocals"))[0].shape[0] == 1
    assert ae.read_audio(ae.audio_path(sid, "instrumental"))[0].shape[0] == 2
    assert ae.start_transcription(sid).status == "succeeded"
    assert seen == ["vocals.wav"] and ae.load_session(sid)["stage"] == "ready"


def test_separation_is_for_uploaded_songs_only(store):
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    with pytest.raises(ae.AudioEditError) as err:
        ae.start_separation(sid)
    assert err.value.code == "not_ready" and err.value.status == 409


def test_separation_unavailable_fails_job(store, monkeypatch):
    monkeypatch.setattr(ae, "separation_status", lambda: {
        "available": False, "weights": None, "missing": "adefossez/HTDemucs (~84 MB)"})
    sid = ae.create_session("song", "", _upload(store, channels=2), "s.wav")["id"]
    job = ae.start_separation(sid)
    assert job.status == "failed" and job.error == "separation_unavailable"
    assert "HTDemucs" in job.message
    assert ae.load_session(sid)["stage"] == "uploaded"


def test_busy_session_refuses_a_second_job(store):
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    ae._new_job(sid, "transcribe").status = "running"
    with pytest.raises(ae.AudioEditError) as err:
        ae.start_transcription(sid)
    assert err.value.code == "busy" and err.value.status == 409


def test_cancel_kills_the_engine_of_live_jobs_only(store):
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    killed = []
    live = ae._new_job(sid, "render", render_id="r" * 32)
    live.status, live.engine = "running", lambda: killed.append("live")
    assert ae.cancel_job(live.id).cancel.is_set() and killed == ["live"]
    done = ae._new_job(sid, "transcribe")
    done.status, done.engine = "succeeded", lambda: killed.append("done")
    assert not ae.cancel_job(done.id).cancel.is_set() and killed == ["live"]


def test_job_cancelled_while_waiting_for_the_gpu_slot_never_runs(store):
    sid = ae.create_session("speech", "", _upload(store), "a.wav")["id"]
    assert ae._RENDER_SLOT.acquire(timeout=1)
    try:
        job = ae._new_job(sid, "transcribe")
        job.cancel.set()
        ae._run(job, lambda j: pytest.fail("a cancelled job must not run"))
    finally:
        ae._RENDER_SLOT.release()
    assert job.status == "cancelled"
    loaded = ae.load_session(sid)
    assert loaded["stage"] == "uploaded" and loaded["error"] == "cancelled"
