"""Audio Edit render pipelines with fake engines (no GPU, no sidecars)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from services import audio_edit as ae
from services import audio_edit_dsp as dsp

SR = ae.SAMPLE_RATE

WORDS = [
    {"i": 0, "word": "the", "start": 0.10, "end": 0.30},
    {"i": 1, "word": "quick", "start": 0.35, "end": 0.70},
    {"i": 2, "word": "brown", "start": 0.80, "end": 1.20},
    {"i": 3, "word": "fox", "start": 1.30, "end": 1.60},
    {"i": 4, "word": "jumps", "start": 1.70, "end": 2.10},
    {"i": 5, "word": "high", "start": 2.20, "end": 2.60},
]


def _fake_decode(src: Path, dst: Path, channels: int) -> float:
    data, sr = sf.read(str(src), always_2d=True, dtype="float32")
    audio = dsp.match_channels(data.T, channels)
    sf.write(str(dst), audio.T, sr, subtype="FLOAT")
    return audio.shape[1] / sr


def _upload(tmp_path: Path, channels: int = 1) -> Path:
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((4 * SR, channels)) * 0.1).astype(np.float32)
    path = tmp_path / f"take{channels}.wav"
    sf.write(str(path), data, SR, subtype="FLOAT")
    return path


def _span(sid="s1", start=0.35, end=0.70, original="quick", replacement="slow"):
    return {"id": sid, "start_s": start, "end_s": end, "original_text": original, "replacement_text": replacement}


class FakeEditor:
    """Stands in for an EditSidecar: writes a constant tone of the right length."""

    def __init__(self, sr: int = 24000):
        self.sr, self.fail, self.on_edit = sr, False, None
        self.payloads: list[dict] = []
        self.killed = self.stopped = 0

    def edit(self, payload, on_progress=None):
        from services.audio_edit_engines import EngineError

        self.payloads.append(payload)
        if self.on_edit:
            self.on_edit(payload)
        if self.fail:
            raise EngineError("CUDA out of memory")
        if on_progress:
            on_progress(0.5)
        frames = sf.info(payload["audio_path"]).frames
        seconds = frames / SR
        reply = {"op": "audio", "path": payload["out_path"], "sample_rate": self.sr}
        if "parts" in payload:
            (p0, p1), dur = payload["parts"][0], payload["fix_duration"][0]
            seconds = seconds - (p1 - p0) + dur
            reply["regions"] = [[p0, round(p0 + dur, 4)]]
        sf.write(payload["out_path"], np.full(int(round(seconds * self.sr)), 0.05, np.float32),
                 self.sr, subtype="FLOAT")
        return reply

    def kill(self):
        self.killed += 1

    def shutdown(self):
        self.stopped += 1


@pytest.fixture
def env(tmp_path, monkeypatch):
    from services import audio_edit_engines as aee
    from services import watermark

    monkeypatch.setenv("OMNIVOICE_AUDIO_EDIT_DIR", str(tmp_path / "audio_edit"))
    monkeypatch.setattr(ae, "decode_to_wav", _fake_decode)
    monkeypatch.setattr(ae, "RUN_INLINE", True)
    monkeypatch.setattr(ae, "_free_vram", lambda: None)
    acks: dict = {}
    monkeypatch.setattr(ae, "_read_acks", lambda: dict(acks))
    monkeypatch.setattr(ae, "_write_acks", lambda value: (acks.clear(), acks.update(value)))
    monkeypatch.setattr(ae, "_registry_records", lambda: {})
    events: list = []
    editor = FakeEditor()
    monkeypatch.setattr(ae, "_release_gpu", lambda: events.append("release"))
    monkeypatch.setattr(ae, "_edit_engine", lambda engine_id: events.append(f"engine:{engine_id}") or editor)
    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": True, "install_id": e, "needs_gpu": False, "available": True, "reason": None})
    marks: list = []

    def fake_mark(waveform, sample_rate, *, context, force=False):
        marks.append((tuple(waveform.shape), sample_rate, context))
        return waveform

    monkeypatch.setattr(watermark, "mark_synthetic", fake_mark)
    ae._JOBS.clear()
    yield SimpleNamespace(tmp=tmp_path, editor=editor, events=events, marks=marks, acks=acks)
    ae._JOBS.clear()


def _speech(env, spans) -> str:
    sid = ae.create_session("speech", "", _upload(env.tmp), "a.wav")["id"]
    ae.update_session(sid, stage="ready", transcript={
        "language": "en", "text": " ".join(w["word"] for w in WORDS), "words": WORDS})
    ae.set_spans(sid, spans)
    return sid


def _samples(seconds: float) -> int:
    return int(round(seconds * SR))


def test_render_request_checks(env, monkeypatch):
    from services import audio_edit_engines as aee

    sid = _speech(env, [_span()])
    with pytest.raises(ae.AudioEditError) as ack:
        ae.start_render(sid, {"engine": "f5tts"})
    assert ack.value.code == "licence_ack_required" and ack.value.status == 403
    ae.acknowledge_licence("f5tts")
    for raw, code in (({"engine": "ace_step"}, "invalid_engine"),
                      ({"engine": "f5tts", "crossfade_ms": 2}, "invalid_request"),
                      ({"engine": "f5tts", "rvc_pitch": 13}, "invalid_request"),
                      ({"engine": "f5tts", "seed": "x"}, "invalid_request"),
                      ("nope", "invalid_request")):
        with pytest.raises(ae.AudioEditError) as err:
            ae.start_render(sid, raw)
        assert (err.value.code, err.value.status) == (code, 422)
    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": False, "install_id": "f5tts", "needs_gpu": False, "available": False,
        "reason": "Install F5-TTS to use it."})
    with pytest.raises(ae.AudioEditError) as missing:
        ae.start_render(sid, {"engine": "f5tts"})
    assert missing.value.code == "engine_not_installed" and missing.value.status == 409
    params = ae.validate_render_request("song", {"engine": "ace_step"})
    assert params["crossfade_ms"] == 30 and params["vocal_gain_db"] == 0.0
    assert params["ace_step"] == {"strength": 0.7, "padding_s": 2.0, "caption": ""}
    assert 0 <= params["seed"] < 2 ** 31


def test_speech_render_splices_only_inside_the_spans(env):
    ae.acknowledge_licence("f5tts")
    sid = _speech(env, [_span(), _span("s2", 1.30, 1.60, "fox", "")])
    job, render = ae.start_render(sid, {"engine": "f5tts", "crossfade_ms": 20, "seed": 7})
    assert job.status == "succeeded" and render["status"] == "succeeded"
    assert render["commercial_use"] is False and render["engine"] == "f5tts"
    assert render["output_path"] == f"renders/{render['id']}/output.wav"
    assert [s["status"] for s in render["spans"]] == ["done", "done"]

    payload, = env.editor.payloads                      # the deletion needs no engine
    assert payload["raw_text"] == "the quick" and payload["target_text"] == "the slow"
    assert payload["parts"] == [[0.25, 0.6]] and payload["fix_duration"] == [0.28]
    assert payload["seed"] == 7
    assert env.events[:2] == ["release", "engine:f5tts"] and env.editor.stopped == 1
    assert len(env.marks) == 1 and env.marks[0][1:] == (SR, "audio_edit.render")

    path = ae.render_audio_path(sid, render["id"])
    out, sr = ae.read_audio(path)
    source, _ = ae.read_audio(ae.session_dir(sid) / "source.wav")
    x = dsp.ms_to_samples(20, SR)
    a1, b1, a2, b2 = _samples(0.35), _samples(0.70), _samples(1.30), _samples(1.60)
    body = _samples(0.53) - _samples(0.25)
    assert sr == SR
    assert out.shape[1] == source.shape[1] - (b1 - a1) + body - ((b2 - a2) + x)
    assert np.array_equal(out[:, :a1 - x], source[:, :a1 - x])
    tail = source.shape[1] - b2 - x
    assert np.array_equal(out[:, -tail:], source[:, -tail:])
    with sf.SoundFile(str(path)) as f:
        assert "commercial_use=false" in f.comment and f.software.startswith("VoiceStudio Audio Edit")


def test_delete_only_render_needs_no_engine_or_ack(env):
    sid = _speech(env, [_span(replacement="")])
    job, render = ae.start_render(sid, {"engine": "f5tts"})
    assert job.status == "succeeded" and render["status"] == "succeeded"
    assert env.editor.payloads == [] and not any(e.startswith("engine:") for e in env.events)
    assert env.marks == []


def test_engine_failure_fails_the_render_and_keeps_partial_files(env):
    ae.acknowledge_licence("f5tts")
    env.editor.fail = True
    sid = _speech(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "f5tts"})
    assert job.status == "failed" and render["status"] == "failed"
    assert render["error"] == "engine_failed" and "CUDA out of memory" in render["message"]
    assert (ae.render_dir(sid, render["id"]) / "span-0.window.wav").is_file()
    assert env.editor.stopped == 1
    with pytest.raises(ae.AudioEditError):
        ae.render_audio_path(sid, render["id"])


def test_cancel_during_an_edit_kills_the_sidecar(env):
    ae.acknowledge_licence("f5tts")

    def cancel_now(payload):
        job = next(j for j in ae._JOBS.values() if j.kind == "render")
        ae.cancel_job(job.id)

    env.editor.on_edit, env.editor.fail = cancel_now, True   # a killed sidecar errors out
    sid = _speech(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "f5tts"})
    assert job.status == "cancelled" and render["status"] == "cancelled"
    assert env.editor.killed == 1


def test_rvc_pass_converts_each_insert(env, monkeypatch):
    ae.acknowledge_licence("f5tts")
    monkeypatch.setattr(ae, "_rvc_voice_files", lambda voice_id: ("model.pth", None))
    calls = []

    def fake_rvc(src, dst, voice_id, pitch):
        calls.append((Path(src).name, voice_id, pitch))
        audio, _ = ae.read_audio(src)
        ae.write_audio(dst, audio * 0.5, 40000)
        return dst

    monkeypatch.setattr(ae, "_run_rvc", fake_rvc)
    sid = _speech(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "f5tts", "rvc_model_id": "alto", "rvc_pitch": 3})
    assert job.status == "succeeded" and calls == [("span-0.insert.wav", "alto", 3)]
    assert render["params"]["rvc_model_id"] == "alto"
    assert env.events.count("release") >= 3          # before F5-TTS, before and after RVC

    def unknown(voice_id):
        raise ae.AudioEditError("invalid_rvc_model", "That RVC voice is not installed.", 422)

    monkeypatch.setattr(ae, "_rvc_voice_files", unknown)
    with pytest.raises(ae.AudioEditError) as err:
        ae.start_render(sid, {"engine": "f5tts", "rvc_model_id": "gone"})
    assert err.value.code == "invalid_rvc_model"


def test_renders_list_newest_first_and_need_a_ready_session(env):
    ae.acknowledge_licence("f5tts")
    sid = _speech(env, [_span()])
    _, first = ae.start_render(sid, {"engine": "f5tts"})
    _, second = ae.start_render(sid, {"engine": "f5tts"})
    assert [r["id"] for r in ae.load_session(sid)["renders"]] == [second["id"], first["id"]]
    ae.update_session(sid, spans=[])
    with pytest.raises(ae.AudioEditError) as none:
        ae.start_render(sid, {"engine": "f5tts"})
    assert none.value.code == "no_spans" and none.value.status == 400
    ae.update_session(sid, stage="uploaded")
    with pytest.raises(ae.AudioEditError) as early:
        ae.start_render(sid, {"engine": "f5tts"})
    assert early.value.code == "not_ready"


# ── Task 10: song renders ───────────────────────────────────────────────

def _song(env, spans) -> str:
    sid = ae.create_session("song", "", _upload(env.tmp, channels=2), "s.wav")["id"]
    d = ae.session_dir(sid)
    source, _ = ae.read_audio(d / "source.wav")
    ae.write_audio(d / "vocals.wav", source.mean(axis=0, keepdims=True) * 0.6, SR)
    ae.write_audio(d / "instrumental.wav", source * 0.4, SR)
    ae.update_session(sid, stage="ready", transcript={
        "language": "en", "text": " ".join(w["word"] for w in WORDS), "words": WORDS})
    ae.set_spans(sid, spans)
    return sid


class FakeAce:
    def __init__(self):
        self.calls: list[dict] = []
        self.stopped = self.cancelled = 0
        self.error = None

    def repaint(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        audio, sr = ae.read_audio(kwargs["src_path"])
        ae.write_audio(kwargs["out_path"], audio * 0.9, sr)
        return Path(kwargs["out_path"])

    def cancel(self):
        self.cancelled += 1

    def stop(self):
        self.stopped += 1


def test_song_mix_residual_keeps_untouched_samples():
    rng = np.random.default_rng(3)
    source = (rng.standard_normal((2, 1000)) * 0.1).astype(np.float32)
    vocals = (rng.standard_normal((1, 1000)) * 0.1).astype(np.float32)
    instrumental = (source * 0.5).astype(np.float32)
    edited = vocals.copy()
    edited[:, 100:200] = 0.3
    mix = ae.song_mix(source, vocals, edited, instrumental, 0.0)
    assert np.array_equal(mix[:, :100], source[:, :100]) and np.array_equal(mix[:, 200:], source[:, 200:])
    assert np.allclose(mix[:, 100:200], source[:, 100:200] + (0.3 - vocals[:, 100:200]), atol=1e-6)
    assert np.array_equal(ae.song_mix(source, vocals, edited, instrumental, -6.0),
                          dsp.remix(edited, instrumental, -6.0))


def test_singer_render_keeps_length_and_outside_audio(env):
    ae.acknowledge_licence("yingmusic")
    env.editor.sr = 48000
    sid = _song(env, [_span(), _span("s2", 1.30, 1.60, "fox", "")])
    job, render = ae.start_render(sid, {"engine": "yingmusic", "crossfade_ms": 20})
    assert job.status == "succeeded" and render["status"] == "succeeded"
    payload, = env.editor.payloads
    assert "parts" not in payload and payload["target_text"] == "the slow"
    assert env.events[:2] == ["release", "engine:yingmusic"] and env.editor.stopped == 1
    assert len(env.marks) == 1                          # the silent deletion is not synthetic
    out, _ = ae.read_audio(ae.render_audio_path(sid, render["id"]))
    source, _ = ae.read_audio(ae.session_dir(sid) / "source.wav")
    x = dsp.ms_to_samples(20, SR)
    a1, b1, a2, b2 = _samples(0.35), _samples(0.70), _samples(1.30), _samples(1.60)
    assert out.shape == source.shape
    for lo, hi in ((0, a1 - x), (b1 + x, a2 - x), (b2 + x, source.shape[1])):
        assert np.array_equal(out[:, lo:hi], source[:, lo:hi])
    assert (ae.render_dir(sid, render["id"]) / "vocals.edited.wav").is_file()


def test_singer_engine_unavailable_is_409(env, monkeypatch):
    from services import audio_edit_engines as aee

    ae.acknowledge_licence("vevo2")
    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": True, "install_id": "vevo2", "needs_gpu": True, "available": False,
        "reason": "Vevo2 needs a CUDA or Apple GPU."})
    sid = _song(env, [_span()])
    with pytest.raises(ae.AudioEditError) as err:
        ae.start_render(sid, {"engine": "vevo2"})
    assert err.value.code == "engine_not_installed" and "GPU" in err.value.message


def test_ace_step_render_repaints_then_separates(env, monkeypatch):
    fake = FakeAce()
    monkeypatch.setattr(ae, "_ace_client", lambda work_dir: fake)
    separated = []

    def fake_separate(path):
        separated.append(Path(path).name)
        audio, sr = ae.read_audio(path)
        audio = dsp.resample(audio, sr, SR)
        return audio.mean(axis=0, keepdims=True), audio

    monkeypatch.setattr(ae, "separate_stems", fake_separate)
    sid = _song(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "ace_step", "seed": 3, "vocal_gain_db": -3,
                                        "ace_step": {"strength": 0.4, "padding_s": 1.0, "caption": "pop"}})
    assert job.status == "succeeded" and render["status"] == "succeeded"
    call, = fake.calls
    assert (call["start_s"], call["end_s"]) == (0.1, 2.6)      # window inside a 0–3.6 s segment
    assert call["lyrics"] == "the slow brown fox jumps high" and call["caption"] == "pop" and call["strength"] == 0.4
    assert call["seed"] == 3 and call["language"] == "en"
    assert Path(call["src_path"]).name == "span-0.segment.wav"
    assert sf.info(str(call["src_path"])).frames == _samples(3.6)
    assert fake.stopped == 1 and separated == ["span-0.wav"] and len(env.marks) == 1
    out, _ = ae.read_audio(ae.render_audio_path(sid, render["id"]))
    assert out.shape[1] == ae.read_audio(ae.session_dir(sid) / "source.wav")[0].shape[1]


def test_ace_step_unavailable_fails_the_render(env, monkeypatch):
    from services.acestep_client import AceStepUnavailable

    fake = FakeAce()
    fake.error = AceStepUnavailable("ACE-Step did not answer within 60 s.")
    monkeypatch.setattr(ae, "_ace_client", lambda work_dir: fake)
    sid = _song(env, [_span()])
    job, render = ae.start_render(sid, {"engine": "ace_step"})
    assert job.status == "failed" and render["error"] == "ace_step_unavailable"
    assert "60 s" in render["message"] and fake.stopped == 1
