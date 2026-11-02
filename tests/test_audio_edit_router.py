"""HTTP surface of /audio-edit with fake decode, ASR, engines and Demucs."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services import audio_edit as ae
from services import audio_edit_dsp as dsp

SR = ae.SAMPLE_RATE
WORDS = [
    {"word": "the", "start": 0.10, "end": 0.30},
    {"word": "quick", "start": 0.35, "end": 0.70},
    {"word": "brown", "start": 0.80, "end": 1.20},
]


def _fake_decode(src: Path, dst: Path, channels: int) -> float:
    data, sr = sf.read(str(src), always_2d=True, dtype="float32")
    audio = dsp.match_channels(data.T, channels)
    sf.write(str(dst), audio.T, sr, subtype="FLOAT")
    return audio.shape[1] / sr


def _wav(seconds: float = 2.0, channels: int = 1) -> bytes:
    rng = np.random.default_rng(2)
    buf = io.BytesIO()
    sf.write(buf, (rng.standard_normal((int(seconds * SR), channels)) * 0.1).astype(np.float32),
             SR, format="WAV", subtype="FLOAT")
    return buf.getvalue()


class _Editor:
    def edit(self, payload, on_progress=None):
        seconds = sf.info(payload["audio_path"]).frames / SR
        (p0, p1), dur = payload["parts"][0], payload["fix_duration"][0]
        n = int(round((seconds - (p1 - p0) + dur) * 24000))
        sf.write(payload["out_path"], np.full(n, 0.05, np.float32), 24000, subtype="FLOAT")
        return {"path": payload["out_path"], "sample_rate": 24000, "regions": [[p0, p0 + dur]]}

    def kill(self):
        pass

    def shutdown(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    from api.dependencies import require_admin, require_desktop
    from api.routers import audio_edit as router_module
    from services import audio_edit_engines as aee
    from services import watermark

    monkeypatch.setenv("OMNIVOICE_AUDIO_EDIT_DIR", str(tmp_path / "audio_edit"))
    monkeypatch.setattr(ae, "decode_to_wav", _fake_decode)
    monkeypatch.setattr(ae, "RUN_INLINE", True)
    monkeypatch.setattr(ae, "_free_vram", lambda: None)
    monkeypatch.setattr(ae, "_release_gpu", lambda: None)
    monkeypatch.setattr(ae, "_registry_records", lambda: {})
    monkeypatch.setattr(ae, "_asr_transcribe", lambda path: {"language": "en", "words": WORDS})
    monkeypatch.setattr(ae, "_edit_engine", lambda engine_id: _Editor())
    monkeypatch.setattr(ae, "separation_status", lambda: {"available": True, "weights": "w", "missing": None})
    monkeypatch.setattr(aee, "engine_status", lambda e: {
        "installed": True, "install_id": e, "needs_gpu": False, "available": True, "reason": None})
    monkeypatch.setattr(aee, "has_gpu", lambda: True)
    monkeypatch.setattr(watermark, "mark_synthetic", lambda waveform, sample_rate, *, context, force=False: waveform)
    saved: dict = {}
    monkeypatch.setattr(ae, "_read_acks", lambda: dict(saved))
    monkeypatch.setattr(ae, "_write_acks", lambda value: (saved.clear(), saved.update(value)))
    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[require_admin] = lambda: None
    app.dependency_overrides[require_desktop] = lambda: None
    ae._JOBS.clear()
    yield TestClient(app)
    ae._JOBS.clear()


def _create(client, mode="speech", channels=1):
    res = client.post("/audio-edit/sessions", data={"mode": mode, "name": "Take"},
                      files={"file": ("take.wav", _wav(channels=channels), "audio/wav")})
    assert res.status_code == 200, res.text
    return res.json()


def _ready_speech(client) -> str:
    sid = _create(client)["session"]["id"]
    assert client.post(f"/audio-edit/sessions/{sid}/transcribe").json()["job"]["status"] == "succeeded"
    span = {"id": "s1", "start_s": 0.35, "end_s": 0.70, "original_text": "quick", "replacement_text": "slow"}
    assert client.put(f"/audio-edit/sessions/{sid}/spans", json={"spans": [span]}).status_code == 200
    return sid


def test_router_is_admin_and_desktop_only():
    from api.dependencies import require_admin, require_desktop
    from api.routers import audio_edit as router_module

    deps = {d.dependency for d in router_module.router.dependencies}
    assert {require_admin, require_desktop} <= deps
    assert len(router_module.router.routes) == 16


def test_engines_listing(client):
    body = client.get("/audio-edit/engines").json()
    assert [e["id"] for e in body["engines"]] == ["f5tts", "ace_step", "yingmusic", "vevo2"]
    assert body["separation"]["available"] is True and body["gpu"] is True


def test_speech_session_flow(client):
    created = _create(client)
    assert created["job"] is None and created["session"]["stage"] == "uploaded"
    sid = _ready_speech(client)
    session = client.get(f"/audio-edit/sessions/{sid}").json()
    assert session["stage"] == "ready" and session["spans"][0]["replacement_text"] == "slow"
    bad = client.put(f"/audio-edit/sessions/{sid}/spans", json={"spans": [
        {"id": "a", "start_s": 0.1, "end_s": 0.5, "original_text": "x", "replacement_text": "y"},
        {"id": "b", "start_s": 0.4, "end_s": 0.9, "original_text": "x", "replacement_text": "y"}]})
    assert bad.status_code == 400 and bad.json()["detail"]["code"] == "invalid_spans"
    audio = client.get(f"/audio-edit/sessions/{sid}/audio/source")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    missing = client.get(f"/audio-edit/sessions/{sid}/audio/vocals")
    assert missing.status_code == 404 and missing.json()["detail"]["code"] == "not_found"
    listing = client.get("/audio-edit/sessions", params={"mode": "speech"}).json()
    assert listing["total"] == 2 and listing["items"][0]["render_count"] == 0
    assert listing["items"][0]["size_bytes"] > 0
    assert client.get("/audio-edit/sessions", params={"mode": "song"}).json()["total"] == 0


def test_upload_rejects_bad_type_and_mode(client):
    exe = client.post("/audio-edit/sessions", data={"mode": "speech"},
                      files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert exe.status_code == 415 and exe.json()["detail"]["code"] == "invalid_audio"
    mode = client.post("/audio-edit/sessions", data={"mode": "podcast"},
                       files={"file": ("a.wav", _wav(), "audio/wav")})
    assert mode.status_code == 422 and mode.json()["detail"]["code"] == "invalid_mode"


def test_song_upload_starts_separation(client, monkeypatch):
    monkeypatch.setattr(ae, "separate_stems", lambda path: (
        ae.read_audio(path)[0].mean(axis=0, keepdims=True), ae.read_audio(path)[0]))
    body = _create(client, mode="song", channels=2)
    assert body["job"]["kind"] == "separate" and body["job"]["status"] == "succeeded"
    assert body["session"]["stage"] == "separated"


def test_render_flow_licence_and_jobs(client):
    sid = _ready_speech(client)
    denied = client.post(f"/audio-edit/sessions/{sid}/render", json={"engine": "f5tts"})
    assert denied.status_code == 403 and denied.json()["detail"]["code"] == "licence_ack_required"
    assert client.post("/audio-edit/licence/ack", json={"engine": "f5tts", "accepted": False}).status_code == 422
    ack = client.post("/audio-edit/licence/ack", json={"engine": "f5tts", "accepted": True}).json()
    assert ack["required"] is True and ack["accepted_at"]
    res = client.post(f"/audio-edit/sessions/{sid}/render", json={"engine": "f5tts", "seed": 1})
    assert res.status_code == 200, res.text
    job, render = res.json()["job"], res.json()["render"]
    assert job["status"] == "succeeded" and render["status"] == "succeeded"
    assert client.get(f"/audio-edit/sessions/{sid}/renders/{render['id']}").json()["id"] == render["id"]
    audio = client.get(f"/audio-edit/sessions/{sid}/renders/{render['id']}/audio")
    assert audio.status_code == 200 and "edited" in audio.headers["content-disposition"]
    assert client.get(f"/audio-edit/jobs/{job['id']}").json()["status"] == "succeeded"
    assert client.post(f"/audio-edit/jobs/{job['id']}/cancel").json()["status"] == "succeeded"
    assert client.get("/audio-edit/jobs/" + "0" * 32).status_code == 404
    assert client.delete("/audio-edit/licence/ack/f5tts").json()["accepted_at"] is None


def test_delete_session(client):
    sid = _create(client)["session"]["id"]
    assert client.delete(f"/audio-edit/sessions/{sid}").json() == {"ok": True}
    assert client.get(f"/audio-edit/sessions/{sid}").status_code == 404
