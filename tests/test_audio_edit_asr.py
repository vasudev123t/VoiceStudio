"""Audio Edit speech recognition: engine/language choice, keys, cloud adapters."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services import audio_edit as ae
from services import audio_edit_asr as asr
from services import audio_edit_dsp as dsp

SR = ae.SAMPLE_RATE


@pytest.fixture
def secrets(monkeypatch):
    store: dict[str, str] = {}
    monkeypatch.setattr(asr, "_store_get", lambda name: store.get(name))
    monkeypatch.setattr(asr, "_store_set", lambda name, value: store.__setitem__(name, value) if value
                        else store.pop(name, None))
    for p in asr.PROVIDERS.values():
        for f in p.fields:
            for env in f.env:
                monkeypatch.delenv(env, raising=False)
    return store


def _fake_decode(src: Path, dst: Path, channels: int) -> float:
    data, sr = sf.read(str(src), always_2d=True, dtype="float32")
    audio = dsp.match_channels(data.T, channels)
    sf.write(str(dst), audio.T, sr, subtype="FLOAT")
    return audio.shape[1] / sr


@pytest.fixture
def client(tmp_path, monkeypatch, secrets):
    from api.dependencies import require_admin, require_desktop
    from api.routers import audio_edit as router_module

    monkeypatch.setenv("OMNIVOICE_AUDIO_EDIT_DIR", str(tmp_path / "audio_edit"))
    monkeypatch.setattr(ae, "decode_to_wav", _fake_decode)
    monkeypatch.setattr(ae, "RUN_INLINE", True)
    monkeypatch.setattr(ae, "_free_vram", lambda: None)
    monkeypatch.setattr(ae, "_release_gpu", lambda: None)
    monkeypatch.setattr(asr, "_hf_installed", lambda repo: None)
    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[require_admin] = lambda: None
    app.dependency_overrides[require_desktop] = lambda: None
    ae._JOBS.clear()
    yield TestClient(app)
    ae._JOBS.clear()


def _wav_upload(seconds: float = 2.0) -> bytes:
    rng = np.random.default_rng(3)
    buf = io.BytesIO()
    sf.write(buf, (rng.standard_normal(int(seconds * SR)) * 0.1).astype(np.float32), SR, format="WAV")
    return buf.getvalue()


def _session(client) -> str:
    res = client.post("/audio-edit/sessions", data={"mode": "speech", "name": "kn"},
                      files={"file": ("take.wav", _wav_upload(), "audio/wav")})
    assert res.status_code == 200, res.text
    return res.json()["session"]["id"]


# ── listing and validation ───────────────────────────────────────────────

def test_engine_listing_has_local_and_cloud(client):
    body = client.get("/audio-edit/asr/engines").json()
    ids = {e["id"]: e for e in body["engines"]}
    assert {"whisper", "indicconformer_kn", "indicwhisper_kn", "vasista22_whisper_kn"} <= set(ids)
    assert {"sarvam", "smallest", "google", "azure", "openai", "deepgram", "assemblyai", "bhashini"} <= set(ids)
    assert ids["whisper"]["available"] is True
    assert ids["vasista22_whisper_kn"]["reason"] == "not_installed"
    assert ids["openai"]["reason"] == "key_missing" and ids["openai"]["kind"] == "cloud"
    assert ids["indicwhisper_kn"]["languages"] == ["kn"]
    assert {"code": "kn", "name": "Kannada"} in body["languages"]


def test_validate_options():
    assert asr.validate_options(None) == {"engine": "whisper", "language": "auto"}
    assert asr.validate_options({"engine": "whisper", "language": "kn"})["language"] == "kn"
    # A Kannada-only model with "auto" decodes as Kannada.
    assert asr.validate_options({"engine": "vasista22_whisper_kn"})["language"] == "kn"
    for bad, code in (({"engine": "nope"}, "invalid_asr_engine"),
                      ({"language": "xx"}, "invalid_asr_language"),
                      ({"engine": "vasista22_whisper_kn", "language": "te"}, "asr_language_unsupported"),
                      ({"engine": "smallest", "language": "kn"}, "asr_language_unsupported")):
        with pytest.raises(asr.AsrError) as err:
            asr.validate_options(bad)
        assert err.value.code == code


# ── keys ─────────────────────────────────────────────────────────────────

def test_keys_are_masked_and_clearable(client, secrets):
    res = client.put("/audio-edit/asr/keys/openai", json={"fields": {"api_key": "sk-test-123456789"}})
    assert res.status_code == 200
    field = res.json()["fields"][0]
    assert field["has_value"] and field["masked"] == "••••6789" and field["value"] is None
    assert "sk-test-123456789" not in json.dumps(client.get("/audio-edit/asr/keys").json())
    assert secrets["audio_edit_asr.openai.api_key"] == "sk-test-123456789"
    ids = {e["id"]: e for e in client.get("/audio-edit/asr/engines").json()["engines"]}
    assert ids["openai"]["available"] is True
    client.put("/audio-edit/asr/keys/openai", json={"fields": {"api_key": ""}})
    assert "audio_edit_asr.openai.api_key" not in secrets


def test_azure_region_is_plain_and_needed(client):
    res = client.put("/audio-edit/asr/keys/azure", json={"fields": {"api_key": "k" * 32}}).json()
    assert res["configured"] is False
    res = client.put("/audio-edit/asr/keys/azure", json={"fields": {"region": "centralindia"}}).json()
    region = next(f for f in res["fields"] if f["id"] == "region")
    assert res["configured"] is True and region["value"] == "centralindia"


def test_env_key_fallback(client, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-env-key-0000")
    field = client.get("/audio-edit/asr/keys").json()["providers"]
    dg = next(p for p in field if p["id"] == "deepgram")
    assert dg["configured"] and dg["fields"][0]["source"] == "env"


def test_unknown_provider_and_field(client):
    assert client.put("/audio-edit/asr/keys/nope", json={"fields": {}}).status_code == 404
    res = client.put("/audio-edit/asr/keys/openai", json={"fields": {"bogus": "x"}})
    assert res.status_code == 422


# ── transcribe gating and language forcing ───────────────────────────────

def test_transcribe_blocks_without_key_or_model(client):
    sid = _session(client)
    res = client.post(f"/audio-edit/sessions/{sid}/transcribe", json={"engine": "sarvam", "language": "kn"})
    assert res.status_code == 409 and res.json()["detail"]["code"] == "asr_key_missing"
    res = client.post(f"/audio-edit/sessions/{sid}/transcribe", json={"engine": "indicwhisper_kn", "language": "kn"})
    assert res.status_code == 409 and res.json()["detail"]["code"] == "asr_model_missing"
    # Nothing started: the session stays uploaded.
    assert client.get(f"/audio-edit/sessions/{sid}").json()["stage"] == "uploaded"


def test_whisper_language_is_forced(client, monkeypatch):
    calls = []

    class Backend:
        id = "faster-whisper"

        def transcribe(self, audio_path, *, word_timestamps=True, language=None):
            calls.append(language)
            # Whisper's own detector would have said Telugu.
            return {"language": "te", "words": [{"word": "ನಮಸ್ಕಾರ", "start": 0.1, "end": 0.6}]}

    import services.asr_backend as ab
    monkeypatch.setattr(ab, "get_active_asr_backend", lambda: Backend())
    sid = _session(client)
    res = client.post(f"/audio-edit/sessions/{sid}/transcribe", json={"engine": "whisper", "language": "kn"})
    assert res.json()["job"]["status"] == "succeeded", res.text
    session = client.get(f"/audio-edit/sessions/{sid}").json()
    assert calls == ["kn"]
    assert session["transcript"]["language"] == "kn" and session["asr"] == {"engine": "whisper", "language": "kn"}
    # Empty body keeps the old behaviour: active engine, auto-detect.
    res = client.post(f"/audio-edit/sessions/{sid}/transcribe")
    assert res.json()["job"]["status"] == "succeeded"
    assert calls == ["kn", None]


def test_backend_without_language_param_falls_back_to_pytorch_whisper(monkeypatch, tmp_path):
    import services.asr_backend as ab

    class NoLang:
        id = "parakeet"

        def transcribe(self, audio_path, *, word_timestamps=True):
            raise AssertionError("must not be used when a language is forced")

    used = []

    class Torch:
        def __init__(self, asr_pipe=None):
            pass

        def transcribe(self, audio_path, *, word_timestamps=True, language=None, task="transcribe"):
            used.append(language)
            return {"chunks": []}

    monkeypatch.setattr(ab, "get_active_asr_backend", lambda: NoLang())
    monkeypatch.setattr(ab, "PyTorchWhisperBackend", Torch)
    out = asr.transcribe(tmp_path / "x.wav", {"engine": "whisper", "language": "kn"})
    assert used == ["kn"] and out["language"] == "kn"


# ── chunking / spreading ─────────────────────────────────────────────────

def test_spread_words_and_chunks():
    words = asr.spread_words("ನಾನು ಮನೆಗೆ ಹೋದೆ", 1.0, 4.0)
    assert [w["word"] for w in words] == ["ನಾನು", "ಮನೆಗೆ", "ಹೋದೆ"]
    assert words[0]["start"] == 1.0 and words[-1]["end"] <= 4.0
    assert all(a["end"] <= b["start"] for a, b in zip(words, words[1:]))
    audio = np.random.default_rng(0).standard_normal(asr.ASR_SR * 70).astype(np.float32) * 0.1
    bounds = asr.chunk_bounds(audio, 28.0)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    assert all(b - a <= 28 * asr.ASR_SR for a, b in bounds)
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))


# ── cloud adapters (HTTP faked) ──────────────────────────────────────────

class _Res:
    def __init__(self, body, status=200):
        self._body, self.status_code, self.text = body, status, json.dumps(body)

    def json(self):
        return self._body


class _Http:
    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self.responder("POST", url, kw)

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.responder("GET", url, kw)


def _audio(seconds=3.0):
    t = np.arange(int(seconds * asr.ASR_SR)) / asr.ASR_SR
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_sarvam_sends_kannada_and_spreads_phrases(monkeypatch, secrets):
    secrets["audio_edit_asr.sarvam.api_key"] = "sv-key"
    http = _Http(lambda m, u, kw: _Res({"transcript": "ನಮಸ್ಕಾರ ಗೆಳೆಯ", "language_code": "kn-IN", "timestamps": {
        "words": ["ನಮಸ್ಕಾರ ಗೆಳೆಯ"], "start_time_seconds": [0.2], "end_time_seconds": [1.8]}}))
    monkeypatch.setattr(asr, "_http", lambda: http)
    out = asr._sarvam(_audio(), "kn", lambda: None, lambda p: None)
    _, url, kw = http.calls[0]
    assert url == "https://api.sarvam.ai/speech-to-text"
    assert kw["data"]["language_code"] == "kn-IN" and kw["headers"]["api-subscription-key"] == "sv-key"
    assert out["language"] == "kn" and [w["word"] for w in out["words"]] == ["ನಮಸ್ಕಾರ", "ಗೆಳೆಯ"]


def test_openai_word_timestamps(monkeypatch, secrets):
    secrets["audio_edit_asr.openai.api_key"] = "sk"
    http = _Http(lambda m, u, kw: _Res({"language": "kannada", "text": "a b",
                                         "words": [{"word": "a", "start": 0.1, "end": 0.3},
                                                   {"word": "b", "start": 0.4, "end": 0.6}]}))
    monkeypatch.setattr(asr, "_http", lambda: http)
    out = asr._openai(_audio(), "kn", lambda: None, lambda p: None)
    assert http.calls[0][2]["data"]["language"] == "kn"
    assert out["words"][1] == {"word": "b", "start": 0.4, "end": 0.6} and out["language"] == "kn"


def test_deepgram_and_auth_error(monkeypatch, secrets):
    secrets["audio_edit_asr.deepgram.api_key"] = "dg"
    http = _Http(lambda m, u, kw: _Res({"results": {"channels": [{"alternatives": [{"words": [
        {"word": "hi", "punctuated_word": "Hi,", "start": 0.0, "end": 0.2}]}]}]}}))
    monkeypatch.setattr(asr, "_http", lambda: http)
    out = asr._deepgram(_audio(), "kn", lambda: None, lambda p: None)
    assert http.calls[0][2]["params"]["language"] == "kn" and out["words"][0]["word"] == "Hi,"
    monkeypatch.setattr(asr, "_http", lambda: _Http(lambda m, u, kw: _Res({"err": "bad"}, 401)))
    with pytest.raises(asr.AsrError) as err:
        asr._deepgram(_audio(), "kn", lambda: None, lambda p: None)
    assert err.value.code == "asr_auth_failed"


def test_google_azure_smallest_bhashini_assemblyai(monkeypatch, secrets):
    for k, v in {"google.api_key": "g", "azure.api_key": "az", "azure.region": "centralindia",
                 "smallest.api_key": "sm", "bhashini.api_key": "bh", "bhashini.user_id": "u1",
                 "assemblyai.api_key": "aa"}.items():
        secrets[f"audio_edit_asr.{k}"] = v

    def responder(method, url, kw):
        if "speech.googleapis.com" in url:
            assert kw["json"]["config"]["languageCode"] == "kn-IN"
            return _Res({"results": [{"alternatives": [{"words": [
                {"word": "ಒಂದು", "startTime": "0.100s", "endTime": "0.500s"}]}]}]})
        if "cognitive.microsoft.com" in url:
            assert url.startswith("https://centralindia.")
            assert json.loads(kw["files"]["definition"][1])["locales"] == ["kn-IN"]
            return _Res({"phrases": [{"locale": "kn-IN", "words": [
                {"text": "ಎರಡು", "offsetMilliseconds": 100, "durationMilliseconds": 300}]}]})
        if "smallest.ai" in url:
            return _Res({"transcription": "hello", "words": [{"word": "hello", "start": 0.1, "end": 0.4}]})
        if "getModelsPipeline" in url:
            return _Res({"pipelineInferenceAPIEndPoint": {"callbackUrl": "https://bh/infer",
                                                          "inferenceApiKey": {"name": "Authorization", "value": "t"}},
                         "pipelineResponseConfig": [{"config": [{"serviceId": "svc"}]}]})
        if url == "https://bh/infer":
            assert kw["json"]["pipelineTasks"][0]["config"]["serviceId"] == "svc"
            return _Res({"pipelineResponse": [{"output": [{"source": "ಮೂರು ನಾಲ್ಕು"}]}]})
        if url.endswith("/v2/upload"):
            return _Res({"upload_url": "https://cdn/x"})
        if url.endswith("/v2/transcript"):
            assert kw["json"]["language_code"] == "kn"
            return _Res({"id": "t1"})
        if "/v2/transcript/t1" in url:
            return _Res({"status": "completed", "language_code": "kn",
                         "words": [{"text": "ಐದು", "start": 100, "end": 400}]})
        raise AssertionError(url)

    monkeypatch.setattr(asr, "_http", lambda: _Http(responder))
    noop = (lambda: None, lambda p: None)
    assert asr._google(_audio(), "kn", *noop)["words"][0] == {"word": "ಒಂದು", "start": 0.1, "end": 0.5}
    assert asr._azure(_audio(), "kn", *noop)["words"][0]["word"] == "ಎರಡು"
    assert asr._smallest(_audio(), "hi", *noop)["words"][0]["word"] == "hello"
    assert [w["word"] for w in asr._bhashini(_audio(), "kn", *noop)["words"]] == ["ಮೂರು", "ನಾಲ್ಕು"]
    assert asr._assemblyai(_audio(), "kn", *noop)["words"][0] == {"word": "ಐದು", "start": 0.1, "end": 0.4}


def test_cloud_engine_end_to_end_through_session(client, monkeypatch, secrets):
    secrets["audio_edit_asr.sarvam.api_key"] = "sv"
    monkeypatch.setattr(asr, "_sarvam", lambda audio, language, *a: {
        "language": language, "words": [{"word": "ನಮಸ್ಕಾರ", "start": 0.1, "end": 0.5}]})
    sid = _session(client)
    res = client.post(f"/audio-edit/sessions/{sid}/transcribe", json={"engine": "sarvam", "language": "kn"})
    assert res.json()["job"]["status"] == "succeeded", res.text
    t = client.get(f"/audio-edit/sessions/{sid}").json()["transcript"]
    assert t["language"] == "kn" and t["engine"] == "sarvam" and t["words"][0]["word"] == "ನಮಸ್ಕಾರ"


def test_download_archive_marks_complete(monkeypatch, tmp_path):
    monkeypatch.setenv("OMNIVOICE_AUDIO_EDIT_DIR", str(tmp_path))
    buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("kannada_models/whisper-medium-kn/config.json", "{}")
        zf.writestr("kannada_models/whisper-medium-kn/preprocessor_config.json", "{}")
    payload = buf.getvalue()

    class Stream:
        headers = {"content-length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield payload

    import requests
    monkeypatch.setattr(requests, "get", lambda url, **kw: Stream())
    monkeypatch.setattr(asr, "RUN_DOWNLOAD_INLINE", True)
    got = asr.start_download("indicwhisper_kn")
    assert got["status"] == "succeeded", got
    path = asr.local_model_path(asr.ENGINES["indicwhisper_kn"])
    assert path and path.endswith("whisper-medium-kn")
    assert asr.engine_status(asr.ENGINES["indicwhisper_kn"])["installed"] is True
