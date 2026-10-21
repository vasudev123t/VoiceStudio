"""Index (re)build for imported RVC voices (review item 1b)."""
import json
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch


@pytest.fixture
def env(monkeypatch, tmp_path):
    import services.rvc as rvc_mod
    from services import rvc_training

    monkeypatch.setenv("OMNIVOICE_RVC_VOICES_DIR", str(tmp_path / "voices"))
    monkeypatch.setenv("OMNIVOICE_RVC_TRAINING_DIR", str(tmp_path / "training"))
    for var in ("OMNIVOICE_APPLIO_DIR", "OMNIVOICE_APPLIO_PYTHON", "OMNIVOICE_RVC_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rvc_mod, "_pref", lambda key, default=None: default)
    monkeypatch.setattr(rvc_mod, "managed_applio_dir", lambda: tmp_path / "no-managed")
    monkeypatch.setattr(rvc_mod, "select_device", lambda preference=None: "cpu")
    rvc_training._reset_for_tests()
    yield tmp_path
    rvc_training._reset_for_tests()
    rvc_mod.reset_service()


def _fake_pth(path, sr=40000):
    torch.save({"weight": {"w": torch.zeros(2)}, "config": [1, 2, sr], "version": "v2",
                "f0": 1, "sr": f"{sr // 1000}k"}, str(path))
    return path


def _stub_applio_core(root: Path) -> Path:
    """core.py stand-in that writes the artifacts each stage verifies."""
    (root / "rvc" / "infer").mkdir(parents=True)
    (root / "rvc" / "infer" / "infer.py").write_text("")
    (root / "core.py").write_text(textwrap.dedent("""
        import json, os, sys
        logs = os.environ["APPLIO_LOGS_DIR"]
        stage = sys.argv[1]
        args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        exp = os.path.join(logs, args["--model-name"])
        os.makedirs(exp, exist_ok=True)
        with open(os.path.join(logs, "calls.jsonl"), "a") as fh:
            fh.write(json.dumps({"stage": stage, **args}) + "\\n")
        if stage == "preprocess":
            os.makedirs(os.path.join(exp, "sliced_audios"), exist_ok=True)
            open(os.path.join(exp, "sliced_audios", "0_0.wav"), "w").close()
        elif stage == "extract":
            os.makedirs(os.path.join(exp, "extracted"), exist_ok=True)
            open(os.path.join(exp, "extracted", "0_0.npy"), "w").close()
        elif stage == "index":
            with open(os.path.join(exp, args["--model-name"] + ".index"), "wb") as fh:
                fh.write(b"FAISS-INDEX")
    """))
    return root


def _wait(job, timeout=30):
    from services import rvc_training

    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.status in rvc_training._TERMINAL:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job stuck: {job.status} {list(job.log)[-5:]}")


def test_index_build_runs_preprocess_extract_index_and_attaches(env, monkeypatch):
    from services import rvc_registry, rvc_training

    root = _stub_applio_core(env / "Applio")
    monkeypatch.setenv("OMNIVOICE_APPLIO_DIR", str(root))
    monkeypatch.setenv("OMNIVOICE_APPLIO_PYTHON", sys.executable)
    voice = rvc_registry.import_from_paths("Salman", _fake_pth(env / "m.pth", 40000))
    assert voice["has_index"] is False and voice["index_source"] is None
    data = env / "clips"
    data.mkdir()
    (data / "a.wav").write_bytes(b"RIFF")
    job = _wait(rvc_training.start_index_build(voice["id"], dataset_dir=data))
    assert job.status == "succeeded", list(job.log)
    after = rvc_registry.get_voice(voice["id"])
    assert after["has_index"] is True and after["index_source"] == "rebuilt"
    calls = [json.loads(line) for line in
             (Path(job.workdir) / "logs" / "calls.jsonl").read_text().splitlines()]
    assert [c["stage"] for c in calls] == ["preprocess", "extract", "index"]
    assert calls[0]["--sample-rate"] == "40000", "index must use the voice's own sample rate"
    assert calls[1]["--embedder-model"] == "contentvec"
    # Heavy features are dropped; the registry keeps its own copy of the index.
    assert not (Path(job.workdir) / "logs" / calls[0]["--model-name"] / "extracted").exists()


def test_rebuild_keeps_previous_index_as_backup(env, monkeypatch):
    from services import rvc_registry

    idx = env / "old.index"
    idx.write_bytes(b"OLD")
    voice = rvc_registry.import_from_paths("A", _fake_pth(env / "a.pth"), idx)
    new = env / "new.index"
    new.write_bytes(b"NEW")
    rvc_registry.attach_index(voice["id"], new)
    model, index = rvc_registry.voice_files(voice["id"])
    assert Path(index).read_bytes() == b"NEW"
    backups = list(Path(model).parent.glob("model.index.bak-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"OLD"


def test_index_build_requires_audio_or_existing_features(env, monkeypatch):
    from services import rvc_registry, rvc_training

    voice = rvc_registry.import_from_paths("A", _fake_pth(env / "a.pth"))
    with pytest.raises(ValueError):
        rvc_training.start_index_build(voice["id"], dataset_dir=None, runner=lambda job: None)
    with pytest.raises(rvc_registry.VoiceNotFound):
        rvc_training.start_index_build("missing-voice", dataset_dir=env)


def test_index_route_starts_job_and_status_lists_kind(env, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import voice_conversion
    from services import rvc_registry, rvc_training

    started = {}

    def fake_start(voice_id, *, dataset_dir=None, job_id=None, runner=None):
        started.update(voice_id=voice_id, files=sorted(p.name for p in dataset_dir.iterdir()))
        job = rvc_training.Job(id=job_id, kind="index", name="A", voice_id=voice_id)
        return job

    monkeypatch.setattr(rvc_training, "start_index_build", fake_start)
    app = FastAPI()
    app.include_router(voice_conversion.router)
    client = TestClient(app)
    voice = rvc_registry.import_from_paths("A", _fake_pth(env / "a.pth"))
    r = client.post(f"/voice-conversion/voices/{voice['id']}/index",
                    files=[("files", ("take one.wav", b"RIFF", "audio/wav"))])
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "index"
    assert started["voice_id"] == voice["id"] and started["files"] == ["0000_take one.wav"]
    assert client.post("/voice-conversion/voices/nope-000000/index").status_code == 404
    bad = client.post(f"/voice-conversion/voices/{voice['id']}/index",
                      files=[("files", ("x.txt", b"x", "text/plain"))])
    assert bad.status_code == 415
