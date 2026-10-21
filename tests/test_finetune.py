"""Fine-tuning jobs (review item 6) — hermetic: a fake trainer stands in for
the real OmniVoice LoRA run (that one is exercised on the GPU box)."""
import json
import sys
import textwrap
import time

import pytest

from services import finetune as ft


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_FINETUNE_DIR", str(tmp_path / "ft"))
    monkeypatch.setenv("OMNIVOICE_DATASETS_DIR", str(tmp_path / "ds"))
    ds = tmp_path / "ds" / "me-abc123"
    (ds / "clips").mkdir(parents=True)
    (ds / "dataset.json").write_text(json.dumps({"options": {"language": "kn"}}))
    (ds / "metadata.csv").write_text("clips/a.wav|ನಮಸ್ಕಾರ\n", encoding="utf-8")
    prefs = {}
    from core import prefs as prefs_mod

    monkeypatch.setattr(prefs_mod, "get", lambda k, d=None: prefs.get(k, d))
    monkeypatch.setattr(prefs_mod, "set_", lambda k, v: prefs.__setitem__(k, v))
    import services.gpu_memory as gm

    monkeypatch.setattr(gm, "release_gpu_holders", lambda reason="": ["tts"])
    ft._JOBS.clear()
    yield tmp_path, prefs
    ft._JOBS.clear()


def _ready(monkeypatch, tmp_path):
    monkeypatch.setattr(ft, "preflight", lambda rid: {
        "id": rid, "label": "x", "base": "b", "runnable": True, "ready": True,
        "missing_packages": [], "weights_on_disk": True, "weights_path": str(tmp_path),
        "defaults": {"steps": 3, "rank": 16}, "note": None})


FAKE = textwrap.dedent('''
    import json, sys, pathlib
    cfg = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = pathlib.Path(cfg["output_dir"])
    print(json.dumps({"event": "stage", "stage": "load"}), flush=True)
    print("noise from a library", flush=True)
    for s in range(1, cfg["steps"] + 1):
        print(json.dumps({"event": "step", "step": s, "steps": cfg["steps"], "loss": 1.0 / s,
                          "loss_ema": 1.0 / s, "peak_vram_mb": 4000}), flush=True)
    if cfg.get("fail"):
        print("boom", file=sys.stderr); sys.exit(3)
    (out / "adapter").mkdir()
    (out / "adapter" / "adapter_config.json").write_text("{}")
    (out / "adapter" / "voicestudio.json").write_text(json.dumps({"recipe": "omnivoice-lora", "steps": cfg["steps"]}))
    print(json.dumps({"event": "done", "adapter": str(out / "adapter")}), flush=True)
''')


def _wait(job, timeout=30):
    end = time.time() + timeout
    while job.status in ("queued", "running") and time.time() < end:
        time.sleep(0.05)
    return job


def test_recipes_report_blockers(monkeypatch):
    monkeypatch.setattr(ft.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(ft, "_cached", lambda repo: None)
    by_id = {r["id"]: r for r in ft.recipes()}
    assert set(by_id) == {"omnivoice-lora", "voxcpm2-lora", "indicf5-kn"}
    assert by_id["omnivoice-lora"]["missing_packages"] == ["peft"]
    assert not any(r["ready"] for r in by_id.values())
    assert by_id["voxcpm2-lora"]["runnable"] is False
    assert by_id["omnivoice-lora"]["defaults"]["rank"] == 16


def test_start_refuses_blocked_and_unknown(env, monkeypatch):
    with pytest.raises(ValueError):
        ft.start("voxcpm2-lora", dataset_id="me-abc123", name="x")
    with pytest.raises(KeyError):
        ft.preflight("nope")
    _ready(monkeypatch, env[0])
    with pytest.raises(KeyError):
        ft.start("omnivoice-lora", dataset_id="../etc", name="x")


def test_job_runs_fake_trainer_and_adapter_activates(env, monkeypatch):
    tmp_path, prefs = env
    _ready(monkeypatch, tmp_path)
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text(FAKE)
    monkeypatch.setattr(ft, "TRAINER", trainer)
    monkeypatch.setenv("SOME_API_TOKEN", "do-not-leak")
    job = _wait(ft.start("omnivoice-lora", dataset_id="me-abc123", name="My Voice!",
                         overrides={"steps": 4}, python=sys.executable))
    assert job.status == "succeeded", (job.error, job.log)
    assert job.step == 4 and job.loss == 0.25 and job.progress == 1.0
    cfg = json.loads((tmp_path / "ft" / job.adapter_id / "config.json").read_text())
    assert cfg["language"] == "kn" and cfg["steps"] == 4 and cfg["rank"] == 16
    assert "do-not-leak" not in json.dumps(cfg)
    assert job.adapter_id.startswith("my-voice-")
    ads = ft.list_adapters()
    assert [a["id"] for a in ads] == [job.adapter_id] and ads[0]["name"] == "My Voice!"
    unloaded = []
    from services import model_manager as mm

    monkeypatch.setattr(mm, "unload_shared_model", lambda: unloaded.append(1) or True)
    ft.set_active_adapter(job.adapter_id)
    assert prefs[ft.PREF_KEY] == job.adapter_id and unloaded
    assert ft.list_adapters()[0]["active"] is True
    with pytest.raises(KeyError):
        ft.set_active_adapter("../../x")
    ft.set_active_adapter(None)
    assert ft.active_adapter_id() is None


def test_failed_trainer_reports_stderr_tail(env, monkeypatch):
    tmp_path, _ = env
    _ready(monkeypatch, tmp_path)
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text(FAKE)
    monkeypatch.setattr(ft, "TRAINER", trainer)
    job = _wait(ft.start("omnivoice-lora", dataset_id="me-abc123", name="x",
                         overrides={"steps": 1, "fail": True}, python=sys.executable))
    assert job.status == "failed" and "boom" in job.error


def test_one_job_at_a_time(env, monkeypatch):
    _ready(monkeypatch, env[0])
    ft._JOBS["busy"] = ft.FinetuneJob(id="busy", recipe="omnivoice-lora", name="b",
                                      dataset_id="me-abc123", status="running")
    with pytest.raises(RuntimeError):
        ft.start("omnivoice-lora", dataset_id="me-abc123", name="x")


def test_apply_active_adapter_is_best_effort(env):
    _, prefs = env

    class M:
        llm = object()

    m = M()
    assert ft.apply_active_adapter(m) is None
    prefs[ft.PREF_KEY] = "missing-adapter"
    assert ft.apply_active_adapter(m) is None and m.llm is M.llm


def test_finetune_routes(env, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import voice_lab

    app = FastAPI()
    app.include_router(voice_lab.router)
    c = TestClient(app)
    monkeypatch.setattr(ft, "_cached", lambda repo: None)
    r = c.get("/voice-lab/finetune/recipes")
    assert r.status_code == 200 and len(r.json()["recipes"]) == 3
    r = c.post("/voice-lab/finetune", json={"recipe": "voxcpm2-lora", "dataset_id": "me-abc123"})
    assert r.status_code == 412
    r = c.post("/voice-lab/finetune", json={"recipe": "zzz", "dataset_id": "me-abc123"})
    assert r.status_code == 404
    assert c.get("/voice-lab/finetune/jobs/nope").status_code == 404
    assert c.get("/voice-lab/finetune/adapters").json() == {"adapters": [], "active": None}
    assert c.put("/voice-lab/finetune/active", json={"adapter_id": "nope-nope"}).status_code == 404
    r = c.post("/voice-lab/finetune", json={"dataset_id": "x", "steps": 0})
    assert r.status_code == 422
