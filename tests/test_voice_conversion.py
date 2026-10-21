"""Voice conversion (RVC / Applio) — service, registry, training and router.

No real RVC weights or GPU: converters are faked, and the Applio paths run
against a *stub* Applio checkout (``core.py`` click-shaped CLI and
``rvc.infer.infer.VoiceConverter``) executed with this interpreter, so the
worker protocol and the preprocess → extract → train → index pipeline run
for real as subprocesses.

Contract under test:
  * params: ranges and f0 methods are validated (service + 422 at the API);
  * device: cuda:0 > mps > cpu, env override wins;
  * backend: Applio preferred when configured, rvc-python otherwise, typed
    503 when neither is installed; the cached backend unloads on a voice switch;
  * rvc-python is driven with its REAL API (load_model(index_path=…),
    set_params(f0up_key=…, f0method=…)) — the old hook used neither;
  * registry CRUD copies files, rejects G_/D_ training checkpoints and ids
    that try to leave the registry;
  * legacy dub hook stays a no-op unless enabled and swallows failures;
  * training builds the verified Applio CLI argv, follows epochs and registers
    the final <name>_<E>e_<S>s.pth + index;
  * router: status, import/list/rename/delete, convert (keeps the original,
    WAV at the model's rate, provenance-marked), history, train, cancel.
"""
import asyncio
import io
import os
import sys
import textwrap
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import torch


# ── helpers ──────────────────────────────────────────────────────────────────


def _wav_bytes(sr=16000, seconds=0.25, freq=220.0) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    data = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


def _write_wav(path, sr=16000, seconds=0.25):
    Path(path).write_bytes(_wav_bytes(sr, seconds))


def _fake_pth(path, sr=48000):
    torch.save({"weight": {"w": torch.zeros(2)}, "config": [1, 2, sr], "version": "v2",
                "f0": 1, "sr": f"{sr // 1000}k"}, str(path))
    return path


@pytest.fixture
def rvc(monkeypatch, tmp_path):
    import services.rvc as rvc_mod

    monkeypatch.setenv("OMNIVOICE_RVC_VOICES_DIR", str(tmp_path / "voices"))
    for var in ("OMNIVOICE_APPLIO_DIR", "OMNIVOICE_APPLIO_PYTHON", "OMNIVOICE_RVC_DEVICE",
                "OMNIVOICE_RVC_ENABLED", "OMNIVOICE_RVC_MODEL", "OMNIVOICE_RVC_VOICE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rvc_mod, "_pref", lambda key, default=None: default)
    monkeypatch.setattr(rvc_mod, "managed_applio_dir", lambda: tmp_path / "no-managed-applio")
    rvc_mod.reset_service()
    yield rvc_mod
    rvc_mod.reset_service()


@pytest.fixture
def registry(rvc):
    from services import rvc_registry

    return rvc_registry


class _FakeBackend:
    name = "applio"
    device = "cuda:0"

    def __init__(self, sr=48000):
        self.sr = sr
        self.calls = []
        self.unloads = 0

    def convert(self, inp, out, model, index, params):
        self.calls.append((inp, out, model, index, params))
        _write_wav(out, sr=self.sr)
        return self.sr

    def unload(self):
        self.unloads += 1

    def close(self):
        pass


# ── params / device ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("kwargs", [
    {"pitch": 25}, {"pitch": -25}, {"f0_method": "dio"}, {"index_rate": 1.5},
    {"protect": 0.6}, {"rms_mix_rate": -0.1}, {"filter_radius": 8},
])
def test_params_reject_out_of_range(rvc, kwargs):
    with pytest.raises(ValueError):
        rvc.ConversionParams(**kwargs).validate()


def test_params_defaults_are_valid(rvc):
    p = rvc.ConversionParams().validate()
    assert (p.pitch, p.f0_method, p.protect) == (0, "rmvpe", 0.33)


def test_device_autoselect_prefers_cuda_then_mps(rvc, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert rvc.select_device() == "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert rvc.select_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert rvc.select_device() == "cpu"


def test_device_env_override_wins(rvc, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("OMNIVOICE_RVC_DEVICE", "cpu")
    assert rvc.select_device() == "cpu"
    monkeypatch.setenv("OMNIVOICE_RVC_DEVICE", "cuda")
    assert rvc.select_device() == "cuda:0"


# ── registry ─────────────────────────────────────────────────────────────────


def test_registry_crud_copies_and_never_moves_originals(registry, tmp_path):
    src = _fake_pth(tmp_path / "Narrator_300e_9000s.pth")
    idx = tmp_path / "Narrator.index"
    idx.write_bytes(b"faiss")
    voice = registry.import_from_paths("Narrator", src, idx)
    assert src.is_file() and idx.is_file(), "originals must stay in place"
    assert voice["name"] == "Narrator" and voice["has_index"] is True
    assert voice["sample_rate"] == 48000 and voice["version"] == "v2"
    model, index = registry.voice_files(voice["id"])
    assert Path(model).parent.parent == Path(os.environ["OMNIVOICE_RVC_VOICES_DIR"]).resolve()
    assert [v["id"] for v in registry.list_voices()] == [voice["id"]]
    assert registry.rename_voice(voice["id"], "  Hero  ")["name"] == "Hero"
    registry.delete_voice(voice["id"])
    assert registry.list_voices() == []
    with pytest.raises(registry.VoiceNotFound):
        registry.get_voice(voice["id"])


@pytest.mark.parametrize("bad", ["../etc", "..", "a/b", "A B", "", "x" * 80])
def test_registry_rejects_ids_outside_root(registry, bad):
    with pytest.raises(registry.VoiceNotFound):
        registry.voice_files(bad)


def test_registry_rejects_training_checkpoint(registry, tmp_path):
    # A D_ checkpoint (optimizer state, no generator weights) is refused.
    bad = tmp_path / "D_bad.pth"
    torch.save({"optimizer": {}, "iteration": 1}, str(bad))
    with pytest.raises(registry.InvalidVoiceFile):
        registry.import_from_paths("x", bad)
    assert registry.list_voices() == []


def test_registry_requires_name_and_pth(registry, tmp_path):
    with pytest.raises(ValueError):
        registry.import_from_paths("   ", _fake_pth(tmp_path / "m.pth"))
    txt = tmp_path / "m.txt"
    txt.write_text("x")
    with pytest.raises(registry.InvalidVoiceFile):
        registry.import_from_paths("x", txt)


# ── backend selection / caching ──────────────────────────────────────────────


def test_no_backend_is_a_typed_unavailable_error(rvc, monkeypatch):
    monkeypatch.setattr(rvc, "rvc_python_available", lambda: False)
    svc = rvc.get_service()
    assert svc.status()["available"] is False
    with pytest.raises(rvc.RVCUnavailable):
        svc.backend()


def _stub_applio(root: Path) -> Path:
    """A minimal Applio-shaped checkout: real worker protocol, fake DSP."""
    (root / "rvc" / "infer").mkdir(parents=True)
    (root / "rvc" / "configs").mkdir(parents=True)
    for pkg in ("rvc", "rvc/infer", "rvc/configs"):
        (root / pkg / "__init__.py").write_text("")
    (root / "rvc" / "configs" / "config.py").write_text(textwrap.dedent("""
        class _Cfg:
            device = "cpu"
            gpu_name = None
        def inference_config():
            return _Cfg()
    """))
    (root / "rvc" / "infer" / "infer.py").write_text(textwrap.dedent("""
        import json, shutil, wave
        class VoiceConverter:
            def __init__(self, config=None):
                self.config = config
                self.tgt_sr = None
                self.unloads = 0
            def cleanup_model(self):
                self.tgt_sr = None
                with open("unloads.log", "a") as fh:
                    fh.write("unload\\n")
            def convert_audio(self, audio_input_path, audio_output_path, model_path,
                              index_path, **kw):
                print("Converting", audio_input_path)  # must not corrupt protocol
                if kw.get("f0_method") == "boom":
                    raise RuntimeError("kaboom")
                self.tgt_sr = 40000
                with wave.open(audio_output_path, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(40000)
                    wf.writeframes(b"\\x00\\x00" * 400)
                with open(audio_output_path + ".json", "w") as fh:
                    json.dump({"model": model_path, "index": index_path, **kw}, fh)
    """))
    (root / "core.py").write_text("")
    return root


def test_applio_preferred_and_worker_protocol_roundtrip(rvc, registry, monkeypatch, tmp_path):
    root = _stub_applio(tmp_path / "Applio")
    monkeypatch.setenv("OMNIVOICE_APPLIO_DIR", str(root))
    monkeypatch.setenv("OMNIVOICE_APPLIO_PYTHON", sys.executable)
    monkeypatch.setattr(rvc, "rvc_python_available", lambda: True)
    svc = rvc.get_service()
    st = svc.status()
    assert st["backend"] == "applio" and st["applio"]["ready"] is True
    assert st["f0_methods"] == ["rmvpe", "crepe", "fcpe"]

    voice = registry.import_from_paths("A", _fake_pth(tmp_path / "a.pth"))
    src = tmp_path / "in.wav"
    _write_wav(src)
    out = tmp_path / "out.wav"
    params = rvc.ConversionParams(pitch=3, index_rate=0.5, protect=0.4, rms_mix_rate=0.25)
    try:
        result = svc.convert_voice(voice["id"], str(src), str(out), params)
        assert result.backend == "applio" and result.sample_rate == 40000
        assert result.device == "cpu"  # reported by the worker's ping
        sent = __import__("json").loads(Path(str(out) + ".json").read_text())
        assert sent["pitch"] == 3 and sent["volume_envelope"] == 0.25
        assert sent["protect"] == 0.4
        assert sent["index_rate"] == 0.0, "no index → index_rate must be 0"
        assert sent["export_format"] == "WAV"
        # Applio has no harvest/pm: refused before reaching the worker.
        with pytest.raises(ValueError):
            svc.convert_voice(voice["id"], str(src), str(out),
                              rvc.ConversionParams(f0_method="harvest"))
        assert svc.status()["worker_running"] is True
    finally:
        rvc.reset_service()


def test_applio_worker_frees_previous_model_on_voice_switch(rvc, registry, monkeypatch,
                                                            tmp_path):
    root = _stub_applio(tmp_path / "Applio")
    monkeypatch.setenv("OMNIVOICE_APPLIO_DIR", str(root))
    monkeypatch.setenv("OMNIVOICE_APPLIO_PYTHON", sys.executable)
    a = registry.import_from_paths("A", _fake_pth(tmp_path / "a.pth"))
    b = registry.import_from_paths("B", _fake_pth(tmp_path / "b.pth"))
    src = tmp_path / "in.wav"
    _write_wav(src)
    svc = rvc.get_service()
    try:
        for vid in (a["id"], a["id"], b["id"]):
            svc.convert_voice(vid, str(src), str(tmp_path / "o.wav"), rvc.ConversionParams())
        unloads = (root / "unloads.log").read_text().splitlines()
        assert unloads == ["unload"], "exactly one unload: on the A → B switch"
    finally:
        rvc.reset_service()


def test_service_unload_delegates_to_cached_backend(rvc, registry, tmp_path, monkeypatch):
    svc = rvc.get_service()
    fake = _FakeBackend()
    monkeypatch.setattr(svc, "backend", lambda: fake)
    svc._backend = fake
    a = registry.import_from_paths("A", _fake_pth(tmp_path / "a.pth"))
    src = tmp_path / "in.wav"
    _write_wav(src)
    r = svc.convert_voice(a["id"], str(src), str(tmp_path / "o.wav"), rvc.ConversionParams())
    assert r.sample_rate == 48000 and fake.calls[0][2].endswith("model.pth")
    svc.unload()
    assert fake.unloads == 1


def test_convert_tensor_returns_model_rate(rvc, registry, tmp_path, monkeypatch):
    svc = rvc.get_service()
    monkeypatch.setattr(svc, "backend", lambda: _FakeBackend(sr=40000))
    v = registry.import_from_paths("A", _fake_pth(tmp_path / "a.pth"))
    tensor, sr, _ = svc.convert_tensor(torch.zeros(1, 2400), 24000, v["id"],
                                       rvc.ConversionParams())
    assert sr == 40000 and tensor.dim() == 2 and tensor.shape[0] == 1


def test_rvc_python_backend_uses_real_api(rvc, monkeypatch, tmp_path):
    calls = {}

    class FakeInference:
        def __init__(self, device):
            calls["device"] = device
            self.vc = type("VC", (), {"tgt_sr": 40000})()

        def load_model(self, path, version="v2", index_path=""):
            calls["load"] = (path, index_path)

        def set_params(self, **kw):
            calls["params"] = kw

        def infer_file(self, inp, out):
            _write_wav(out, 40000)

        def unload_model(self):
            calls["unloaded"] = True

    mod = type(sys)("rvc_python.infer")
    mod.RVCInference = FakeInference
    monkeypatch.setitem(sys.modules, "rvc_python", type(sys)("rvc_python"))
    monkeypatch.setitem(sys.modules, "rvc_python.infer", mod)
    backend = rvc.RvcPythonBackend(device="cpu")
    out = tmp_path / "o.wav"
    sr = backend.convert(str(tmp_path / "i.wav"), str(out), "/m.pth", "/m.index",
                         rvc.ConversionParams(pitch=-2, f0_method="harvest", filter_radius=5))
    assert sr == 40000
    assert calls["device"] == "cpu:0"
    assert calls["load"] == ("/m.pth", "/m.index")
    assert calls["params"]["f0up_key"] == -2 and calls["params"]["f0method"] == "harvest"
    assert calls["params"]["filter_radius"] == 5
    assert set(calls["params"]) <= {"index_rate", "filter_radius", "resample_sr",
                                    "rms_mix_rate", "protect", "f0up_key", "f0method"}


# ── legacy dub hook ──────────────────────────────────────────────────────────


def test_dub_hook_noop_when_disabled(rvc, tmp_path):
    wav = tmp_path / "seg.wav"
    _write_wav(wav)
    before = wav.read_bytes()
    assert rvc.is_enabled() is False
    assert rvc.apply_rvc(str(wav)) == str(wav)
    assert wav.read_bytes() == before


def test_dub_hook_converts_in_place_and_swallows_errors(rvc, tmp_path, monkeypatch):
    model = _fake_pth(tmp_path / "m.pth")
    monkeypatch.setenv("OMNIVOICE_RVC_ENABLED", "1")
    monkeypatch.setenv("OMNIVOICE_RVC_MODEL", str(model))
    monkeypatch.setenv("OMNIVOICE_RVC_PITCH", "4")
    svc = rvc.get_service()
    fake = _FakeBackend(sr=40000)
    monkeypatch.setattr(svc, "backend", lambda: fake)
    wav = tmp_path / "seg.wav"
    _write_wav(wav, sr=24000)
    assert rvc.apply_rvc(str(wav)) == str(wav)
    assert fake.calls[0][4].pitch == 4
    with wave.open(str(wav)) as wf:
        assert wf.getframerate() == 40000

    def boom(*a, **k):
        raise RuntimeError("gpu on fire")

    fake.convert = boom
    _write_wav(wav, sr=24000)
    assert rvc.apply_rvc(str(wav)) == str(wav)  # swallowed, original kept
    with wave.open(str(wav)) as wf:
        assert wf.getframerate() == 24000
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("tmp")]


def test_dub_hook_accepts_registered_voice_id(rvc, registry, tmp_path, monkeypatch):
    v = registry.import_from_paths("A", _fake_pth(tmp_path / "a.pth"))
    monkeypatch.setenv("OMNIVOICE_RVC_ENABLED", "1")
    monkeypatch.setenv("OMNIVOICE_RVC_VOICE", v["id"])
    cfg = rvc.rvc_config()
    assert cfg["model_path"].endswith("model.pth")


# ── training ─────────────────────────────────────────────────────────────────


def test_train_argv_matches_applio_cli(rvc):
    from services import rvc_training as tr

    s = tr.TrainSettings(name="Vasu", sample_rate=48000, epochs=200, batch_size=8,
                         save_every=25).validate()
    cmds = tr.build_stage_commands(s, python="py", applio="/A", model_name="Vasu",
                                   dataset="/d", gpu="0", cores=6)
    core = os.path.join("/A", "core.py")
    assert cmds["preprocess"][:3] == ["py", core, "preprocess"]
    assert cmds["preprocess"][cmds["preprocess"].index("--sample-rate") + 1] == "48000"
    assert "--cut-preprocess" in cmds["preprocess"]
    assert cmds["extract"][cmds["extract"].index("--f0-method") + 1] == "rmvpe"
    assert cmds["extract"][cmds["extract"].index("--gpu") + 1] == "0"
    t = cmds["train"]
    assert t[t.index("--total-epoch") + 1] == "200" and t[t.index("--batch-size") + 1] == "8"
    assert "--save-only-latest" in t and "--checkpointing" not in t
    assert cmds["index"][-2:] == ["--index-algorithm", "Auto"]
    assert "--no-pretrained" not in t
    s.pretrained = False
    t2 = tr.build_stage_commands(s, python="py", applio="/A", model_name="Vasu",
                                 dataset="/d", gpu="0", cores=6)["train"]
    assert t2[-1] == "--no-pretrained"


def test_pretrained_status_lists_missing_base_models(tmp_path):
    from services import rvc_training as tr

    hg = tmp_path / "rvc" / "models" / "pretraineds" / "hifi-gan"
    hg.mkdir(parents=True)
    (hg / "f0G40k.pth").write_bytes(b"x")
    (hg / "f0D40k.pth").write_bytes(b"x")
    (hg / "f0G48k.pth").write_bytes(b"")  # empty = missing
    st = tr.pretrained_status(tmp_path)
    assert st["40000"] == {"present": True, "missing": [], "download_bytes": 0}
    assert [m["file"] for m in st["48000"]["missing"]] == ["f0G48k.pth", "f0D48k.pth"]
    assert st["32000"]["download_bytes"] == 73_950_049 + 142_875_703


@pytest.mark.parametrize("kw", [{"sample_rate": 44100}, {"epochs": 0}, {"batch_size": 64},
                                {"f0_method": "harvest"}, {"gpu": "cuda"}])
def test_train_settings_validation(rvc, kw):
    from services import rvc_training as tr

    with pytest.raises(ValueError):
        tr.TrainSettings(name="x", **kw).validate()


def test_find_final_weights_picks_highest_epoch(tmp_path):
    from services import rvc_training as tr

    for name in ("V_10e_100s.pth", "V_200e_2000s.pth", "G_2333333.pth", "V_99e_5s.pth"):
        (tmp_path / name).write_bytes(b"x")
    assert tr.find_final_weights(tmp_path, "V").name == "V_200e_2000s.pth"


_FAKE_CORE = r'''
import os, sys, time, torch
args = sys.argv[1:]
cmd = args[0]
opt = {args[i]: (args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else True)
       for i in range(1, len(args)) if args[i].startswith("--")}
exp = os.path.join(os.environ["APPLIO_LOGS_DIR"], opt["--model-name"])
os.makedirs(exp, exist_ok=True)
if cmd == "preprocess":
    os.makedirs(os.path.join(exp, "sliced_audios"), exist_ok=True)
    open(os.path.join(exp, "sliced_audios", "0_0.wav"), "wb").write(b"x")
    print("Model preprocessed successfully.")
elif cmd == "extract":
    os.makedirs(os.path.join(exp, "extracted"), exist_ok=True)
    open(os.path.join(exp, "extracted", "0.npy"), "wb").write(b"x")
elif cmd == "train":
    if os.environ.get("FAKE_TRAIN_FAIL"):
        print(f"Training failed for model {opt['--model-name']}. Please check the console logs.")
        sys.exit(0)  # core.py really exits 0 on failure
    total = int(opt["--total-epoch"])
    for e in range(1, total + 1):
        print(f"{opt['--model-name']} | epoch={e} | step={e * 10} | 0:00:01", flush=True)
        if os.environ.get("FAKE_TRAIN_SLOW"):
            time.sleep(0.5)
    torch.save({"weight": {}, "config": [48000], "version": "v2", "f0": 1},
               os.path.join(exp, f"{opt['--model-name']}_{total}e_{total * 10}s.pth"))
    open(os.path.join(exp, opt["--model-name"] + ".index"), "wb").write(b"idx")
elif cmd == "index":
    open(os.path.join(exp, opt["--model-name"] + ".index"), "wb").write(b"idx")
'''


def _wait(job, timeout=60):
    deadline = time.time() + timeout
    while job.status not in ("succeeded", "failed", "cancelled") and time.time() < deadline:
        time.sleep(0.05)
    return job


@pytest.fixture
def stub_training(rvc, registry, tmp_path, monkeypatch):
    from services import rvc_training as tr

    root = _stub_applio(tmp_path / "Applio")
    (root / "core.py").write_text(_FAKE_CORE)
    monkeypatch.setenv("OMNIVOICE_APPLIO_DIR", str(root))
    monkeypatch.setenv("OMNIVOICE_APPLIO_PYTHON", sys.executable)
    monkeypatch.setenv("OMNIVOICE_RVC_TRAINING_DIR", str(tmp_path / "train"))
    monkeypatch.setattr(tr, "_auto_gpu", lambda: "-")
    ds = tmp_path / "dataset"
    ds.mkdir()
    _write_wav(ds / "a.wav")
    tr._reset_for_tests()
    yield tr, ds
    for job in tr.list_jobs():
        tr.cancel_job(job.id)
    tr._reset_for_tests()


def test_training_pipeline_registers_voice(stub_training, registry):
    tr, ds = stub_training
    job = tr.start_training(tr.TrainSettings(name="My Voice", epochs=3, save_every=1),
                            dataset_dir=ds)
    _wait(job)
    assert job.status == "succeeded", list(job.log)[-20:]
    assert job.epoch == 3 and job.progress == 1.0
    voice = registry.get_voice(job.voice_id)
    assert voice["name"] == "My Voice" and voice["has_index"] and voice["source"] == "trained"
    assert any("epoch=2" in line for line in job.log)


def test_training_failure_message_with_exit_zero_fails_job(stub_training, monkeypatch, registry):
    tr, ds = stub_training
    monkeypatch.setenv("FAKE_TRAIN_FAIL", "1")
    job = _wait(tr.start_training(tr.TrainSettings(name="F", epochs=2), dataset_dir=ds))
    assert job.status == "failed" and "Training failed" in job.error
    assert registry.list_voices() == []


def test_training_cancel_and_single_job(stub_training, monkeypatch):
    tr, ds = stub_training
    monkeypatch.setenv("FAKE_TRAIN_SLOW", "1")
    job = tr.start_training(tr.TrainSettings(name="C", epochs=50), dataset_dir=ds)
    with pytest.raises(tr.JobConflict):
        tr.start_training(tr.TrainSettings(name="D", epochs=1), dataset_dir=ds)
    deadline = time.time() + 30
    while job.epoch < 1 and time.time() < deadline:
        time.sleep(0.05)
    tr.cancel_job(job.id)
    assert _wait(job).status == "cancelled"


def test_training_requires_audio(stub_training, tmp_path):
    tr, _ = stub_training
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        tr.start_training(tr.TrainSettings(name="E"), dataset_dir=empty)


def test_setup_commands_mirror_run_install(tmp_path):
    from services import rvc_training as tr

    steps = dict(tr.setup_commands(tmp_path / "applio", uv="uv", git="git", cuda=True))
    assert steps["clone"][:4] == ["git", "clone", "--depth", "1"]
    assert "--python" in steps["venv"] and "3.12" in steps["venv"]
    assert "https://download.pytorch.org/whl/cu128" in steps["requirements"]
    assert steps["prerequisites"][2:4] == ["prerequisites", "--no-pretraineds-hifigan"]


# ── router ───────────────────────────────────────────────────────────────────


@pytest.fixture
def client(rvc, registry, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import core.config as cfg
    import services.watermark as wm
    from api.routers import voice_conversion as vc
    from services import rvc_training as tr

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", str(outputs))
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    marks = []

    async def fake_mark(tensor, sr, *, context, **kw):
        marks.append((context, sr))
        return tensor

    monkeypatch.setattr(wm, "mark_synthetic_async", fake_mark)

    async def run_blocking(fn, *, what, timeout):
        return await asyncio.to_thread(fn)

    monkeypatch.setattr(vc, "_run_blocking", run_blocking)
    svc = rvc.get_service()
    fake = _FakeBackend(sr=40000)
    monkeypatch.setattr(svc, "backend", lambda: fake)
    tr._reset_for_tests()
    app = FastAPI()
    app.include_router(vc.router)
    c = TestClient(app)
    c.fake, c.outputs, c.marks = fake, outputs, marks
    yield c
    tr._reset_for_tests()


def _import(client, tmp_path, name="Hero", index=True):
    pth = _fake_pth(tmp_path / f"{name}.pth")
    files = {"model": (f"{name}.pth", pth.read_bytes(), "application/octet-stream")}
    if index:
        files["index"] = (f"{name}.index", b"faiss", "application/octet-stream")
    return client.post("/voice-conversion/voices/import", data={"name": name}, files=files)


def test_api_status_reports_missing_backend(client, rvc, monkeypatch):
    monkeypatch.setattr(rvc, "rvc_python_available", lambda: False)
    body = client.get("/voice-conversion/status").json()
    assert body["available"] is False and body["applio"]["ready"] is False
    assert body["docs"].endswith("voice-conversion.md")


def test_api_voice_crud(client, tmp_path):
    r = _import(client, tmp_path)
    assert r.status_code == 200, r.text
    vid = r.json()["id"]
    assert r.json()["has_index"] is True and r.json()["sample_rate"] == 48000
    assert [v["id"] for v in client.get("/voice-conversion/voices").json()["voices"]] == [vid]
    assert client.patch(f"/voice-conversion/voices/{vid}", json={"name": "Villain"}).json()[
        "name"] == "Villain"
    assert client.delete(f"/voice-conversion/voices/{vid}").status_code == 200
    assert client.delete(f"/voice-conversion/voices/{vid}").status_code == 404
    assert client.patch("/voice-conversion/voices/..%2Fx", json={"name": "n"}).status_code == 404


def test_api_import_rejects_wrong_extensions(client):
    r = client.post("/voice-conversion/voices/import", data={"name": "x"},
                    files={"model": ("x.zip", b"zz", "application/zip")})
    assert r.status_code == 415
    r = client.post("/voice-conversion/voices/import", data={"name": "x"},
                    files={"model": ("x.pth", b"zz", "application/octet-stream"),
                           "index": ("x.txt", b"zz", "text/plain")})
    assert r.status_code == 415


def test_api_convert_keeps_original_and_marks_output(client, tmp_path):
    vid = _import(client, tmp_path).json()["id"]
    src = _wav_bytes(16000, 0.5)
    r = client.post("/voice-conversion/convert",
                    data={"voice_id": vid, "pitch": "-3", "f0_method": "crepe",
                          "index_rate": "0.6", "protect": "0.45", "rms_mix_rate": "0.3"},
                    files={"audio": ("take 1.wav", src, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_rate"] == 40000 and body["voice_name"] == "Hero"
    assert body["audio_url"] == f"/audio/{body['filename']}"
    assert (client.outputs / body["source_filename"]).read_bytes() == src
    with wave.open(str(client.outputs / body["filename"])) as wf:
        assert wf.getframerate() == 40000
    assert client.marks == [("voice_conversion.convert", 40000)]
    p = client.fake.calls[0][4]
    assert (p.pitch, p.f0_method, p.index_rate, p.protect, p.rms_mix_rate) == \
        (-3, "crepe", 0.6, 0.45, 0.3)
    assert client.fake.calls[0][3].endswith("model.index")
    items = client.get("/voice-conversion/history").json()["items"]
    assert [i["id"] for i in items] == [body["id"]]
    assert client.delete(f"/voice-conversion/history/{body['id']}").status_code == 200
    assert not (client.outputs / body["filename"]).exists()


@pytest.mark.parametrize("field,value", [("pitch", "30"), ("pitch", "-25"),
                                         ("f0_method", "dio"), ("index_rate", "2"),
                                         ("protect", "0.9"), ("filter_radius", "9")])
def test_api_convert_param_validation(client, tmp_path, field, value):
    vid = _import(client, tmp_path).json()["id"]
    r = client.post("/voice-conversion/convert", data={"voice_id": vid, field: value},
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 422


def test_api_convert_unknown_voice_and_unavailable_backend(client, rvc, tmp_path, monkeypatch):
    r = client.post("/voice-conversion/convert", data={"voice_id": "nope-123456"},
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 404
    vid = _import(client, tmp_path).json()["id"]

    def unavailable():
        raise rvc.RVCUnavailable("not installed")

    monkeypatch.setattr(rvc.get_service(), "backend", unavailable)
    r = client.post("/voice-conversion/convert", data={"voice_id": vid},
                    files={"audio": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "rvc_unavailable"
    assert not [p for p in client.outputs.iterdir() if p.name.startswith("vc_")], \
        "a failed conversion must not leave files behind"


def test_api_train_validation_and_job_lifecycle(client, rvc, tmp_path, monkeypatch):
    from services import rvc_training as tr

    assert client.post("/voice-conversion/train", data={"name": "x"}).status_code == 400
    r = client.post("/voice-conversion/train", data={"name": "x", "sample_rate": "44100"},
                    files=[("files", ("a.wav", _wav_bytes(), "audio/wav"))])
    assert r.status_code == 422
    r = client.post("/voice-conversion/train", data={"name": "x"},
                    files=[("files", ("a.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 415
    # No Applio configured → typed 503 with setup docs.
    r = client.post("/voice-conversion/train", data={"name": "x"},
                    files=[("files", ("a.wav", _wav_bytes(), "audio/wav"))])
    assert r.status_code == 503

    real_start = tr.start_training

    def start(settings, **kw):
        return real_start(settings, runner=lambda job: job.log.append("fake run"), **kw)

    monkeypatch.setattr(tr, "start_training", start)
    monkeypatch.setenv("OMNIVOICE_RVC_TRAINING_DIR", str(tmp_path / "train"))
    r = client.post("/voice-conversion/train",
                    data={"name": "Mine", "epochs": "10", "batch_size": "6"},
                    files=[("files", ("a.wav", _wav_bytes(), "audio/wav")),
                           ("files", ("b.flac", _wav_bytes(), "audio/flac"))])
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["settings"]["batch_size"] == 6 and job["settings"]["dataset_files"] == 2
    _wait(tr.get_job(job["id"]))
    status = client.get(f"/voice-conversion/train/{job['id']}").json()
    assert status["status"] == "succeeded" and "fake run" in status["log_tail"]
    assert client.post(f"/voice-conversion/train/{job['id']}/cancel").json()["status"] == \
        "succeeded"
    assert client.get("/voice-conversion/train/missing").status_code == 404
    assert client.get("/voice-conversion/jobs").json()["jobs"][0]["id"] == job["id"]


def test_api_setup_requires_explicit_confirmation(client):
    # Setup clones a repo and pip-installs: admin-gated (review item 1e), so a
    # non-loopback client is refused before the handler runs.
    assert client.post("/voice-conversion/setup", json={}).status_code == 403
    from fastapi.testclient import TestClient

    local = TestClient(client.app, client=("127.0.0.1", 50000))
    r = local.post("/voice-conversion/setup", json={})
    assert r.status_code == 400 and "confirm" in r.json()["detail"]


def test_api_settings_rejects_unauthorized_applio_path(client):
    r = client.put("/voice-conversion/settings",
                   json={"applio_dir_authorization": "0" * 64})
    assert r.status_code == 403


def test_generate_rvc_post_step_validation(rvc, registry, tmp_path):
    from fastapi import HTTPException

    from api.routers.generation import _rvc_post_step_or_404

    assert _rvc_post_step_or_404(None, pitch=0, f0_method="rmvpe", index_rate=0.5,
                                 protect=0.3) is None
    with pytest.raises(HTTPException) as exc:
        _rvc_post_step_or_404("missing-000000", pitch=0, f0_method="rmvpe",
                              index_rate=0.5, protect=0.3)
    assert exc.value.status_code == 404
    v = registry.import_from_paths("A", _fake_pth(tmp_path / "a.pth"))
    vid, params = _rvc_post_step_or_404(v["id"], pitch=2, f0_method="crepe",
                                        index_rate=0.5, protect=0.3)
    assert vid == v["id"] and params.pitch == 2
