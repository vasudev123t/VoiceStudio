"""Review item 2: idle/explicit release of GPU holders."""
import sys
import time

import pytest

from services import gpu_memory, rvc


class _FakeProc:
    def __init__(self):
        self.pid = 4242
        self.killed = False
        self.waited = False

    def poll(self):
        return 0 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        self.killed = True
        return 0


def _backend(tmp_path):
    b = rvc.ApplioBackend(tmp_path, tmp_path / "python")
    b._proc = _FakeProc()
    sent = []

    def fake_send(payload, timeout):
        sent.append(payload["op"])
        return {"ok": True}

    b._send = fake_send
    return b, sent


def test_idle_worker_is_reaped_gracefully(tmp_path):
    b, sent = _backend(tmp_path)
    proc = b._proc
    b._last_used = time.monotonic()
    assert b.reap_if_idle(300) is False and b.running()
    b._last_used = time.monotonic() - 301
    assert b.reap_if_idle(300) is True
    assert sent == ["exit"] and proc.waited and not b.running()


def test_reaper_never_interrupts_a_conversion(tmp_path):
    b, sent = _backend(tmp_path)
    b._last_used = time.monotonic() - 10_000
    b._lock.acquire()
    try:
        assert b.reap_if_idle(1) is False
    finally:
        b._lock.release()
    assert b.running() and sent == []


def test_zero_idle_timeout_disables_reaper(tmp_path, monkeypatch):
    b, _ = _backend(tmp_path)
    b._last_used = time.monotonic() - 10_000
    monkeypatch.setenv("OMNIVOICE_RVC_IDLE_S", "0")
    assert rvc.idle_timeout_s() == 0
    assert b.reap_if_idle(rvc.idle_timeout_s()) is False


def test_worker_env_drops_credentials():
    env = rvc.worker_env({"PATH": "x", "HF_TOKEN": "s", "OMNIVOICE_API_KEY": "s",
                          "OMNIVOICE_SHARE_PIN": "1", "CUDA_VISIBLE_DEVICES": "0",
                          "PYTHONPATH": "y", "HF_HOME": "D:/hf"})
    assert env == {"PATH": "x", "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "D:/hf"}


def test_worker_log_is_rotated(tmp_path):
    log = tmp_path / "w.log"
    log.write_bytes(b"x" * 100)
    rvc._rotate_log(log, max_bytes=10)
    assert not log.exists() and (tmp_path / "w.log.1").stat().st_size == 100


def test_service_reports_and_releases_worker(tmp_path, monkeypatch):
    svc = rvc.VoiceConversionService()
    b, sent = _backend(tmp_path)
    b._loaded = str(tmp_path / "hero" / "model.pth")
    svc._backend = b
    info = svc.loaded_info()
    assert info["backend"] == "applio" and info["pid"] == 4242
    assert svc.release() is True and sent == ["exit"] and svc.loaded_info() is None
    assert svc.release() is False


@pytest.mark.asyncio
async def test_lifecycle_lists_and_unloads_rvc_worker(tmp_path, monkeypatch):
    from services import model_lifecycle

    svc = rvc.VoiceConversionService()
    b, sent = _backend(tmp_path)
    svc._backend = b
    monkeypatch.setattr(rvc, "_service", svc)
    ids = [m["id"] for m in model_lifecycle.list_loaded()["models"]]
    assert "rvc-worker" in ids
    res = await model_lifecycle.unload("rvc-worker")
    assert res["success"] is True and sent == ["exit"]
    res = await model_lifecycle.unload("rvc-worker")
    assert res["success"] is False


@pytest.mark.asyncio
async def test_aligners_are_listed_and_released(monkeypatch):
    import services.asr_backend as ab
    from services import model_lifecycle

    monkeypatch.setattr(ab, "_ALIGN_CACHE", {("en", "cuda"): (object(), {}), ("kn", "cuda"): None})
    monkeypatch.setattr(ab, "_ALIGN_LAST_USED", time.monotonic())
    assert ab.release_align_cache(600) == 0  # recently used
    entry = next(m for m in model_lifecycle.list_loaded()["models"] if m["id"] == "asr-aligners")
    assert "en" in entry["name"]
    res = await model_lifecycle.unload("asr-aligners")
    assert res["success"] and res["count"] == 1
    assert ab._ALIGN_CACHE == {("kn", "cuda"): None}


def test_windows_counter_parsing():
    text = ("\\\\msi\\gpu process memory(pid_41124_luid_0x00000000_0x0000d1b1_phys_0)"
            "\\dedicated usage\t2228224000\n"
            "\\\\msi\\gpu process memory(pid_41124_luid_0x00000000_0x0000d1b1_phys_1)"
            "\\dedicated usage\t0\n"
            "\\\\msi\\gpu process memory(pid_26236_luid_0x00000000_0x0000d1b1_phys_0)"
            "\\dedicated usage\t1538260992\n")
    usage = gpu_memory.parse_windows_counters(text)
    assert round(usage[41124]) == 2125 and round(usage[26236]) == 1467


def test_release_gpu_holders_runs_every_step(monkeypatch, tmp_path):
    from services import model_manager as mm

    calls = []
    monkeypatch.setattr(mm, "unload_shared_model", lambda: calls.append("tts") or True)
    monkeypatch.setattr(mm, "free_vram", lambda: calls.append("free"))
    svc = rvc.VoiceConversionService()
    b, sent = _backend(tmp_path)
    svc._backend = b
    monkeypatch.setattr(rvc, "_service", svc)
    released = gpu_memory.release_gpu_holders(reason="test")
    assert "tts" in released and "rvc-worker" in released
    assert calls[-1] == "free" and sent == ["exit"]


def test_training_frees_gpu_first(monkeypatch):
    from services import rvc_training as tr

    seen = []
    monkeypatch.setattr(gpu_memory, "release_gpu_holders",
                        lambda reason: seen.append(reason) or ["tts", "rvc-worker"])
    job = tr.Job(id="j", kind="train", name="x")
    tr._free_gpu_for_job(job)
    assert seen == ["rvc train job"]
    assert "tts, rvc-worker" in job.log[-1]
    monkeypatch.setenv("OMNIVOICE_RVC_TRAIN_KEEP_GPU", "1")
    tr._free_gpu_for_job(job)
    assert seen == ["rvc train job"]
