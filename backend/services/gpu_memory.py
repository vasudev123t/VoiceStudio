"""Measure and free GPU memory held by VoiceStudio (review item 2).

``snapshot()`` answers "who is holding my 8 GB?" without initialising CUDA in
the backend: device totals from ``nvidia-smi``, per-process usage from
``nvidia-smi --query-compute-apps`` and, on Windows (WDDM reports ``[N/A]``
per process there), from the ``GPU Process Memory`` performance counters.
Processes VoiceStudio owns are labelled (backend, Applio worker, sidecars).

``release_gpu_holders()`` is the one "Free GPU" path: shared TTS model,
engine sidecars, dictation ASR, wav2vec2 aligners, the Applio worker and the
in-process engines. Each step is best-effort and skipped while leased.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_COUNTER_RE = re.compile(r"pid_(\d+)_.*?\)\\dedicated usage\s*$", re.I)


def _run(argv: list[str], timeout: float = 8.0) -> Optional[str]:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _num(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return None


def _nvidia_smi() -> Optional[str]:
    try:
        from core.nvidia_smi import find_nvidia_smi

        return find_nvidia_smi()
    except Exception:  # noqa: BLE001
        return None


def device_totals(smi: Optional[str] = None) -> list[dict]:
    smi = smi or _nvidia_smi()
    if not smi:
        return []
    out = _run([smi, "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({"index": int(_num(parts[0]) or 0), "name": parts[1],
                         "total_mb": _num(parts[2]), "used_mb": _num(parts[3]),
                         "free_mb": _num(parts[4])})
    return gpus


def _smi_processes(smi: str) -> dict[int, float]:
    out = _run([smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    usage: dict[int, float] = {}
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            mb = _num(parts[1])
            if mb is not None:
                usage[int(parts[0])] = usage.get(int(parts[0]), 0.0) + mb
    return usage


def parse_windows_counters(text: str) -> dict[int, float]:
    """``Get-Counter`` "Path<TAB>Bytes" lines → {pid: MiB} (summed over adapters)."""
    usage: dict[int, float] = {}
    for line in text.splitlines():
        path, _, value = line.rpartition("\t")
        m = _COUNTER_RE.search(path.strip())
        val = _num(value)
        if m and val is not None and val > 0:
            pid = int(m.group(1))
            usage[pid] = usage.get(pid, 0.0) + val / (1024 * 1024)
    return usage


def _windows_processes() -> dict[int, float]:
    script = ("(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction "
              "SilentlyContinue).CounterSamples | ForEach-Object { $_.Path + \"`t\" + "
              "[string][int64]$_.CookedValue }")
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=20)
    return parse_windows_counters(out or "")


def owned_processes() -> dict[int, str]:
    """PIDs VoiceStudio owns → role label."""
    roles = {os.getpid(): "backend"}
    try:
        from services import rvc

        info = rvc.get_service().loaded_info() if rvc._service is not None else None
        if info and info.get("pid") and info["backend"] == "applio":
            roles[int(info["pid"])] = "rvc-worker"
    except Exception:  # noqa: BLE001
        pass
    try:
        from services.subprocess_backend import list_live_sidecars

        for s in list_live_sidecars():
            if s.get("pid"):
                roles[int(s["pid"])] = f"sidecar:{s['id']}"
    except Exception:  # noqa: BLE001
        pass
    return roles


def snapshot(*, include_other: bool = True) -> dict:
    smi = _nvidia_smi()
    gpus = device_totals(smi)
    usage: dict[int, float] = _smi_processes(smi) if smi else {}
    source = "nvidia-smi" if usage else None
    if not usage and sys.platform == "win32":
        usage = _windows_processes()
        source = "windows-perf-counters" if usage else None
    roles = owned_processes()
    procs = []
    for pid, mb in sorted(usage.items(), key=lambda kv: -kv[1]):
        role = roles.get(pid)
        if role is None and not include_other:
            continue
        procs.append({"pid": pid, "role": role or "other", "used_mb": round(mb, 1)})
    owned_mb = round(sum(p["used_mb"] for p in procs if p["role"] != "other"), 1)
    return {"gpus": gpus, "processes": procs, "voicestudio_mb": owned_mb,
            "source": source, "available": bool(gpus)}


def release_gpu_holders(*, reason: str = "manual") -> list[str]:
    """Release every model/process VoiceStudio holds on the GPU (sync)."""
    released: list[str] = []

    def step(name: str, fn) -> None:
        try:
            if fn():
                released.append(name)
        except Exception:  # noqa: BLE001 — one stuck holder must not stop the rest
            logger.warning("GPU release step %s failed (%s)", name, reason, exc_info=True)

    from services import model_manager as mm

    step("tts", mm.unload_shared_model)

    def _sidecars():
        from services.subprocess_backend import unload_all_sidecars

        return unload_all_sidecars() > 0

    def _capture():
        from services.asr_backend import release_idle_capture_backend

        return release_idle_capture_backend(0.0)

    def _aligners():
        from services.asr_backend import release_align_cache

        return release_align_cache(0.0) > 0

    def _rvc():
        from services import rvc

        return rvc._service is not None and rvc.get_service().release()

    def _engines():
        from api.routers.engines import _ENGINE_INSTANCES
        from services.tts_backend import OmniVoiceBackend

        any_held = False
        for cls, inst in list(_ENGINE_INSTANCES.items()):
            if cls is OmniVoiceBackend:
                continue
            if any(getattr(inst, a, None) is not None
                   for a in getattr(inst, "_MODEL_ATTRS", ("_model", "_tts"))):
                inst.unload()
                any_held = True
        return any_held

    def _diar():
        if getattr(mm, "_diar_pipeline", None) is None:
            return False
        mm._diar_pipeline = None
        return True

    for name, fn in (("sidecars", _sidecars), ("capture-asr", _capture),
                     ("asr-aligners", _aligners), ("rvc-worker", _rvc),
                     ("engines", _engines), ("diarization", _diar)):
        step(name, fn)
    try:
        mm.free_vram()
    except Exception:  # noqa: BLE001
        pass
    logger.info("Released GPU holders (%s): %s", reason, ", ".join(released) or "none")
    return released
