"""RVC (Retrieval-based Voice Conversion) service — the "Voice conversion" feature.

Speech-to-speech conversion that keeps the performer's prosody, timing, laughs
and breaths and swaps only the timbre for a trained RVC voice model.

Converter backends (both optional, both lazily loaded, never a hard dependency):

* **Applio** (preferred) — https://github.com/IAHispano/Applio (MIT). Used when
  an Applio checkout + its Python env is configured (``OMNIVOICE_APPLIO_DIR``,
  the ``voice_conversion.applio_dir`` setting, or the managed install under
  ``<data>/tools/applio``). Applio runs in ITS OWN interpreter as a persistent
  worker process (``services/rvc_applio_worker.py``) so its torch/faiss pins
  never touch the backend env; the worker keeps the loaded model warm and
  frees VRAM on a voice switch.
* **rvc-python** — ``rvc_python.infer.RVCInference`` imported in-process when
  the package is installed in the backend env.

Device: ``cuda:0`` when torch sees CUDA, else ``mps``, else ``cpu``; override
with ``OMNIVOICE_RVC_DEVICE`` or the ``voice_conversion.device`` setting.

Legacy dubbing hook (unchanged contract, still env-driven and off by default):
  OMNIVOICE_RVC_ENABLED=1, OMNIVOICE_RVC_MODEL=<.pth>, OMNIVOICE_RVC_INDEX=<.index>,
  OMNIVOICE_RVC_PITCH, OMNIVOICE_RVC_INDEX_RATE, OMNIVOICE_RVC_FILTER_RAD,
  OMNIVOICE_RVC_PROTECT, OMNIVOICE_RVC_F0_METHOD (rmvpe|crepe|fcpe|harvest|pm).
  ``OMNIVOICE_RVC_VOICE`` may name a registered voice id instead of a model path.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("omnivoice.rvc")

#: Every pitch extractor the API accepts. Each backend supports a subset
#: (verified against the sources: Applio's ``core.py infer --f0-method``
#: choices; rvc-python's ``VC.vc_single`` f0 methods).
F0_METHODS = ("rmvpe", "crepe", "fcpe", "harvest", "pm")
BACKEND_F0_METHODS = {
    "applio": ("rmvpe", "crepe", "fcpe"),
    "rvc-python": ("rmvpe", "crepe", "harvest", "pm"),
}
#: Parameters a backend actually honours (others are accepted and ignored).
BACKEND_PARAMS = {
    "applio": ("pitch", "f0_method", "index_rate", "protect", "rms_mix_rate",
               "clean_audio", "f0_autotune"),
    "rvc-python": ("pitch", "f0_method", "index_rate", "protect", "rms_mix_rate",
                   "filter_radius"),
}

CONVERT_TIMEOUT_S = float(os.environ.get("OMNIVOICE_RVC_TIMEOUT_S", "900"))
WORKER_START_TIMEOUT_S = float(os.environ.get("OMNIVOICE_RVC_WORKER_START_TIMEOUT_S", "180"))


class RVCUnavailable(RuntimeError):
    """No converter backend is installed/configured."""


class RVCConversionError(RuntimeError):
    """The converter ran but failed."""


# ── Parameters ───────────────────────────────────────────────────────────────


@dataclass
class ConversionParams:
    pitch: int = 0                 # semitones, -24..24
    f0_method: str = "rmvpe"
    index_rate: float = 0.75       # 0..1 — how strongly the index pulls timbre
    protect: float = 0.33          # 0..0.5 — keeps breaths / voiceless consonants (0.5 = off)
    filter_radius: int = 3         # 0..7 — median filter on pitch (rvc-python)
    #: 0..1 — Applio's "volume envelope". 0 keeps the performer's (input)
    #: loudness contour, 1 replaces it with the converted audio's own. 0.1
    #: keeps the micro-dynamics of an acted line (breaths, laughs, emphasis).
    rms_mix_rate: float = 0.1
    clean_audio: bool = False      # Applio noise cleanup on the input
    clean_strength: float = 12.0
    f0_autotune: bool = False
    embedder_model: str = "contentvec"

    def validate(self) -> "ConversionParams":
        if not isinstance(self.pitch, int) or not -24 <= self.pitch <= 24:
            raise ValueError("pitch must be an integer between -24 and 24 semitones.")
        if self.f0_method not in F0_METHODS:
            raise ValueError(f"f0_method must be one of: {', '.join(F0_METHODS)}.")
        for name, lo, hi in (("index_rate", 0.0, 1.0), ("protect", 0.0, 0.5),
                             ("rms_mix_rate", 0.0, 1.0), ("clean_strength", 1.0, 30.0)):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not lo <= float(value) <= hi:
                raise ValueError(f"{name} must be between {lo:g} and {hi:g}.")
        if not isinstance(self.filter_radius, int) or not 0 <= self.filter_radius <= 7:
            raise ValueError("filter_radius must be an integer between 0 and 7.")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# ── Configuration ────────────────────────────────────────────────────────────


def _pref(key: str, default=None):
    try:
        from core import prefs

        return prefs.get(key, default)
    except Exception:  # noqa: BLE001 — prefs unavailable (stand-alone use)
        return default


def select_device(preference: Optional[str] = None) -> str:
    """``cuda:0`` > ``mps`` > ``cpu`` unless explicitly overridden."""
    pref = (preference or os.environ.get("OMNIVOICE_RVC_DEVICE")
            or _pref("voice_conversion.device") or "auto")
    pref = str(pref).strip().lower()
    if pref and pref != "auto":
        return "cuda:0" if pref == "cuda" else pref
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def device_override() -> Optional[str]:
    """The explicit device choice, or None for auto-detect."""
    pref = (os.environ.get("OMNIVOICE_RVC_DEVICE") or _pref("voice_conversion.device") or "")
    pref = str(pref).strip().lower()
    return None if pref in ("", "auto") else ("cuda:0" if pref == "cuda" else pref)


def managed_applio_dir() -> Path:
    from core.config import DATA_DIR

    return Path(DATA_DIR) / "tools" / "applio"


def applio_dir() -> Optional[Path]:
    """Configured Applio checkout (must contain ``core.py`` and ``rvc/``)."""
    candidates = [
        os.environ.get("OMNIVOICE_APPLIO_DIR", "").strip(),
        str(_pref("voice_conversion.applio_dir") or "").strip(),
    ]
    for raw in candidates:
        if raw:
            path = Path(os.path.expanduser(raw))
            return path if _looks_like_applio(path) else None
    managed = managed_applio_dir()
    return managed if _looks_like_applio(managed) else None


def configured_applio_dir_raw() -> Optional[str]:
    return (os.environ.get("OMNIVOICE_APPLIO_DIR", "").strip()
            or str(_pref("voice_conversion.applio_dir") or "").strip() or None)


def _looks_like_applio(path: Path) -> bool:
    return (path / "core.py").is_file() and (path / "rvc" / "infer" / "infer.py").is_file()


def applio_python(root: Optional[Path] = None) -> Optional[Path]:
    """Applio's interpreter: ``env\\python.exe`` (run-install.bat, conda) or
    ``.venv`` (run-install.sh / uv), or ``OMNIVOICE_APPLIO_PYTHON``."""
    explicit = os.environ.get("OMNIVOICE_APPLIO_PYTHON", "").strip() or str(
        _pref("voice_conversion.applio_python") or "").strip()
    if explicit:
        path = Path(os.path.expanduser(explicit))
        return path if path.is_file() else None
    root = root or applio_dir()
    if root is None:
        return None
    for rel in ("env/python.exe", ".venv/Scripts/python.exe", "env/Scripts/python.exe",
                ".venv/bin/python", "env/bin/python"):
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def rvc_python_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("rvc_python") is not None
    except (ImportError, ValueError):
        return False


# ── FFmpeg used by Applio's "light" noise cleanup ───────────────────────────
#
# Applio's ``rvc/lib/tools/audio_cleanup.py::_ffmpeg()`` prefers an
# ``ffmpeg.exe`` in the Applio root over PATH. The Windows bundle ships FFmpeg
# 4.3.2 (2021), whose ``afftdn`` has no ``gs`` (gain smoothing) option, so
# "Light noise cleanup" fails with ``Option 'gs' not found``. VoiceStudio probes
# every candidate, hands the first one that supports ``afftdn gs`` to the worker
# (``OMNIVOICE_RVC_FFMPEG``; the worker patches ``_ffmpeg``), and — when none
# does — runs a gs-free ``afftdn`` pre-clean itself instead of letting Applio
# fail. Nothing in the Applio checkout is modified.

_FFMPEG_PROBES: dict[str, dict] = {}
_FFMPEG_PROBE_LOCK = threading.Lock()
_GS_OPTION_RE = re.compile(r"^\s+gs\s", re.M)


def probe_ffmpeg(path: str) -> dict:
    """``{"path", "version", "supports_gs", "ok"}`` for one ffmpeg binary (cached)."""
    key = os.path.normcase(os.path.abspath(path))
    with _FFMPEG_PROBE_LOCK:
        if key in _FFMPEG_PROBES:
            return dict(_FFMPEG_PROBES[key])
    info = {"path": str(path), "version": None, "supports_gs": False, "ok": False}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        ver = subprocess.run([str(path), "-hide_banner", "-version"], capture_output=True,
                             timeout=15, creationflags=flags)
        first = (ver.stdout or b"").decode("utf-8", "replace").splitlines()[:1]
        if first:
            match = re.search(r"ffmpeg version (\S+)", first[0])
            info["version"] = match.group(1) if match else first[0][:80]
        helptext = subprocess.run([str(path), "-hide_banner", "-h", "filter=afftdn"],
                                  capture_output=True, timeout=15, creationflags=flags)
        out = (helptext.stdout or b"").decode("utf-8", "replace")
        info["ok"] = ver.returncode == 0
        info["supports_gs"] = bool(_GS_OPTION_RE.search(out))
    except (OSError, subprocess.SubprocessError):
        pass
    with _FFMPEG_PROBE_LOCK:
        _FFMPEG_PROBES[key] = dict(info)
    return info


def _ffmpeg_candidates(root: Optional[Path]) -> list[str]:
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    seen: list[str] = []

    def add(path) -> None:
        if path and os.path.isfile(str(path)):
            norm = os.path.normcase(os.path.abspath(str(path)))
            if norm not in (os.path.normcase(os.path.abspath(p)) for p in seen):
                seen.append(str(path))

    add(os.environ.get("OMNIVOICE_RVC_FFMPEG", "").strip() or None)
    add(shutil.which(name))
    try:
        from services.ffmpeg_utils import find_ffmpeg

        add(find_ffmpeg())
    except Exception:  # noqa: BLE001 — stand-alone use
        pass
    if root is not None:
        add(Path(root) / name)
    return seen


def cleanup_ffmpeg_status(root: Optional[Path] = None) -> dict:
    """Which ffmpeg Applio's light cleanup will use, and whether it can.

    ``applio_bundled`` describes the binary Applio would pick on its own.
    ``selected`` is what VoiceStudio passes to the worker (``supports_gs``
    False means VoiceStudio pre-cleans with a gs-free filter instead).
    """
    root = root if root is not None else applio_dir()
    candidates = [probe_ffmpeg(p) for p in _ffmpeg_candidates(root)]
    bundled = None
    if root is not None:
        exe = Path(root) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if exe.is_file():
            bundled = probe_ffmpeg(str(exe))
    selected = next((c for c in candidates if c["ok"] and c["supports_gs"]), None)
    return {
        "selected": selected,
        "applio_bundled": bundled,
        "candidates": candidates,
        "fallback": None if selected else "voicestudio_afftdn",
    }


def preclean_audio(input_path: str, output_path: str, strength: float = 12.0) -> str:
    """gs-free ``afftdn`` denoise with VoiceStudio's own ffmpeg (fallback path)."""
    candidates = _ffmpeg_candidates(None)
    if not candidates:
        raise RVCConversionError("Noise cleanup needs FFmpeg; none was found.")
    nr = max(1.0, min(30.0, float(strength)))
    cmd = [candidates[0], "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
           "-af", f"afftdn=nr={nr:g}:nf=-50:tn=1", "-ac", "1", output_path]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    proc = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=flags)
    if proc.returncode != 0 or not os.path.isfile(output_path):
        tail = (proc.stderr or b"").decode("utf-8", "replace")[-300:]
        raise RVCConversionError(f"Noise cleanup failed: {tail}")
    return output_path


# ── Backends ─────────────────────────────────────────────────────────────────


class ConverterBackend:
    name = "base"

    def convert(self, input_path: str, output_path: str, model_path: str,
                index_path: Optional[str], params: ConversionParams) -> int:
        """Convert ``input_path`` → ``output_path`` (WAV). Returns the sample rate."""
        raise NotImplementedError

    def unload(self) -> None:
        """Drop the loaded model and free accelerator memory."""

    def close(self) -> None:
        self.unload()

    @property
    def device(self) -> str:
        return select_device()


class RvcPythonBackend(ConverterBackend):
    """In-process ``rvc_python.infer.RVCInference`` (rvc-python ≥ 0.1)."""

    name = "rvc-python"

    def __init__(self, device: Optional[str] = None):
        self._device = device or select_device()
        self._engine = None
        self._loaded: Optional[tuple[str, str]] = None

    @property
    def device(self) -> str:
        return self._device

    def _ensure_engine(self):
        if self._engine is None:
            try:
                from rvc_python.infer import RVCInference  # type: ignore
            except ImportError as exc:
                raise RVCUnavailable("rvc-python is not installed in the backend env.") from exc
            # rvc-python spells CPU as "cpu:0".
            device = "cpu:0" if self._device == "cpu" else self._device
            self._engine = RVCInference(device=device)
        return self._engine

    def convert(self, input_path, output_path, model_path, index_path, params):
        engine = self._ensure_engine()
        key = (model_path, index_path or "")
        if self._loaded != key:
            if self._loaded is not None:
                self.unload()
                engine = self._ensure_engine()
            # NB: the real API is load_model(path, version, index_path) and
            # set_params(f0up_key=…, f0method=…); the pre-refactor hook called a
            # non-existent set_index_path() and misspelled both keys.
            engine.load_model(model_path, index_path=index_path or "")
            self._loaded = key
        engine.set_params(
            f0up_key=params.pitch,
            f0method=params.f0_method,
            index_rate=params.index_rate if index_path else 0.0,
            filter_radius=params.filter_radius,
            rms_mix_rate=params.rms_mix_rate,
            protect=params.protect,
        )
        engine.infer_file(input_path, output_path)
        try:
            return int(engine.vc.tgt_sr)
        except Exception:  # noqa: BLE001
            return _wav_rate(output_path)

    def unload(self) -> None:
        if self._engine is not None and self._loaded is not None:
            try:
                self._engine.unload_model()
            except Exception:  # noqa: BLE001
                logger.debug("rvc-python unload failed", exc_info=True)
        self._loaded = None
        _empty_cuda_cache()

    def close(self) -> None:
        self.unload()
        self._engine = None
        _empty_cuda_cache()


#: Environment variables never handed to the Applio worker: credentials have
#: no business in a third-party inference process (it runs code from a cloned
#: repo and its pip deps).
_WORKER_ENV_DENY = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|_PIN$|CREDENTIAL)", re.I)
_WORKER_LOG_MAX_BYTES = 5 * 1024 * 1024


def worker_env(base: Optional[dict] = None) -> dict:
    env = dict(os.environ if base is None else base)
    for name in list(env):
        if _WORKER_ENV_DENY.search(name):
            env.pop(name, None)
    env.pop("PYTHONPATH", None)  # never leak the backend's site into Applio
    return env


def _rotate_log(path: Path, max_bytes: int = _WORKER_LOG_MAX_BYTES) -> None:
    """Keep the worker log bounded: one ``.1`` generation, then start fresh."""
    try:
        if path.is_file() and path.stat().st_size > max_bytes:
            old = path.with_suffix(path.suffix + ".1")
            if old.exists():
                old.unlink()
            path.rename(old)
    except OSError:
        logger.debug("worker log rotation failed", exc_info=True)


def idle_timeout_s() -> float:
    """Seconds an idle Applio worker may keep its process (and CUDA context).

    ``OMNIVOICE_RVC_IDLE_S``; default 300 s. ``0`` disables the reaper.
    """
    try:
        return max(0.0, float(os.environ.get("OMNIVOICE_RVC_IDLE_S", "300")))
    except ValueError:
        return 300.0


class ApplioBackend(ConverterBackend):
    """Applio ``VoiceConverter`` in a persistent worker process.

    The worker owns a CUDA context (~0.5–1.5 GB on an 8 GB laptop GPU) for as
    long as it lives, so it is reaped after :func:`idle_timeout_s` of
    inactivity and respawned on the next conversion.
    """

    name = "applio"

    def __init__(self, root: Path, python: Path, device: Optional[str] = None):
        self.root = Path(root)
        self.python = Path(python)
        self._device_override = device
        self._proc = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lock = threading.Lock()
        self._loaded: Optional[str] = None
        self.worker_device: Optional[str] = None
        self.worker_gpu: Optional[str] = None
        self._last_used = time.monotonic()

    @property
    def device(self) -> str:
        return self.worker_device or self._device_override or "auto"

    def _log_path(self) -> Path:
        from core.config import DATA_DIR

        path = Path(DATA_DIR) / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path / "rvc_applio_worker.log"

    def _spawn(self):
        worker = Path(__file__).with_name("rvc_applio_worker.py")
        env = worker_env()
        env["OMNIVOICE_RVC_WORKER_DEVICE"] = self._device_override or "auto"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        ffmpeg = cleanup_ffmpeg_status(self.root).get("selected")
        if ffmpeg:
            env["OMNIVOICE_RVC_FFMPEG"] = ffmpeg["path"]
        else:
            env.pop("OMNIVOICE_RVC_FFMPEG", None)
        _rotate_log(self._log_path())
        log = open(self._log_path(), "ab")
        argv = [str(self.python), str(worker)]
        try:
            from core.contained_subprocess import spawn_owned

            proc = spawn_owned(argv, cwd=str(self.root), env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=log)
        except Exception:  # noqa: BLE001 — fall back to a plain child
            proc = subprocess.Popen(argv, cwd=str(self.root), env=env, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=log)
        finally:
            log.close()
        self._lines = queue.Queue()

        def _pump(stream, sink):
            for raw in iter(stream.readline, b""):
                sink.put(raw.decode("utf-8", "replace"))
            sink.put(None)

        threading.Thread(target=_pump, args=(proc.stdout, self._lines), daemon=True).start()
        self._proc = proc

    def _request(self, payload: dict, timeout: float) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
            hello = self._send({"op": "ping"}, WORKER_START_TIMEOUT_S)
            self.worker_device, self.worker_gpu = hello.get("device"), hello.get("gpu")
        return self._send(payload, timeout)

    def _send(self, payload: dict, timeout: float) -> dict:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._kill()
            raise RVCConversionError(f"Applio worker exited unexpectedly ({exc}). "
                                     f"See {self._log_path()}.") from exc
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise RVCConversionError("Applio worker timed out.")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                self._kill()
                raise RVCConversionError(
                    f"Applio worker exited unexpectedly. See {self._log_path()}.")
            line = line.strip()
            if not line.startswith("{"):
                continue
            reply = json.loads(line)
            if reply.get("fatal"):
                self._kill()
                raise RVCUnavailable(f"Applio could not start: {reply.get('error')}")
            if not reply.get("ok"):
                raise RVCConversionError(reply.get("error") or "Applio conversion failed.")
            return reply

    def convert(self, input_path, output_path, model_path, index_path, params):
        if params.f0_method not in BACKEND_F0_METHODS["applio"]:
            raise ValueError(f"Applio does not support f0_method={params.f0_method!r}; use "
                             f"{', '.join(BACKEND_F0_METHODS['applio'])}.")
        if params.clean_audio and not cleanup_ffmpeg_status(self.root).get("selected"):
            # No ffmpeg with ``afftdn gs``: clean here instead of letting
            # Applio fail with "Option 'gs' not found".
            cleaned = str(Path(output_path).with_name(Path(output_path).stem + "_preclean.wav"))
            input_path = preclean_audio(input_path, cleaned, params.clean_strength)
            params = ConversionParams(**{**params.to_dict(), "clean_audio": False})
        with self._lock:
            if self._loaded and self._loaded != model_path:
                self._request({"op": "unload"}, 60)  # free VRAM before the next voice
            reply = self._request({
                "op": "convert", "input": input_path, "output": output_path,
                "model": model_path, "index": index_path or "",
                "params": {**params.to_dict(),
                           "index_rate": params.index_rate if index_path else 0.0},
            }, CONVERT_TIMEOUT_S)
            self._loaded = model_path
            self._last_used = time.monotonic()
            return int(reply.get("sample_rate") or _wav_rate(output_path))

    def unload(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._loaded:
                try:
                    self._send({"op": "unload"}, 60)
                except Exception:  # noqa: BLE001
                    logger.debug("Applio unload failed", exc_info=True)
            self._loaded = None

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        self._loaded = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self.running() else None

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used

    def _stop(self) -> None:
        """Graceful exit (worker frees CUDA, then the process ends), else kill."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                self._send({"op": "exit"}, 15)
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 — fall through to a hard kill
                logger.debug("Applio worker graceful exit failed", exc_info=True)
        self._kill()

    def close(self) -> None:
        with self._lock:
            self._stop()

    def reap_if_idle(self, idle_s: float) -> bool:
        """Stop the worker process once idle for ``idle_s``. Never blocks a
        running conversion: a busy lock means "in use", so nothing happens."""
        if idle_s <= 0 or not self.running() or self.idle_seconds() < idle_s:
            return False
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if not self.running() or self.idle_seconds() < idle_s:
                return False
            logger.info("Applio worker idle for %.0fs; stopping it to free the GPU.",
                        self.idle_seconds())
            self._stop()
            return True
        finally:
            self._lock.release()


def _safe_cleanup_status(root: Path) -> Optional[dict]:
    try:
        info = cleanup_ffmpeg_status(root)
    except Exception:  # noqa: BLE001 — status must never fail
        return None
    sel = info.get("selected") or {}
    bundled = info.get("applio_bundled") or {}
    return {
        "path": sel.get("path"),
        "version": sel.get("version"),
        "supports_gs": bool(sel),
        "applio_bundled_version": bundled.get("version"),
        "applio_bundled_supports_gs": bundled.get("supports_gs"),
        "fallback": info.get("fallback"),
    }


def _wav_rate(path: str) -> int:
    try:
        import soundfile as sf

        return int(sf.info(path).samplerate)
    except Exception:  # noqa: BLE001
        import wave

        with wave.open(path, "rb") as wf:
            return wf.getframerate()


def _empty_cuda_cache() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ── Service ──────────────────────────────────────────────────────────────────


@dataclass
class ConversionResult:
    output_path: str
    sample_rate: int
    backend: str
    device: str
    elapsed_s: float
    params: dict = field(default_factory=dict)


class VoiceConversionService:
    """Picks a backend, caches it, and converts files with registered voices."""

    def __init__(self):
        self._backend: Optional[ConverterBackend] = None
        self._backend_key: Optional[tuple] = None
        self._lock = threading.RLock()

    def _desired_key(self) -> Optional[tuple]:
        root = applio_dir()
        py = applio_python(root) if root else None
        if root and py:
            return ("applio", str(root), str(py), device_override())
        if rvc_python_available():
            return ("rvc-python", device_override())
        return None

    def backend(self) -> ConverterBackend:
        with self._lock:
            key = self._desired_key()
            if key is None:
                raise RVCUnavailable(
                    "No voice-conversion backend found. Install Applio (Voice conversion → "
                    "Setup) or `pip install rvc-python` into the backend environment.")
            if self._backend is not None and self._backend_key == key:
                return self._backend
            if self._backend is not None:
                self._backend.close()
            if key[0] == "applio":
                self._backend = ApplioBackend(Path(key[1]), Path(key[2]), key[3])
            else:
                self._backend = RvcPythonBackend(key[1])
            self._backend_key = key
            return self._backend

    def status(self) -> dict:
        root = applio_dir()
        py = applio_python(root) if root else None
        key = self._desired_key()
        active = key[0] if key else None
        loaded = self._backend if (self._backend and self._backend_key == key) else None
        return {
            "available": key is not None,
            "backend": active,
            "device": (loaded.device if loaded is not None and active == "applio"
                       else select_device()),
            "device_override": device_override(),
            "applio": {
                "configured_dir": configured_applio_dir_raw(),
                "dir": str(root) if root else None,
                "python": str(py) if py else None,
                "managed_dir": str(managed_applio_dir()),
                "ready": bool(root and py),
            },
            "rvc_python": rvc_python_available(),
            "f0_methods": list(BACKEND_F0_METHODS.get(active, F0_METHODS)),
            "supported_params": list(BACKEND_PARAMS.get(active, ())),
            "worker_running": bool(isinstance(loaded, ApplioBackend)
                                   and loaded._proc is not None
                                   and loaded._proc.poll() is None),
            "cleanup_ffmpeg": _safe_cleanup_status(root) if root else None,
        }

    def convert_file(self, input_path: str, output_path: str, model_path: str,
                     index_path: Optional[str], params: ConversionParams) -> ConversionResult:
        params.validate()
        backend = self.backend()
        allowed = BACKEND_F0_METHODS.get(backend.name)
        if allowed and params.f0_method not in allowed:
            raise ValueError(f"The {backend.name} backend does not support "
                             f"f0_method={params.f0_method!r}; use {', '.join(allowed)}.")
        start = time.time()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            sr = backend.convert(input_path, output_path, model_path, index_path, params)
        except (RVCUnavailable, ValueError):
            raise
        except RVCConversionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RVCConversionError(f"{type(exc).__name__}: {exc}") from exc
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RVCConversionError("The converter produced no audio.")
        return ConversionResult(output_path, int(sr), backend.name, backend.device,
                                round(time.time() - start, 2), params.to_dict())

    def convert_voice(self, voice_id: str, input_path: str, output_path: str,
                      params: ConversionParams) -> ConversionResult:
        from services import rvc_registry

        model, index = rvc_registry.voice_files(voice_id)
        return self.convert_file(input_path, output_path, model, index, params)

    def convert_tensor(self, audio_tensor, sample_rate: int, voice_id: str,
                       params: ConversionParams):
        """TTS post-step: tensor in → ``(tensor, model_sample_rate)`` out."""
        import torch

        from services import rvc_registry

        rvc_registry.voice_files(voice_id)  # 404 early, before any temp I/O
        with tempfile.TemporaryDirectory(prefix="ov-rvc-") as tmp:
            src = os.path.join(tmp, "in.wav")
            dst = os.path.join(tmp, "out.wav")
            _write_wav(src, audio_tensor, sample_rate)
            result = self.convert_voice(voice_id, src, dst, params)
            data, sr = _read_wav(dst)
        tensor = torch.from_numpy(data).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor, sr, result

    def unload(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.unload()

    def shutdown(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
            self._backend = None
            self._backend_key = None

    def loaded_info(self) -> Optional[dict]:
        """What this service currently holds, for the loaded-models panel."""
        b = self._backend
        if isinstance(b, ApplioBackend) and b.running():
            return {"backend": "applio", "pid": b.pid, "device": b.device,
                    "model": b._loaded, "idle_s": round(b.idle_seconds(), 1)}
        if isinstance(b, RvcPythonBackend) and b._engine is not None:
            return {"backend": "rvc-python", "pid": os.getpid(), "device": b.device,
                    "model": (b._loaded or (None,))[0], "idle_s": None}
        return None

    def release(self) -> bool:
        """Free everything (stop the worker). True when something was held."""
        held = self.loaded_info() is not None
        if held:
            with self._lock:
                if self._backend is not None:
                    self._backend.close()
        return held

    def release_idle(self, idle_s: Optional[float] = None) -> bool:
        idle_s = idle_timeout_s() if idle_s is None else idle_s
        b = self._backend
        if isinstance(b, ApplioBackend):
            return b.reap_if_idle(idle_s)
        return False


def _write_wav(path: str, audio_tensor, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    data = audio_tensor.detach().cpu().float().numpy() if hasattr(audio_tensor, "detach") \
        else np.asarray(audio_tensor, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=0) if data.shape[0] <= 8 else data.mean(axis=1)
    sf.write(path, data, int(sample_rate), subtype="FLOAT")


def _read_wav(path: str):
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, int(sr)


_service: Optional[VoiceConversionService] = None
_service_lock = threading.Lock()


def get_service() -> VoiceConversionService:
    global _service
    with _service_lock:
        if _service is None:
            _service = VoiceConversionService()
        return _service


def reset_service() -> None:
    """Tests / settings changes: drop the cached backend (kills the worker)."""
    global _service
    with _service_lock:
        if _service is not None:
            _service.shutdown()
        _service = None


# ── Legacy env hook (dubbing) ────────────────────────────────────────────────


def rvc_config() -> Optional[dict]:
    """Return RVC config dict, or None if disabled/unconfigured."""
    if os.environ.get("OMNIVOICE_RVC_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        return None
    model = os.environ.get("OMNIVOICE_RVC_MODEL", "").strip()
    index = os.environ.get("OMNIVOICE_RVC_INDEX", "").strip() or None
    voice = os.environ.get("OMNIVOICE_RVC_VOICE", "").strip()
    if voice and not model:
        try:
            from services import rvc_registry

            model, registered_index = rvc_registry.voice_files(voice)
            index = index or registered_index
        except Exception:  # noqa: BLE001
            logger.warning("RVC enabled but OMNIVOICE_RVC_VOICE %r is not registered", voice)
            return None
    if not model or not os.path.exists(model):
        logger.warning("RVC enabled but OMNIVOICE_RVC_MODEL missing or invalid: %r", model)
        return None
    try:
        return {
            "model_path": model,
            "index_path": index,
            "pitch": int(os.environ.get("OMNIVOICE_RVC_PITCH", "0")),
            "index_rate": float(os.environ.get("OMNIVOICE_RVC_INDEX_RATE", "0.7")),
            "filter_radius": int(os.environ.get("OMNIVOICE_RVC_FILTER_RAD", "3")),
            "protect": float(os.environ.get("OMNIVOICE_RVC_PROTECT", "0.33")),
            "f0_method": os.environ.get("OMNIVOICE_RVC_F0_METHOD", "rmvpe").strip() or "rmvpe",
        }
    except ValueError:
        logger.warning("RVC enabled but an OMNIVOICE_RVC_* number is malformed")
        return None


def is_enabled() -> bool:
    return rvc_config() is not None


def apply_rvc(wav_path: str) -> str:
    """Apply RVC to a segment wav in place. Returns the same path.

    No-op when RVC is disabled, unconfigured, or no backend is installed.
    Errors during inference are logged and swallowed (returns original path).
    Runs on the auto-selected device (CUDA/MPS/CPU) via the shared service.
    """
    cfg = rvc_config()
    if cfg is None:
        return wav_path
    try:
        params = ConversionParams(
            pitch=cfg["pitch"], f0_method=cfg["f0_method"], index_rate=cfg["index_rate"],
            filter_radius=cfg["filter_radius"], protect=cfg["protect"],
        ).validate()
    except ValueError as exc:
        logger.warning("RVC skipped — invalid OMNIVOICE_RVC_* parameter: %s", exc)
        return wav_path
    out_fd, out_path = tempfile.mkstemp(suffix=".wav", dir=os.path.dirname(wav_path) or None)
    os.close(out_fd)
    try:
        get_service().convert_file(wav_path, out_path, cfg["model_path"], cfg["index_path"],
                                   params)
        os.replace(out_path, wav_path)
    except RVCUnavailable as exc:
        logger.warning("RVC enabled but unavailable; skipping conversion: %s", exc)
    except Exception:  # noqa: BLE001
        logger.exception("RVC inference failed for %s", wav_path)
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)
    return wav_path
