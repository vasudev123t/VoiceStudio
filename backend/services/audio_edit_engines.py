"""Sidecar clients for the Audio Edit engines (F5-TTS, YingMusic, Vevo2).

Each engine runs ``backend/engines/<dir>/main.py`` in the private venv the
sidecar installer made. The parent sends one ``edit`` frame per span and
reads ``progress`` frames until ``audio`` (the sidecar wrote a WAV and
replies with its path) or ``error``. These are not TTS backends and are
never listed in the engine picker. ACE-Step is an HTTP server instead; see
services/acestep_client.py.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Optional

from services.subprocess_backend import SubprocessBackend

_ENGINES_DIR = Path(__file__).resolve().parents[1] / "engines"
#: One span can take minutes on CPU; a wedged sidecar is still reclaimed.
EDIT_TIMEOUT_S = 900.0

#: Audio Edit engine id -> (sidecar_install id, env var of its install dir).
INSTALLS = {
    "f5tts": ("f5tts", "OMNIVOICE_F5TTS_DIR"),
    "ace_step": ("ace-step", "OMNIVOICE_ACESTEP_DIR"),
    "yingmusic": ("yingmusic", "OMNIVOICE_YINGMUSIC_DIR"),
    "vevo2": ("vevo2", "OMNIVOICE_VEVO2_DIR"),
}
DISPLAY_NAMES = {
    "f5tts": "F5-TTS",
    "ace_step": "ACE-Step 1.5",
    "yingmusic": "YingMusic-Singer-Plus",
    "vevo2": "Vevo2",
}
#: Licence-registry model ids each engine's output depends on.
ENGINE_MODELS = {
    "f5tts": ("SWivid/F5-TTS", "charactr/vocos-mel-24khz"),
    "ace_step": ("ACE-Step/Ace-Step1.5",),
    "yingmusic": ("ASLP-lab/YingMusic-Singer-Plus",),
    "vevo2": ("RMSnow/Vevo2",),
}


class EngineError(RuntimeError):
    """An engine failed or is not installed; the message is user-facing."""


def engine_python(engine_id: str) -> Optional[Path]:
    from services.sidecar_install import engine_venv_python
    return engine_venv_python(INSTALLS[engine_id][1])


def engine_checkout(engine_id: str) -> Optional[Path]:
    value = os.environ.get(INSTALLS[engine_id][1])
    return Path(value) if value else None


class EditSidecar(SubprocessBackend):
    """A long-lived engine sidecar that answers ``edit`` frames."""

    engine_key = "f5tts"
    sidecar_dir = "f5tts"
    recv_timeout_s = EDIT_TIMEOUT_S

    @property
    def sample_rate(self) -> int:
        return 44100

    @property
    def supported_languages(self) -> list[str]:
        return ["multi"]

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if engine_python(cls.engine_key) is None:
            return False, f"{DISPLAY_NAMES[cls.engine_key]} is not installed."
        return True, ""

    @classmethod
    def venv_python(cls) -> Path:
        py = engine_python(cls.engine_key)
        if py is None:
            raise EngineError(f"{DISPLAY_NAMES[cls.engine_key]} is not installed.")
        return py

    @classmethod
    def sidecar_script(cls) -> Path:
        return _ENGINES_DIR / cls.sidecar_dir / "main.py"

    def edit(self, payload: dict, on_progress: Optional[Callable[[float], None]] = None) -> dict:
        with self._lock:
            self._spawn()
            self._send({"op": "edit", **payload})
            while True:
                frame = self._recv_with_timeout(self.recv_timeout_s)
                if frame is None:
                    tail = self._stderr_tail_text()
                    raise EngineError(f"{self.display_name} stopped unexpectedly. {tail}".strip())
                op = frame.get("op")
                if op == "progress":
                    if on_progress is not None:
                        on_progress(float(frame.get("fraction") or 0.0))
                    continue
                if op == "error":
                    raise EngineError(str(frame.get("message") or f"{self.display_name} failed."))
                if op == "audio":
                    path = frame.get("path")
                    if not isinstance(path, str) or not Path(path).is_file():
                        raise EngineError(f"{self.display_name} returned no audio file.")
                    self._touch()
                    return frame

    def kill(self) -> None:
        """Hard-stop the sidecar (cancel). Lock-free: an edit may hold _lock."""
        self._force_kill()


class F5TTSEdit(EditSidecar):
    id = "audio-edit-f5tts"
    display_name = "F5-TTS"
    engine_key = "f5tts"
    sidecar_dir = "f5tts"


EDITORS: dict[str, type[EditSidecar]] = {"f5tts": F5TTSEdit}
_INSTANCES: dict[str, EditSidecar] = {}
_INSTANCES_LOCK = threading.Lock()


def get_edit_engine(engine_id: str) -> EditSidecar:
    cls = EDITORS[engine_id]
    with _INSTANCES_LOCK:
        if engine_id not in _INSTANCES:
            _INSTANCES[engine_id] = cls()
        return _INSTANCES[engine_id]


#: Engines that are unusable without CUDA or MPS (F5-TTS falls back to CPU).
GPU_ENGINES = frozenset({"ace_step", "yingmusic", "vevo2"})


def has_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    mps = getattr(torch.backends, "mps", None)
    return bool(torch.cuda.is_available() or (mps is not None and mps.is_available()))


def engine_status(engine_id: str) -> dict:
    """``{installed, install_id, needs_gpu, available, reason}`` for one engine."""
    install_id, _env_var = INSTALLS[engine_id]
    installed = engine_python(engine_id) is not None
    if engine_id == "ace_step" and os.environ.get("OMNIVOICE_ACESTEP_URL"):
        installed = True
    needs_gpu = engine_id in GPU_ENGINES
    reason = None
    if not installed:
        reason = f"Install {DISPLAY_NAMES[engine_id]} to use it."
    elif needs_gpu and not has_gpu():
        reason = f"{DISPLAY_NAMES[engine_id]} needs a CUDA or Apple GPU."
    return {"installed": installed, "install_id": install_id, "needs_gpu": needs_gpu,
            "available": reason is None, "reason": reason}
