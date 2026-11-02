"""Client for ACE-Step 1.5's local API server (Audio Edit song repaint).

Starts ``python -m acestep.api_server`` from the ACE-Step venv on
127.0.0.1 at a free port (or reuses a loopback server named by
OMNIVOICE_ACESTEP_URL), waits for /health, posts one ``repaint`` task
with the source mix uploaded, polls /query_result and downloads the
result from /v1/audio. Nothing here talks to anything but loopback.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from services.audio_edit_engines import engine_checkout, engine_python

STARTUP_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 1.0
REPAINT_TIMEOUT_S = 900.0
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class AceStepUnavailable(RuntimeError):
    code = "ace_step_unavailable"


class AceStepError(RuntimeError):
    """The server accepted the task but it failed."""


class AceStepCancelled(RuntimeError):
    """The render was cancelled while ACE-Step was working."""


def is_loopback_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _data(resp: httpx.Response):
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


class AceStepClient:
    def __init__(self, work_dir: Path) -> None:
        self._work_dir = Path(work_dir)
        self._proc: Optional[subprocess.Popen] = None
        self._log = None
        self._base: Optional[str] = None
        self._cancelled = threading.Event()

    @property
    def started_server(self) -> bool:
        return self._proc is not None

    def ensure_running(self) -> str:
        url = os.environ.get("OMNIVOICE_ACESTEP_URL", "").strip()
        if url:
            if not is_loopback_url(url):
                raise AceStepUnavailable(
                    "OMNIVOICE_ACESTEP_URL must point at this computer (127.0.0.1 or localhost)."
                )
            self._base = url.rstrip("/")
        elif self._proc is None or self._proc.poll() is not None:
            self._base = self._spawn()
        self._wait_healthy()
        return self._base

    def _spawn(self) -> str:
        py, checkout = engine_python("ace_step"), engine_checkout("ace_step")
        if py is None or checkout is None:
            raise AceStepUnavailable("Install ACE-Step 1.5 to use it.")
        port = _free_port()
        tmp = self._work_dir / "acestep-tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "ACESTEP_API_HOST": "127.0.0.1",
            "ACESTEP_API_PORT": str(port),
            "ACESTEP_INIT_LLM": "false",
            "ACESTEP_TMPDIR": str(tmp),
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        env.pop("ACESTEP_API_KEY", None)
        self._log = open(self._work_dir / "acestep-server.log", "ab")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        self._proc = subprocess.Popen(
            [str(py), "-m", "acestep.api_server"],
            cwd=str(checkout), env=env, stdin=subprocess.DEVNULL,
            stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags,
        )
        return f"http://127.0.0.1:{port}"

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            self._check_cancel()
            if self._proc is not None and self._proc.poll() is not None:
                raise AceStepUnavailable(
                    "ACE-Step stopped while starting. See acestep-server.log in the session folder."
                )
            try:
                resp = httpx.get(f"{self._base}/health", timeout=2.0)
                if resp.status_code == 200 and (_data(resp) or {}).get("status") == "ok":
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.25)
        raise AceStepUnavailable(f"ACE-Step did not answer within {int(STARTUP_TIMEOUT_S)} s.")

    def _check_cancel(self) -> None:
        if self._cancelled.is_set():
            raise AceStepCancelled("cancelled")

    def repaint(self, src_path: Path, start_s: float, end_s: float, lyrics: str,
                caption: str, strength: float, seed: int, out_path: Path,
                language: str = "en") -> Path:
        self._check_cancel()
        base = self.ensure_running()
        form = {
            "task_type": "repaint",
            "repainting_start": f"{start_s:.3f}",
            "repainting_end": f"{end_s:.3f}",
            "lyrics": lyrics,
            "prompt": caption,
            "vocal_language": language,
            "audio_cover_strength": f"{strength:g}",
            "chunk_mask_mode": "explicit",
            "use_random_seed": "false",
            "seed": str(int(seed)),
            "batch_size": "1",
            "inference_steps": "8",
            "thinking": "false",
            "use_cot_caption": "false",
            "use_cot_language": "false",
            "audio_format": "wav",
        }
        src_path = Path(src_path)
        with open(src_path, "rb") as fh:
            resp = httpx.post(f"{base}/release_task", data=form,
                              files={"src_audio": (src_path.name, fh, "audio/wav")}, timeout=60.0)
        task_id = (_data(resp) or {}).get("task_id")
        if not task_id:
            raise AceStepError("ACE-Step did not accept the repaint task.")
        file_url = self._poll(base, task_id)
        out_path = Path(out_path)
        with httpx.stream("GET", f"{base}{file_url}", timeout=120.0) as dl:
            dl.raise_for_status()
            with open(out_path, "wb") as out:
                for chunk in dl.iter_bytes():
                    out.write(chunk)
        return out_path

    def _poll(self, base: str, task_id: str) -> str:
        deadline = time.monotonic() + REPAINT_TIMEOUT_S
        while time.monotonic() < deadline:
            self._check_cancel()
            if self._proc is not None and self._proc.poll() is not None:
                raise AceStepError("ACE-Step stopped during the repaint.")
            items = _data(httpx.post(f"{base}/query_result",
                                     json={"task_id_list": [task_id]}, timeout=30.0)) or []
            item = items[0] if items else {}
            status = item.get("status")
            if status == 2:
                raise AceStepError(f"ACE-Step repaint failed: {item.get('error') or 'unknown error'}")
            if status == 1:
                result = item.get("result")
                files = json.loads(result) if isinstance(result, str) else (result or [])
                if not files or not files[0].get("file"):
                    raise AceStepError("ACE-Step returned no audio.")
                return str(files[0]["file"])
            time.sleep(POLL_INTERVAL_S)
        raise AceStepError("ACE-Step repaint timed out.")

    def cancel(self) -> None:
        self._cancelled.set()

    def stop(self) -> None:
        """Stop the server only if this client started it."""
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._log is not None:
            self._log.close()
            self._log = None
