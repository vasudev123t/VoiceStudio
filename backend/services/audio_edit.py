"""Audio Edit: regenerate chosen words inside speech or sung vocals.

Sessions live under ``<DATA_DIR>/audio_edit/<session_id>/`` (override with
``OMNIVOICE_AUDIO_EDIT_DIR``). Separation, transcription and renders run as
background-thread jobs, like the Voice Lab dataset builder; only one render
holds the GPU at a time. Unedited audio is never re-encoded: each edit is
spliced into the original samples with short equal-power crossfades.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from services.audio_edit_dsp import as_2d

logger = logging.getLogger("omnivoice.audio_edit")

MODES = ("speech", "song")
ENGINES = ("f5tts", "ace_step", "yingmusic", "vevo2")
MODE_ENGINES = {"speech": ("f5tts",), "song": ("ace_step", "yingmusic", "vevo2")}
#: Engines whose weights are non-commercial: the user acknowledges once.
ACK_ENGINES = frozenset({"f5tts", "yingmusic", "vevo2"})
MAX_DURATION_S = {"speech": 600.0, "song": 480.0}
SAMPLE_RATE = 44100
CHANNELS = {"speech": 1, "song": 2}
AUDIO_KINDS = ("source", "vocals", "instrumental")
LIVE_STATUSES = frozenset({"queued", "running"})
MAX_SPANS = 50
MAX_REPLACEMENT_CHARS = 500
SPEECH_CONTEXT_S = 1.5
SONG_CONTEXT_S = 2.0
ALLOWED_UPLOAD_EXTS = frozenset({
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
    ".webm", ".mp4", ".mov", ".mkv",
})
#: Tests flip this to run jobs synchronously on the calling thread.
RUN_INLINE = False

_ID_RE = re.compile(r"[0-9a-f]{32}")
_SPAN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_STAGE_JOB = {"separating": "separate", "transcribing": "transcribe"}
#: Serialises read-modify-write of session.json and render.json.
_STATE_LOCK = threading.RLock()


class AudioEditError(Exception):
    """A user-facing failure with a stable ``code`` the UI translates."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def detail(self) -> dict:
        return {"code": self.code, "message": self.message}


# ── paths ────────────────────────────────────────────────────────────────

def root() -> Path:
    override = os.environ.get("OMNIVOICE_AUDIO_EDIT_DIR")
    if override:
        return Path(override)
    from core.config import DATA_DIR
    return Path(DATA_DIR) / "audio_edit"


def _check_id(value: object, what: str = "session") -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise AudioEditError("not_found", f"Unknown {what}.", 404)
    return value


def session_dir(session_id: str) -> Path:
    base = root().resolve()
    path = (base / _check_id(session_id)).resolve()
    if path.parent != base:
        raise AudioEditError("not_found", "Unknown session.", 404)
    return path


def render_dir(session_id: str, render_id: str) -> Path:
    return session_dir(session_id) / "renders" / _check_id(render_id, "render")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ── audio I/O ────────────────────────────────────────────────────────────

def decode_to_wav(src: Path, dst: Path, channels: int) -> float:
    """Decode any local audio/video file to 44.1 kHz float WAV; returns seconds."""
    import soundfile as sf
    from services.dataset_builder import _ffmpeg

    cmd = [
        _ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file", "-y", "-i", str(src),
        "-vn", "-ac", str(channels), "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioEditError("invalid_audio", f"Could not decode the file: {exc}", 400) from exc
    if proc.returncode != 0 or not dst.is_file():
        raise AudioEditError("invalid_audio", "Could not read that file as audio.", 400)
    return float(sf.info(str(dst)).duration)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return np.ascontiguousarray(data.T), int(sr)


def write_audio(path: Path, audio, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), as_2d(audio).T, sr, subtype="FLOAT")


# ── jobs registry (runner in Task 8) ─────────────────────────────────────

@dataclass
class EditJob:
    id: str
    session_id: str
    kind: str                      # "separate" | "transcribe" | "render"
    render_id: Optional[str] = None
    status: str = "queued"         # queued | running | succeeded | failed | cancelled
    stage: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None    # an AudioEditError code
    message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    engine: Any = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "session_id", "kind", "render_id", "status", "stage",
            "progress", "error", "message", "created_at")}


_JOBS: dict[str, EditJob] = {}
_JOBS_LOCK = threading.Lock()


def get_job(job_id: str) -> EditJob:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise AudioEditError("not_found", "Unknown job.", 404)
    return job


def _job_alive(session_id: str, key: str) -> bool:
    """A live job for *session_id* whose kind or render id is *key*."""
    with _JOBS_LOCK:
        return any(j.session_id == session_id and j.status in LIVE_STATUSES
                   and key in (j.kind, j.render_id) for j in _JOBS.values())


def _session_busy(session_id: str) -> bool:
    with _JOBS_LOCK:
        return any(j.session_id == session_id and j.status in LIVE_STATUSES
                   for j in _JOBS.values())


# ── sessions and renders ─────────────────────────────────────────────────

def _load_raw(session_id: str) -> dict:
    path = session_dir(session_id) / "session.json"
    if not path.is_file():
        raise AudioEditError("not_found", "Unknown session.", 404)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_session(session: dict) -> None:
    data = {k: v for k, v in session.items() if k != "renders"}
    _write_json(session_dir(data["id"]) / "session.json", data)


def update_session(session_id: str, **changes) -> dict:
    with _STATE_LOCK:
        session = _load_raw(session_id)
        session.update(changes)
        _save_session(session)
        return session


def _save_render(render: dict) -> None:
    _write_json(render_dir(render["session_id"], render["id"]) / "render.json", render)


def load_render(session_id: str, render_id: str) -> dict:
    path = render_dir(session_id, render_id) / "render.json"
    if not path.is_file():
        raise AudioEditError("not_found", "Unknown render.", 404)
    with _STATE_LOCK:
        render = json.loads(path.read_text(encoding="utf-8"))
        if render.get("status") in LIVE_STATUSES and not _job_alive(session_id, render_id):
            render.update(status="failed", error="interrupted",
                          message="The app stopped before this render finished.",
                          finished_at=render.get("finished_at") or time.time())
            _save_render(render)
    return render


def update_render(session_id: str, render_id: str, **changes) -> dict:
    with _STATE_LOCK:
        render = json.loads((render_dir(session_id, render_id) / "render.json").read_text(encoding="utf-8"))
        render.update(changes)
        _save_render(render)
        return render


def load_session(session_id: str) -> dict:
    with _STATE_LOCK:
        session = _load_raw(session_id)
        stage = session.get("stage")
        if stage in _STAGE_JOB and not _job_alive(session_id, _STAGE_JOB[stage]):
            fallback = "uploaded" if stage == "separating" or session["mode"] == "speech" else "separated"
            session.update(stage=fallback, error="interrupted")
            _save_session(session)
        renders = []
        for render_id in reversed(session.get("render_ids", [])):
            try:
                renders.append(load_render(session_id, render_id))
            except (AudioEditError, ValueError, OSError):
                continue
    session["renders"] = renders
    return session


def create_session(mode: str, name: str, upload_path: Path, original_name: str) -> dict:
    if mode not in MODES:
        raise AudioEditError("invalid_mode", "Mode must be speech or song.", 422)
    session_id = uuid.uuid4().hex
    d = root() / session_id
    d.mkdir(parents=True, exist_ok=False)
    try:
        duration = decode_to_wav(Path(upload_path), d / "source.wav", CHANNELS[mode])
        if duration > MAX_DURATION_S[mode]:
            minutes = int(MAX_DURATION_S[mode] // 60)
            raise AudioEditError("too_long", f"{mode.capitalize()} files are limited to {minutes} minutes.", 400)
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    label = (name or "").strip()[:120] or Path(original_name or "").stem[:120] or "Untitled"
    session = {
        "id": session_id, "mode": mode, "name": label, "stage": "uploaded", "error": None,
        "duration_s": round(duration, 3), "sample_rate": SAMPLE_RATE, "channels": CHANNELS[mode],
        "created_at": time.time(), "transcript": None, "spans": [], "render_ids": [],
    }
    _save_session(session)
    return load_session(session_id)


def list_sessions() -> list[dict]:
    base = root()
    if not base.is_dir():
        return []
    sessions = []
    for child in base.iterdir():
        if not (child.is_dir() and _ID_RE.fullmatch(child.name) and (child / "session.json").is_file()):
            continue
        try:
            sessions.append(load_session(child.name))
        except (AudioEditError, ValueError, OSError) as exc:
            logger.warning("skipping unreadable Audio Edit session %s: %s", child.name, exc)
    sessions.sort(key=lambda s: s.get("created_at", 0), reverse=True)
    return sessions


def delete_session(session_id: str) -> None:
    d = session_dir(session_id)
    if not (d / "session.json").is_file():
        raise AudioEditError("not_found", "Unknown session.", 404)
    if _session_busy(session_id):
        raise AudioEditError("busy", "Cancel the running job before deleting this session.", 409)
    shutil.rmtree(d)


def audio_path(session_id: str, kind: str) -> Path:
    if kind not in AUDIO_KINDS:
        raise AudioEditError("not_found", "Unknown audio kind.", 404)
    path = session_dir(session_id) / f"{kind}.wav"
    if not path.is_file():
        raise AudioEditError("not_found", f"This session has no {kind} audio.", 404)
    return path


def render_audio_path(session_id: str, render_id: str) -> Path:
    render = load_render(session_id, render_id)
    if render.get("status") != "succeeded" or not render.get("output_path"):
        raise AudioEditError("not_ready", "This render has no audio yet.", 409)
    path = session_dir(session_id) / render["output_path"]
    if not path.is_file():
        raise AudioEditError("not_found", "The rendered file is missing.", 404)
    return path


# ── spans and transcript ─────────────────────────────────────────────────

def _bad_spans(message: str) -> AudioEditError:
    return AudioEditError("invalid_spans", message, 400)


def validate_spans(raw: object, duration_s: float) -> list[dict]:
    """Normalise and check client spans; sorted by start, never overlapping."""
    if not isinstance(raw, list):
        raise _bad_spans("Spans must be a list.")
    if len(raw) > MAX_SPANS:
        raise _bad_spans(f"At most {MAX_SPANS} spans per session.")
    seen: set[str] = set()
    spans = []
    for item in raw:
        if not isinstance(item, dict):
            raise _bad_spans("Each span must be an object.")
        span_id = item.get("id")
        if not isinstance(span_id, str) or not _SPAN_ID_RE.fullmatch(span_id) or span_id in seen:
            raise _bad_spans("Each span needs a unique id of letters, digits, - or _.")
        try:
            start, end = float(item["start_s"]), float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            raise _bad_spans(f"Span {span_id} needs numeric start_s and end_s.") from None
        if not (0.0 <= start < end <= duration_s + 1e-6):
            raise _bad_spans(f"Span {span_id} is outside the audio.")
        original = str(item.get("original_text") or "").strip()
        replacement = str(item.get("replacement_text") or "").strip()
        if len(replacement) > MAX_REPLACEMENT_CHARS:
            raise _bad_spans(f"Span {span_id}: replacements are limited to {MAX_REPLACEMENT_CHARS} characters.")
        if replacement == original:
            raise _bad_spans(f"Span {span_id} does not change anything.")
        seen.add(span_id)
        spans.append({"id": span_id, "start_s": round(start, 3), "end_s": round(min(end, duration_s), 3),
                      "original_text": original, "replacement_text": replacement})
    spans.sort(key=lambda s: s["start_s"])
    for prev, cur in zip(spans, spans[1:]):
        if cur["start_s"] < prev["end_s"]:
            raise _bad_spans(f"Spans {prev['id']} and {cur['id']} overlap.")
    return spans


def set_spans(session_id: str, raw: object) -> dict:
    with _STATE_LOCK:
        session = _load_raw(session_id)
        if session.get("stage") != "ready" or not session.get("transcript"):
            raise AudioEditError("not_ready", "Transcribe the audio before marking spans.", 409)
        session["spans"] = validate_spans(raw, float(session["duration_s"]))
        _save_session(session)
    return load_session(session_id)


def words_from_asr(result: dict) -> list[dict]:
    """Flatten an ASR result to ``[{i, word, start, end}]`` with timings."""
    raw = result.get("words")
    if not isinstance(raw, list) or not raw:
        raw = [w for seg in result.get("segments") or [] for w in (seg.get("words") or [])]
    words = []
    for w in raw:
        text = str(w.get("word") or "").strip()
        start, end = w.get("start"), w.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
            continue
        words.append({"i": len(words), "word": text, "start": round(float(start), 3), "end": round(float(end), 3)})
    if not words:
        raise AudioEditError("no_word_timestamps",
                             "The speech recogniser returned no word timings. Pick an engine with word timestamps in Settings.",
                             422)
    return words


@dataclass(frozen=True)
class Window:
    """The audio an engine sees for one span: the span plus context words."""
    start_s: float
    end_s: float
    raw_text: str
    target_text: str


def window_bounds(spans: list[dict], idx: int, duration_s: float) -> tuple[float, float]:
    """Midpoints to the neighbouring spans, so windows never overlap."""
    lo = 0.0 if idx == 0 else (spans[idx - 1]["end_s"] + spans[idx]["start_s"]) / 2
    hi = duration_s if idx == len(spans) - 1 else (spans[idx]["end_s"] + spans[idx + 1]["start_s"]) / 2
    return lo, hi


def span_window(span: dict, words: list[dict], duration_s: float, context_s: float,
                lo_s: float = 0.0, hi_s: Optional[float] = None) -> Window:
    hi_s = duration_s if hi_s is None else hi_s
    start, end = span["start_s"], span["end_s"]
    eps = 1e-3
    before = [w for w in words if w["end"] <= start + eps and w["start"] >= max(lo_s, start - context_s) - eps]
    after = [w for w in words if w["start"] >= end - eps and w["end"] <= min(hi_s, end + context_s) + eps]
    w_start = before[0]["start"] if before else max(lo_s, start - min(context_s, 0.25))
    w_end = after[-1]["end"] if after else min(hi_s, end + min(context_s, 0.25))
    w_start, w_end = max(lo_s, min(w_start, start)), min(hi_s, max(w_end, end))
    b_text = " ".join(w["word"] for w in before)
    a_text = " ".join(w["word"] for w in after)
    raw = " ".join(t for t in (b_text, span["original_text"], a_text) if t)
    target = " ".join(t for t in (b_text, span["replacement_text"], a_text) if t)
    return Window(round(w_start, 3), round(w_end, 3), raw, target)
