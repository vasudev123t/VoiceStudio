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

from services import audio_edit_dsp as dsp
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


# ── licences and commercial use ──────────────────────────────────────────

_ACK_PREF = "audio_edit.licence_acks"
#: Licence-registry ids a render depends on beyond the engine's own models.
HTDEMUCS_RECORD = "htdemucs"
RVC_RECORD = ("User-provided Sherpa ONNX / RVC voice models / GPT-SoVITS server / "
              "custom MLX and env-overridden repos")


def _read_acks() -> dict:
    from core import prefs

    value = prefs.get(_ACK_PREF, {})
    return dict(value) if isinstance(value, dict) else {}


def _write_acks(acks: dict) -> None:
    from core import prefs

    prefs.set_(_ACK_PREF, dict(acks))


def _check_engine(engine: object) -> str:
    if engine not in ENGINES:
        raise AudioEditError("invalid_engine", f"Engine must be one of: {', '.join(ENGINES)}.", 422)
    return str(engine)


def licence_acks() -> dict[str, float]:
    """Accepted non-commercial licences: ``{engine: accepted_at}``."""
    return {k: float(v) for k, v in _read_acks().items()
            if k in ACK_ENGINES and isinstance(v, (int, float)) and not isinstance(v, bool)}


def acknowledge_licence(engine: str) -> dict:
    _check_engine(engine)
    if engine not in ACK_ENGINES:
        return {"engine": engine, "required": False, "accepted_at": None}
    with _STATE_LOCK:
        acks = licence_acks()
        acks[engine] = time.time()
        _write_acks(acks)
    return {"engine": engine, "required": True, "accepted_at": acks[engine]}


def revoke_licence(engine: str) -> dict:
    _check_engine(engine)
    with _STATE_LOCK:
        acks = licence_acks()
        acks.pop(engine, None)
        _write_acks(acks)
    return {"engine": engine, "required": engine in ACK_ENGINES, "accepted_at": None}


def require_licence(engine: str) -> None:
    if engine in ACK_ENGINES and engine not in licence_acks():
        from services.audio_edit_engines import DISPLAY_NAMES

        raise AudioEditError("licence_ack_required",
                             f"Acknowledge the {DISPLAY_NAMES[engine]} licence before rendering with it.", 403)


def _registry_records() -> dict[str, dict]:
    from services.model_licenses import load_registry

    registry = load_registry()
    records: dict[str, dict] = {}
    for section in ("models", "dynamic_assets"):
        for record in registry.get(section) or []:
            if isinstance(record, dict) and isinstance(record.get("id"), str):
                records[record["id"]] = record
    return records


def _cleared(record: Optional[dict]) -> bool:
    return bool(record) and record.get("commercial_use") is True and record.get("review_status") == "cleared"


def render_commercial_use(engine: str, *, rvc: bool, demucs: bool,
                          records: Optional[dict] = None) -> bool:
    """True only if every licence record behind the output allows commercial use."""
    from services.audio_edit_engines import ENGINE_MODELS

    records = _registry_records() if records is None else records
    ids = list(ENGINE_MODELS[engine])
    if demucs:
        ids.append(HTDEMUCS_RECORD)
    if rvc:
        ids.append(RVC_RECORD)
    return all(_cleared(records.get(i)) for i in ids)


def separation_status() -> dict:
    """HTDemucs readiness, shared with Voice Lab's dataset builder."""
    from services import dataset_builder

    sep = dataset_builder.capabilities()["separation"]
    return {"available": bool(sep.get("available")), "weights": sep.get("weights"),
            "missing": sep.get("missing")}


def engines_overview() -> dict:
    from services import audio_edit_engines as aee

    records = _registry_records()
    acks = licence_acks()
    engines = []
    for engine in ENGINES:
        mode = "speech" if engine in MODE_ENGINES["speech"] else "song"
        licence = " / ".join(dict.fromkeys(
            str((records.get(i) or {}).get("license") or "unknown") for i in aee.ENGINE_MODELS[engine]))
        engines.append({
            "id": engine, "mode": mode, "display_name": aee.DISPLAY_NAMES[engine],
            **aee.engine_status(engine),
            "licence": licence, "needs_ack": engine in ACK_ENGINES, "acked_at": acks.get(engine),
            "commercial_use": render_commercial_use(engine, rvc=False, demucs=mode == "song", records=records),
        })
    return {"engines": engines, "separation": separation_status(), "gpu": aee.has_gpu()}


# ── job runner ───────────────────────────────────────────────────────────

#: One Audio Edit job holds the GPU at a time; the rest wait their turn.
_RENDER_SLOT = threading.Semaphore(1)


class Cancelled(Exception):
    """The user cancelled the job."""


def _check_cancel(job: EditJob) -> None:
    if job.cancel.is_set():
        raise Cancelled()


def _free_vram() -> None:
    try:
        from services.model_manager import free_vram

        free_vram()
    except Exception:  # noqa: BLE001 — best effort after a job
        logger.debug("free_vram after an Audio Edit job failed", exc_info=True)


def _release_gpu() -> None:
    """Unload models other workspaces hold before an Audio Edit GPU stage."""
    try:
        from services.gpu_memory import release_gpu_holders

        release_gpu_holders(reason="audio_edit")
    except Exception:  # noqa: BLE001 — a failed release must not fail the edit
        logger.warning("Could not free the GPU before an Audio Edit stage", exc_info=True)


def _new_job(session_id: str, kind: str, render_id: Optional[str] = None) -> EditJob:
    job = EditJob(id=uuid.uuid4().hex, session_id=session_id, kind=kind, render_id=render_id)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
    return job


def _launch(job: EditJob, work) -> EditJob:
    if RUN_INLINE:
        _run(job, work)
    else:
        threading.Thread(target=_run, args=(job, work), daemon=True,
                         name=f"audio-edit-{job.kind}-{job.id[:8]}").start()
    return job


def _run(job: EditJob, work) -> None:
    acquired = False
    try:
        job.stage = "waiting"
        while not (acquired := _RENDER_SLOT.acquire(timeout=0.25)):
            _check_cancel(job)
        _check_cancel(job)
        job.status = "running"
        if job.kind == "render":
            update_render(job.session_id, job.render_id, status="running")
        work(job)
        job.status, job.progress, job.stage = "succeeded", 1.0, "done"
    except Cancelled:
        job.status = "cancelled"
    except AudioEditError as exc:
        job.status, job.error, job.message = "failed", exc.code, exc.message
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI as a failed job
        logger.exception("Audio Edit %s job %s failed", job.kind, job.id)
        job.status, job.error, job.message = "failed", "failed", f"{type(exc).__name__}: {exc}"[:500]
    finally:
        job.engine = None
        if acquired:
            _RENDER_SLOT.release()
        _free_vram()
        _finish(job)


def _finish(job: EditJob) -> None:
    """Persist a finished job's outcome into the session or render."""
    try:
        if job.kind == "render":
            changes: dict = {"status": job.status, "progress": round(job.progress, 3),
                             "finished_at": time.time()}
            if job.status != "succeeded":
                changes.update(error=job.error or job.status, message=job.message)
            update_render(job.session_id, job.render_id, **changes)
        elif job.status != "succeeded":
            session = _load_raw(job.session_id)
            back = "uploaded" if job.kind == "separate" or session["mode"] == "speech" else "separated"
            update_session(job.session_id, stage=back, error=job.error or job.status)
    except (AudioEditError, OSError, ValueError):
        logger.warning("Audio Edit job %s finished after its session was removed", job.id)


def cancel_job(job_id: str) -> EditJob:
    job = get_job(job_id)
    if job.status in LIVE_STATUSES:
        job.cancel.set()
        kill = job.engine
        if callable(kill):
            try:
                kill()
            except Exception:  # noqa: BLE001
                logger.warning("Could not stop the engine for job %s", job_id, exc_info=True)
    return job


# ── separation and transcription ─────────────────────────────────────────

def _start_stage_job(session_id: str, kind: str, allowed: dict, running_stage: str,
                     message: str) -> EditJob:
    with _STATE_LOCK:
        session = _load_raw(session_id)
        if _session_busy(session_id):
            raise AudioEditError("busy", "This session is already working.", 409)
        if session.get("stage") not in allowed[session["mode"]]:
            raise AudioEditError("not_ready", message, 409)
        job = _new_job(session_id, kind)
        session.update(stage=running_stage, error=None)
        _save_session(session)
    return job


def start_separation(session_id: str) -> EditJob:
    job = _start_stage_job(session_id, "separate", {"speech": (), "song": ("uploaded",)},
                           "separating", "Only a freshly uploaded song can be separated.")
    return _launch(job, _separate_work)


def start_transcription(session_id: str) -> EditJob:
    job = _start_stage_job(session_id, "transcribe",
                           {"speech": ("uploaded", "ready"), "song": ("separated", "ready")},
                           "transcribing", "Separate the song before transcribing it.")
    return _launch(job, _transcribe_work)


def separate_stems(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """HTDemucs: mono vocals and the summed stereo accompaniment, at SAMPLE_RATE."""
    status = separation_status()
    if not status["available"]:
        raise AudioEditError(
            "separation_unavailable",
            f"Song mode needs HTDemucs ({status['missing'] or 'model weights'}). Download it in Voice Lab first.",
            409)
    import torch
    from demucs.apply import apply_model

    from services.dataset_builder import load_htdemucs

    _release_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio, sr = read_audio(path)
    model = load_htdemucs(Path(status["weights"]), device)
    try:
        model_sr = int(model.samplerate)
        mix = dsp.resample(dsp.match_channels(audio, int(model.audio_channels)), sr, model_sr)
        wav = torch.from_numpy(np.ascontiguousarray(mix))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            stems = apply_model(model, ((wav - mean) / std)[None], device=device,
                                split=True, overlap=0.25, progress=False)[0]
        stems = (stems * std + mean).cpu().numpy().astype(np.float32)
        v = list(model.sources).index("vocals")
        vocals = stems[v]
        instrumental = np.sum([stems[i] for i in range(len(model.sources)) if i != v], axis=0)
    finally:
        del model
        _free_vram()
    n = int(round(audio.shape[1] * SAMPLE_RATE / sr))
    vocals = dsp.fit_length(dsp.match_channels(dsp.resample(vocals, model_sr, SAMPLE_RATE), 1), n)
    instrumental = dsp.fit_length(dsp.match_channels(dsp.resample(instrumental, model_sr, SAMPLE_RATE), 2), n)
    return vocals.astype(np.float32), instrumental.astype(np.float32)


def _separate_work(job: EditJob) -> None:
    d = session_dir(job.session_id)
    job.stage, job.progress = "separate", 0.05
    vocals, instrumental = separate_stems(d / "source.wav")
    _check_cancel(job)
    write_audio(d / "vocals.wav", vocals, SAMPLE_RATE)
    write_audio(d / "instrumental.wav", instrumental, SAMPLE_RATE)
    update_session(job.session_id, stage="separated", error=None)


def _asr_transcribe(path: Path) -> dict:
    from services.asr_backend import get_active_asr_backend

    return get_active_asr_backend().transcribe(str(path), word_timestamps=True)


def _transcribe_work(job: EditJob) -> None:
    session = _load_raw(job.session_id)
    d = session_dir(job.session_id)
    source = d / ("vocals.wav" if session["mode"] == "song" else "source.wav")
    job.stage, job.progress = "transcribe", 0.05
    _release_gpu()
    result = _asr_transcribe(source)
    _check_cancel(job)
    words = words_from_asr(result if isinstance(result, dict) else {})
    transcript = {"language": result.get("language") if isinstance(result, dict) else None,
                  "text": " ".join(w["word"] for w in words), "words": words}
    _write_json(d / "transcript.json", transcript)
    update_session(job.session_id, stage="ready", error=None, transcript=transcript, spans=[])


# ── render requests ──────────────────────────────────────────────────────

CROSSFADE_RANGE_MS = (5, 200)
RVC_PITCH_RANGE = (-12, 12)
VOCAL_GAIN_RANGE_DB = (-12.0, 6.0)


def _bad_request(message: str) -> AudioEditError:
    return AudioEditError("invalid_request", message, 422)


def _number(raw: dict, key: str, default, lo, hi, kind=float):
    value = raw.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad_request(f"{key} must be a number.")
    if kind is int and float(value) != int(value):
        raise _bad_request(f"{key} must be a whole number.")
    value = kind(value)
    if not lo <= value <= hi:
        raise _bad_request(f"{key} must be between {lo:g} and {hi:g}.")
    return value


def _rvc_voice_files(voice_id: str) -> tuple[str, Optional[str]]:
    from services import rvc_registry

    try:
        return rvc_registry.voice_files(voice_id)
    except (rvc_registry.VoiceNotFound, ValueError):
        raise AudioEditError("invalid_rvc_model", "That RVC voice is not installed.", 422) from None


def validate_render_request(mode: str, raw: object) -> dict:
    if not isinstance(raw, dict):
        raise _bad_request("The render request must be an object.")
    engine = raw.get("engine")
    if engine not in MODE_ENGINES[mode]:
        raise AudioEditError("invalid_engine",
                             f"{mode.capitalize()} sessions use: {', '.join(MODE_ENGINES[mode])}.", 422)
    rvc_id = raw.get("rvc_model_id") or None
    if rvc_id is not None:
        if not isinstance(rvc_id, str):
            raise _bad_request("rvc_model_id must be a string.")
        _rvc_voice_files(rvc_id)
    if raw.get("seed") is None:
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
    else:
        seed = _number(raw, "seed", 0, 0, 2 ** 31 - 1, int)
    ace = raw.get("ace_step") or {}
    if not isinstance(ace, dict):
        raise _bad_request("ace_step must be an object.")
    return {
        "engine": engine,
        "crossfade_ms": _number(raw, "crossfade_ms", 30, *CROSSFADE_RANGE_MS, kind=int),
        "rvc_model_id": rvc_id,
        "rvc_pitch": _number(raw, "rvc_pitch", 0, *RVC_PITCH_RANGE, kind=int),
        "vocal_gain_db": _number(raw, "vocal_gain_db", 0.0, *VOCAL_GAIN_RANGE_DB),
        "seed": seed,
        "ace_step": {
            "strength": _number(ace, "strength", 0.7, 0.0, 1.0),
            "padding_s": _number(ace, "padding_s", 2.0, 0.0, 10.0),
            "caption": str(ace.get("caption") or "").strip()[:300],
        },
    }


def start_render(session_id: str, raw: object) -> tuple[EditJob, dict]:
    from services import audio_edit_engines as aee

    session = load_session(session_id)
    if session.get("stage") != "ready" or not session.get("transcript"):
        raise AudioEditError("not_ready", "Transcribe the audio before rendering.", 409)
    if not session.get("spans"):
        raise AudioEditError("no_spans", "Mark at least one span to edit.", 400)
    params = validate_render_request(session["mode"], raw)
    engine = params["engine"]
    regenerates = any(s["replacement_text"] for s in session["spans"])
    if regenerates:
        status = aee.engine_status(engine)
        if not status["available"]:
            raise AudioEditError("engine_not_installed",
                                 status["reason"] or f"{aee.DISPLAY_NAMES[engine]} is not available.", 409)
        require_licence(engine)
    rvc_used = bool(params["rvc_model_id"]) and regenerates
    render_id = uuid.uuid4().hex
    with _STATE_LOCK:
        if _job_alive(session_id, "separate") or _job_alive(session_id, "transcribe"):
            raise AudioEditError("busy", "Wait for the current step to finish.", 409)
        job = _new_job(session_id, "render", render_id)
        render = {
            "id": render_id, "session_id": session_id, "engine": engine, "status": "queued",
            "progress": 0.0, "spans": [{"span_id": s["id"], "status": "queued", "path": None}
                                       for s in session["spans"]],
            "output_path": None, "error": None, "message": None,
            "created_at": time.time(), "finished_at": None,
            "commercial_use": render_commercial_use(engine, rvc=rvc_used, demucs=session["mode"] == "song"),
            "params": params, "span_snapshot": [dict(s) for s in session["spans"]],
        }
        _save_render(render)
        current = _load_raw(session_id)
        current["render_ids"] = list(current.get("render_ids") or []) + [render_id]
        _save_session(current)
    _launch(job, _render_work)
    return job, load_render(session_id, render_id)


# ── render helpers ───────────────────────────────────────────────────────

def _edit_engine(engine_id: str):
    from services.audio_edit_engines import get_edit_engine

    return get_edit_engine(engine_id)


def _engine_edit(engine, payload: dict, job: EditJob, on_progress) -> dict:
    """The one place Audio Edit asks a sidecar for synthetic audio."""
    from services.audio_edit_engines import EngineError

    try:
        reply = engine.edit(payload, on_progress)
    except EngineError as exc:
        if job.cancel.is_set():
            raise Cancelled() from None
        raise AudioEditError("engine_failed", str(exc)[:500], 500) from None
    _check_cancel(job)
    return reply


def _mark(audio: np.ndarray, sr: int) -> np.ndarray:
    """Watermark one synthetic insert (never the untouched original)."""
    import torch

    from services import watermark

    a = as_2d(audio)
    marked = watermark.mark_synthetic(torch.from_numpy(np.ascontiguousarray(a)), sr,
                                      context="audio_edit.render")
    return np.asarray(marked.detach().cpu().numpy(), dtype=np.float32).reshape(a.shape)


def _with_margins(audio, r0: int, r1: int, x: int) -> np.ndarray:
    """``audio[r0 - x : r1 + x]``, zero-padded where it runs off either end."""
    a = as_2d(audio)
    lo, hi = r0 - x, r1 + x
    seg = a[:, max(0, lo):min(a.shape[1], hi)]
    return np.pad(seg, ((0, 0), (max(0, -lo), max(0, hi - a.shape[1])))).astype(np.float32)


def _set_span(job: EditJob, states: list[dict], idx: int, status: str, path: Optional[str] = None) -> None:
    states[idx]["status"] = status
    if path:
        states[idx]["path"] = path
    update_render(job.session_id, job.render_id, spans=states, progress=round(job.progress, 3))


def _progress(job: EditJob, n: int, total: int, lo: float = 0.05, width: float = 0.6):
    def report(fraction: float) -> None:
        job.progress = round(lo + width * (n + min(max(fraction, 0.0), 1.0)) / total, 3)
    return report


def _run_rvc(src: Path, dst: Path, voice_id: str, pitch: int) -> Path:
    from services import rvc

    model_path, index_path = _rvc_voice_files(voice_id)
    try:
        rvc.get_service().convert_file(str(src), str(dst), model_path, index_path,
                                       rvc.ConversionParams(pitch=pitch))
    except (rvc.RVCUnavailable, rvc.RVCConversionError, ValueError, OSError) as exc:
        raise AudioEditError("rvc_failed", f"RVC failed: {exc}"[:500], 500) from None
    return dst


def _apply_rvc(job: EditJob, inserts: dict[int, np.ndarray], rdir: Path, params: dict) -> dict[int, np.ndarray]:
    if not params.get("rvc_model_id") or not inserts:
        return inserts
    job.stage = "rvc"
    _release_gpu()
    converted: dict[int, np.ndarray] = {}
    try:
        for idx, audio in inserts.items():
            _check_cancel(job)
            src, dst = rdir / f"span-{idx}.insert.wav", rdir / f"span-{idx}.rvc.wav"
            write_audio(src, audio, SAMPLE_RATE)
            _run_rvc(src, dst, params["rvc_model_id"], params["rvc_pitch"])
            out, sr = read_audio(dst)
            out = dsp.resample(out, sr, SAMPLE_RATE)
            converted[idx] = dsp.fit_length(dsp.match_channels(out, audio.shape[0]), audio.shape[1])
    finally:
        _release_gpu()
    return converted


def _write_output(path: Path, audio, sr: int, render: dict) -> None:
    import soundfile as sf

    a = as_2d(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    flag = "true" if render.get("commercial_use") else "false"
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=a.shape[0], format="WAV", subtype="FLOAT") as f:
        f.software = "VoiceStudio Audio Edit"
        f.comment = f"synthetic=audio_edit; engine={render['engine']}; commercial_use={flag}"
        f.write(a.T)


def _render_work(job: EditJob) -> None:
    session = _load_raw(job.session_id)
    render = load_render(job.session_id, job.render_id)
    if session["mode"] == "speech":
        audio = _render_speech(job, session, render)
    else:
        audio = _render_song(job, session, render)
    _check_cancel(job)
    job.stage, job.progress = "write", 0.97
    _write_output(render_dir(job.session_id, job.render_id) / "output.wav", audio, SAMPLE_RATE, render)
    update_render(job.session_id, job.render_id, output_path=f"renders/{job.render_id}/output.wav")


def _render_speech(job: EditJob, session: dict, render: dict) -> np.ndarray:
    sid, rid = job.session_id, job.render_id
    rdir = render_dir(sid, rid)
    params, spans = render["params"], render["span_snapshot"]
    words = session["transcript"]["words"]
    source, sr = read_audio(session_dir(sid) / "source.wav")
    duration = source.shape[1] / sr
    x = dsp.ms_to_samples(params["crossfade_ms"], sr)
    states = [dict(item) for item in render["spans"]]
    todo = [i for i, s in enumerate(spans) if s["replacement_text"]]
    inserts: dict[int, np.ndarray] = {}
    if todo:
        job.stage = "regenerate"
        _release_gpu()
        engine = _edit_engine("f5tts")
        job.engine = engine.kill
        try:
            for n, idx in enumerate(todo):
                _check_cancel(job)
                span = spans[idx]
                lo, hi = window_bounds(spans, idx, duration)
                win = span_window(span, words, duration, SPEECH_CONTEXT_S, lo, hi)
                ws, we = int(round(win.start_s * sr)), int(round(win.end_s * sr))
                window_path = rdir / f"span-{idx}.window.wav"
                write_audio(window_path, source[:, ws:we], sr)
                new_len = dsp.estimate_duration(span["original_text"], span["replacement_text"],
                                                span["end_s"] - span["start_s"])
                _set_span(job, states, idx, "running")
                reply = _engine_edit(engine, {
                    "audio_path": str(window_path), "out_path": str(rdir / f"span-{idx}.wav"),
                    "raw_text": win.raw_text, "target_text": win.target_text, "seed": params["seed"],
                    "parts": [[round(span["start_s"] - win.start_s, 4), round(span["end_s"] - win.start_s, 4)]],
                    "fix_duration": [round(new_len, 4)],
                }, job, _progress(job, n, len(todo)))
                regions = reply.get("regions") or []
                if not regions:
                    raise AudioEditError("engine_failed", "F5-TTS did not say where the new words are.", 500)
                out, out_sr = read_audio(Path(reply["path"]))
                out = dsp.resample(dsp.match_channels(out, 1), out_sr, sr)
                r0, r1 = (int(round(float(t) * sr)) for t in regions[0])
                inserts[idx] = _with_margins(out, r0, r1, x)
                _set_span(job, states, idx, "generated", f"renders/{rid}/span-{idx}.wav")
        finally:
            job.engine = None
            engine.shutdown()
    inserts = _apply_rvc(job, inserts, rdir, params)
    job.stage, job.progress = "splice", 0.85
    edits = []
    for idx, span in enumerate(spans):
        insert = None
        if idx in inserts:
            context = dsp.context_around(source, sr, span["start_s"], span["end_s"])
            insert = _mark(dsp.match_loudness(inserts[idx], context, sr), sr)
        edits.append((span["start_s"], span["end_s"], insert))
        states[idx]["status"] = "done"
    update_render(sid, rid, spans=states)
    return dsp.apply_splices(source, sr, edits, params["crossfade_ms"])


def _render_song(job: EditJob, session: dict, render: dict) -> np.ndarray:
    raise AudioEditError("not_ready", "Song renders arrive in the next task.", 409)
