"""Voice Lab: dataset builder (review item 4) — and the home of the A/B
scoring, fine-tuning and nonverbal-library endpoints added next to it.

* ``GET  /voice-lab/capabilities``            which stages/models are on disk
* ``POST /voice-lab/datasets``                build a dataset (multipart)
* ``GET  /voice-lab/datasets``                list datasets
* ``GET  /voice-lab/datasets/{id}``           summary + manifest + nonverbal
* ``GET  /voice-lab/datasets/{id}/audio``     one clip (``?file=clips/x.wav``)
* ``GET  /voice-lab/jobs[/{id}]``, ``POST /voice-lab/jobs/{id}/cancel``
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from core.path_security import portable_filename, upload_suffix

router = APIRouter(prefix="/voice-lab", tags=["voice-lab"])

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".mkv",
              ".mov"}
_MAX_BYTES = 4 * 1024 ** 3
_CHUNK = 1024 * 1024


async def _save(upload: UploadFile, dest: Path, limit: int) -> int:
    total = 0
    with open(dest, "wb") as out:
        while True:
            chunk = await upload.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise HTTPException(status_code=413, detail="Recordings are too large.")
            out.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")
    return total


@router.get("/capabilities")
def capabilities(asr_model: str = Query("Systran/faster-whisper-large-v3", max_length=200)):
    from services import dataset_builder as db

    return db.capabilities(asr_model)


@router.post("/datasets")
async def build_dataset(
    name: str = Form("My voice", min_length=1, max_length=80),
    language: Optional[str] = Form(None, max_length=12),
    separate: bool = Form(True),
    transcribe: bool = Form(True),
    speaker_filter: bool = Form(True),
    tag_nonverbal: bool = Form(True),
    min_clip_s: float = Form(1.5),
    max_clip_s: float = Form(12.0),
    min_quality: float = Form(0.45),
    speaker_threshold: Optional[float] = Form(None),
    files: List[UploadFile] = File(...),
    reference: Optional[UploadFile] = File(None),
):
    """Build a dataset from uploaded recordings, optionally filtered against
    a reference clip of the target speaker. Returns the job; poll
    ``GET /voice-lab/jobs/{id}``."""
    from services import dataset_builder as db

    opts = db.BuildOptions(
        name=name.strip(), language=(language or "").strip().lower() or None,
        separate=separate, transcribe=transcribe, speaker_filter=speaker_filter,
        tag_nonverbal=tag_nonverbal, min_clip_s=min_clip_s, max_clip_s=max_clip_s,
        min_quality=min_quality, speaker_threshold=speaker_threshold,
    )
    try:
        opts.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    uploads = [f for f in files if f is not None and f.filename]
    if not uploads:
        raise HTTPException(status_code=400, detail="Add at least one recording.")
    if db._active() is not None:
        raise HTTPException(status_code=409, detail="A dataset build is already running.")
    incoming = db.datasets_root() / f".incoming-{uuid.uuid4().hex[:8]}"
    incoming.mkdir(parents=True)
    sources: list[Path] = []
    ref_path: Optional[Path] = None
    try:
        total = 0
        for i, up in enumerate(uploads):
            ext = (upload_suffix(up.filename) or "").lower()
            if ext not in AUDIO_EXTS:
                raise HTTPException(status_code=415, detail=f"Unsupported file: {up.filename}")
            stem = portable_filename(Path(up.filename).stem, default="recording", max_bytes=80)
            dest = incoming / f"{i:03d}_{stem}{ext}"
            total += await _save(up, dest, _MAX_BYTES - total)
            sources.append(dest)
        if reference is not None and reference.filename:
            ext = (upload_suffix(reference.filename) or "").lower()
            if ext not in AUDIO_EXTS:
                raise HTTPException(status_code=415, detail="Unsupported reference file.")
            ref_path = incoming / f"reference{ext}"
            await _save(reference, ref_path, 512 * 1024 ** 2)
    except BaseException:
        shutil.rmtree(incoming, ignore_errors=True)
        raise

    def runner(job, srcs, options, ref):
        try:
            return db.run_build(job, srcs, options, ref)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)

    try:
        job = db.start_build(sources, opts, reference=ref_path, runner=runner)
    except RuntimeError as exc:
        shutil.rmtree(incoming, ignore_errors=True)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.public()


@router.get("/datasets")
def list_datasets():
    from services import dataset_builder as db

    return {"datasets": db.list_datasets()}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, limit: int = Query(2000, ge=1, le=20000)):
    from services import dataset_builder as db

    try:
        return db.load_dataset(dataset_id, limit=limit)
    except KeyError:
        raise HTTPException(status_code=404, detail="That dataset doesn't exist.") from None


@router.get("/datasets/{dataset_id}/audio")
def dataset_audio(dataset_id: str, file: str = Query(..., max_length=200)):
    from services import dataset_builder as db

    try:
        path = db.dataset_file(dataset_id, file)
    except KeyError:
        raise HTTPException(status_code=404, detail="Clip not found.") from None
    return FileResponse(path, media_type="audio/wav")


@router.get("/jobs")
def list_jobs():
    from services import dataset_builder as db

    return {"jobs": db.list_jobs()}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, log_tail: int = Query(80, ge=0, le=2000)):
    from services import dataset_builder as db

    try:
        return db.get_job(job_id).public(log_tail)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found.") from None


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    from services import dataset_builder as db

    try:
        return db.cancel_job(job_id).public()
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found.") from None


# ── A/B take scoring (review item 5) ────────────────────────────────────────

import asyncio as _asyncio  # noqa: E402
import json as _json  # noqa: E402
import threading as _threading  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

_score_lock = _threading.Lock()


class ScoreRequest(BaseModel):
    history_ids: List[str] = Field(..., min_length=1, max_length=12)
    reference_profile_id: Optional[str] = Field(None, max_length=64)


def _scores_path() -> Path:
    from core.config import DATA_DIR

    return Path(DATA_DIR) / "take_scores.json"


def _load_scores() -> dict:
    try:
        return _json.loads(_scores_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_scores(new: dict) -> None:
    data = _load_scores()
    data.update(new)
    if len(data) > 2000:  # bounded: keep the newest
        data = dict(sorted(data.items(), key=lambda kv: kv[1].get("scored_at", 0))[-2000:])
    tmp = _scores_path().with_suffix(".tmp")
    tmp.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_scores_path())


def _resolve_takes(history_ids: list[str], reference_profile_id: Optional[str]):
    import os

    from core.config import OUTPUTS_DIR, VOICES_DIR
    from core.db import db_conn

    def inside(root: str, name: Optional[str]) -> Optional[str]:
        if not name or os.path.basename(name) != name:
            return None
        path = os.path.realpath(os.path.join(root, name))
        return path if path.startswith(os.path.realpath(root)) and os.path.isfile(path) else None

    takes, profile_ids = [], []
    with db_conn() as conn:
        for hid in history_ids:
            row = conn.execute(
                "SELECT id, text, language, profile_id, audio_path FROM generation_history WHERE id = ?",
                (hid,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Take {hid} not found.")
            path = inside(OUTPUTS_DIR, row["audio_path"])
            if path is None:
                raise HTTPException(status_code=404, detail=f"Audio for take {hid} is missing.")
            takes.append({"id": row["id"], "path": path, "text": row["text"] or "",
                          "language": row["language"]})
            profile_ids.append(row["profile_id"])
        pid = reference_profile_id or next((p for p in profile_ids if p), None)
        reference = None
        if pid:
            prof = conn.execute("SELECT ref_audio_path FROM voice_profiles WHERE id = ?",
                                (pid,)).fetchone()
            if prof is not None:
                reference = inside(VOICES_DIR, prof["ref_audio_path"])
    return takes, reference


@router.get("/score/capabilities")
def score_capabilities():
    from services import take_scoring

    return take_scoring.capabilities()


@router.post("/score")
async def score_takes(body: ScoreRequest):
    """Score takes (SECS vs the voice's reference, WER/CER vs the take's text,
    MOS) and rank them. Results persist and are returned by ``GET /scores``."""
    import time as _time

    from services import take_scoring

    takes, reference = _resolve_takes(body.history_ids, body.reference_profile_id)
    if not _score_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Scoring is already running.")
    try:
        results = await _asyncio.to_thread(take_scoring.score_takes, takes, reference=reference)
    finally:
        _score_lock.release()
    now = _time.time()
    _save_scores({r["id"]: {**r, "scored_at": now, "reference": bool(reference)}
                  for r in results if not r.get("error")})
    return {"results": results, "reference_used": bool(reference),
            "capabilities": take_scoring.capabilities()}


@router.get("/scores")
def get_scores(ids: str = Query(..., max_length=2000)):
    wanted = [i for i in ids.split(",") if i][:200]
    data = _load_scores()
    return {"scores": {i: data[i] for i in wanted if i in data}}
