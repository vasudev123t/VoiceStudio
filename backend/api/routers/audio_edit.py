"""Audio Edit: regenerate chosen words in speech or sung vocals.

Admin-only and desktop-only. Sessions live under data/audio_edit/; the
heavy work runs on background jobs the UI polls through /jobs/{id}.
"""
from __future__ import annotations

import shutil
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from api.dependencies import require_admin, require_desktop
from core.path_security import upload_suffix

router = APIRouter(
    prefix="/audio-edit",
    tags=["audio-edit"],
    dependencies=[Depends(require_admin), Depends(require_desktop)],
)

_MAX_UPLOAD_BYTES = 2 * 1024 ** 3
_CHUNK_BYTES = 1024 * 1024


def _svc():
    from services import audio_edit

    return audio_edit


def _call(fn, *args, **kwargs):
    ae = _svc()
    try:
        return fn(*args, **kwargs)
    except ae.AudioEditError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail()) from None


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


async def _save_upload(upload: UploadFile, dest) -> None:
    written = 0
    with open(dest, "wb") as fh:
        while chunk := await upload.read(_CHUNK_BYTES):
            written += len(chunk)
            if written > _MAX_UPLOAD_BYTES:
                raise _error(413, "too_large", "Uploads are limited to 2 GB.")
            fh.write(chunk)
    if written == 0:
        raise _error(400, "invalid_audio", "The uploaded file is empty.")


@router.get("/engines")
def list_engines() -> dict:
    return _call(_svc().engines_overview)


@router.post("/licence/ack")
def acknowledge_licence(body: dict = Body(...)) -> dict:
    if body.get("accepted") is not True:
        raise _error(422, "invalid_request", "Tick the acknowledgement first.")
    return _call(_svc().acknowledge_licence, body.get("engine"))


@router.delete("/licence/ack/{engine}")
def revoke_licence(engine: str) -> dict:
    return _call(_svc().revoke_licence, engine)


@router.post("/sessions")
async def create_session(
    file: UploadFile = File(...),
    mode: str = Form(...),
    name: str = Form("", max_length=120),
) -> dict:
    ae = _svc()
    ext = (upload_suffix(file.filename or "") or "").lower()
    if ext not in ae.ALLOWED_UPLOAD_EXTS:
        raise _error(415, "invalid_audio", "Upload a WAV, FLAC, MP3, M4A, OGG, Opus or video file.")
    if mode not in ae.MODES:
        raise _error(422, "invalid_mode", "Mode must be speech or song.")
    incoming = ae.root() / f".incoming-{uuid.uuid4().hex}"
    incoming.mkdir(parents=True, exist_ok=True)
    try:
        dest = incoming / f"upload{ext}"
        await _save_upload(file, dest)
        session = await run_in_threadpool(_call, ae.create_session, mode, name, dest, file.filename or "")
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    job = _call(ae.start_separation, session["id"]) if session["mode"] == "song" else None
    return {"session": _call(ae.load_session, session["id"]), "job": job.to_dict() if job else None}


@router.get("/sessions")
def list_sessions(
    mode: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    return _call(_svc().list_session_summaries, mode, limit, offset)


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return _call(_svc().load_session, session_id)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    _call(_svc().delete_session, session_id)
    return {"ok": True}


@router.post("/sessions/{session_id}/separate")
def separate(session_id: str) -> dict:
    return {"job": _call(_svc().start_separation, session_id).to_dict()}


@router.post("/sessions/{session_id}/transcribe")
def transcribe(session_id: str, body: Optional[dict[str, Any]] = Body(None)) -> dict:
    """Body ``{engine, language}`` picks the recogniser; omitted = active Whisper, auto-detect."""
    return {"job": _call(_svc().start_transcription, session_id, body).to_dict()}


# ── speech recognition engines and cloud keys ────────────────────────────

def _asr():
    from services import audio_edit_asr

    return audio_edit_asr


def _asr_call(fn, *args):
    asr = _asr()
    try:
        return fn(*args)
    except asr.AsrError as exc:
        raise _error(exc.status, exc.code, exc.message) from None


@router.get("/asr/engines")
def asr_engines() -> dict:
    return _asr_call(_asr().engines_overview)


@router.post("/asr/engines/{engine}/download")
def asr_download(engine: str) -> dict:
    return _asr_call(_asr().start_download, engine)


@router.get("/asr/keys")
def asr_keys() -> dict:
    return _asr_call(_asr().list_providers)


@router.put("/asr/keys/{provider}")
def asr_set_keys(provider: str, body: dict = Body(...)) -> dict:
    """Body ``{fields: {api_key: "…"}}``; an omitted field is kept, ``""`` clears it."""
    return _asr_call(_asr().set_provider_keys, provider, body.get("fields"))


@router.put("/sessions/{session_id}/spans")
def put_spans(session_id: str, body: dict = Body(...)) -> dict:
    return _call(_svc().set_spans, session_id, body.get("spans"))


@router.post("/sessions/{session_id}/render")
def render(session_id: str, body: dict[str, Any] = Body(...)) -> dict:
    job, record = _call(_svc().start_render, session_id, body)
    return {"job": job.to_dict(), "render": record}


@router.get("/sessions/{session_id}/renders/{render_id}")
def get_render(session_id: str, render_id: str) -> dict:
    ae = _svc()
    _call(ae.session_dir, session_id)
    return _call(ae.load_render, session_id, _call(ae._check_id, render_id, "render"))


@router.get("/sessions/{session_id}/renders/{render_id}/audio")
def render_audio(session_id: str, render_id: str) -> FileResponse:
    ae = _svc()
    path = _call(ae.render_audio_path, session_id, _call(ae._check_id, render_id, "render"))
    name = _call(ae.load_session, session_id)["name"]
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip() or "audio"
    return FileResponse(path, media_type="audio/wav", filename=f"{safe} (edited).wav")


@router.get("/sessions/{session_id}/audio/{kind}")
def session_audio(session_id: str, kind: str) -> FileResponse:
    return FileResponse(_call(_svc().audio_path, session_id, kind), media_type="audio/wav")


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _call(_svc().get_job, job_id).to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    return _call(_svc().cancel_job, job_id).to_dict()
