"""Recent voice conversions (source + converted WAV pairs) for A/B playback.

Both files live in OUTPUTS_DIR (served by the ``/audio`` mount) as
``vc_<id>.wav`` (converted) and ``vc_<id>_src.<ext>`` (an untouched copy of
the input). The index is a small JSON file capped at ``MAX_ENTRIES``; pruning
removes the oldest pairs, so conversions never grow without bound.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

MAX_ENTRIES = int(os.environ.get("OMNIVOICE_RVC_HISTORY_MAX", "50"))
_ID = re.compile(r"^[a-f0-9]{8,32}$")
_lock = threading.Lock()


def _outputs() -> Path:
    from core.config import OUTPUTS_DIR

    path = Path(OUTPUTS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path() -> Path:
    from core.config import DATA_DIR

    return Path(DATA_DIR) / "rvc_conversions.json"


def _load() -> list[dict]:
    try:
        data = json.loads(_index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in data if isinstance(e, dict) and _ID.fullmatch(str(e.get("id", "")))] \
        if isinstance(data, list) else []


def _save(entries: list[dict]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".rvc-history-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=1)
    os.replace(tmp, path)


def _remove_files(entry: dict) -> None:
    out = _outputs()
    for key in ("filename", "source_filename"):
        name = entry.get(key)
        if name and "/" not in name and "\\" not in name:
            try:
                (out / name).unlink()
            except OSError:
                pass


def add(entry: dict) -> dict:
    with _lock:
        entries = [entry] + [e for e in _load() if e.get("id") != entry["id"]]
        for stale in entries[MAX_ENTRIES:]:
            _remove_files(stale)
        _save(entries[:MAX_ENTRIES])
    return entry


def list_entries() -> list[dict]:
    with _lock:
        out = _outputs()
        return [e for e in _load() if (out / str(e.get("filename", ""))).is_file()]


def delete(entry_id: str) -> bool:
    if not _ID.fullmatch(entry_id or ""):
        return False
    with _lock:
        entries = _load()
        hit: Optional[dict] = next((e for e in entries if e.get("id") == entry_id), None)
        if hit is None:
            return False
        _remove_files(hit)
        _save([e for e in entries if e.get("id") != entry_id])
    return True


def clear() -> int:
    with _lock:
        entries = _load()
        for e in entries:
            _remove_files(e)
        _save([])
    return len(entries)
