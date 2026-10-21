"""Per-launch secret that guards admin routes even on loopback.

Loopback used to be trusted unconditionally, so any local process (a browser
exploit helper, a rogue npm script, malware running as the user) could call
``/system/set-env``, ``/voice-conversion/setup`` (git clone + pip install) or
the model installers. When the desktop shell launches the backend it now
passes a random token in ``OMNIVOICE_LOCAL_ADMIN_TOKEN``; the Electron main
process adds it as ``X-VoiceStudio-Local-Token`` to every request it proxies,
and admin dependencies require it from loopback callers too.

* The token is read once and removed from ``os.environ`` so engine sidecars,
  the Applio worker and other children never inherit it.
* Without the env var (a backend started by hand, Docker, tests) behaviour is
  unchanged: loopback stays trusted, exactly as documented in
  ``docs/api-auth.md``.
* Only HTTP requests are checked; WebSocket handshakes keep their own guards.
"""
from __future__ import annotations

import hmac
import os
import threading

ENV_VAR = "OMNIVOICE_LOCAL_ADMIN_TOKEN"
HEADER = "x-voicestudio-local-token"
_MIN_LENGTH = 32

_lock = threading.Lock()
_token: str | None = None
_loaded = False


def _load() -> str | None:
    global _token, _loaded
    with _lock:
        if not _loaded:
            raw = os.environ.pop(ENV_VAR, "").strip()
            _token = raw if len(raw) >= _MIN_LENGTH else None
            _loaded = True
        return _token


def configured() -> bool:
    """True when this backend was launched with a per-session admin token."""
    return _load() is not None


def set_token_for_tests(value: str | None) -> None:
    global _token, _loaded
    with _lock:
        _token = value
        _loaded = True


def headers() -> dict:
    """Header for in-process self-calls (MCP ASGI client) — never logged."""
    token = _load()
    return {HEADER: token} if token else {}


def request_authorized(request) -> bool:
    """Whether a loopback HTTP request may use admin routes.

    True when no token is configured, for non-HTTP scopes, or when the
    request presents the exact token (constant-time compare).
    """
    token = _load()
    if token is None:
        return True
    scope = getattr(request, "scope", None) or {}
    if scope.get("type", "http") != "http":
        return True
    presented = ""
    try:
        presented = request.headers.get(HEADER, "") or ""
    except Exception:  # noqa: BLE001
        presented = ""
    return hmac.compare_digest(presented.encode("utf-8"), token.encode("utf-8"))
