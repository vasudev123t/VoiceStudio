"""Loopback admin routes need the desktop's per-launch token (review item 1e)."""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core import local_admin

TOKEN = "t" * 43


@pytest.fixture
def app():
    from api.dependencies import require_admin, require_desktop

    application = FastAPI()

    @application.post("/admin", dependencies=[Depends(require_admin)])
    def admin():
        return {"ok": True}

    @application.post("/desktop", dependencies=[Depends(require_desktop)])
    def desktop():
        return {"ok": True}

    @application.get("/public")
    def public():
        return {"ok": True}

    yield application
    local_admin.set_token_for_tests(None)


def _client(app):
    return TestClient(app, client=("127.0.0.1", 50000))


def test_without_token_loopback_stays_trusted(app):
    local_admin.set_token_for_tests(None)
    assert _client(app).post("/admin").status_code == 200


def test_with_token_loopback_admin_requires_it(app):
    local_admin.set_token_for_tests(TOKEN)
    client = _client(app)
    denied = client.post("/admin")
    assert denied.status_code == 403
    assert "local admin token required" in denied.json()["detail"]
    assert client.post("/admin", headers={local_admin.HEADER: "wrong"}).status_code == 403
    assert client.post("/admin", headers={local_admin.HEADER: TOKEN}).status_code == 200
    assert client.post("/desktop").status_code == 403
    assert client.post("/desktop", headers={local_admin.HEADER: TOKEN}).status_code == 200
    # Consumption routes are untouched.
    assert client.get("/public").status_code == 200


def test_token_is_scrubbed_from_environment(monkeypatch):
    monkeypatch.setenv(local_admin.ENV_VAR, TOKEN)
    monkeypatch.setattr(local_admin, "_loaded", False)
    monkeypatch.setattr(local_admin, "_token", None)
    try:
        assert local_admin.configured() is True
        import os

        assert local_admin.ENV_VAR not in os.environ, "children must not inherit the token"
        assert local_admin.headers() == {local_admin.HEADER: TOKEN}
    finally:
        local_admin.set_token_for_tests(None)


def test_short_tokens_are_ignored(monkeypatch):
    monkeypatch.setenv(local_admin.ENV_VAR, "short")
    monkeypatch.setattr(local_admin, "_loaded", False)
    try:
        assert local_admin.configured() is False
    finally:
        local_admin.set_token_for_tests(None)


def test_voice_conversion_setup_is_admin_gated(monkeypatch):
    from api.routers import voice_conversion

    application = FastAPI()
    application.include_router(voice_conversion.router)
    local_admin.set_token_for_tests(TOKEN)
    try:
        r = _client(application).post("/voice-conversion/setup", json={"confirm": False})
        assert r.status_code == 403
        r = _client(application).post("/voice-conversion/setup", json={"confirm": False},
                                      headers={local_admin.HEADER: TOKEN})
        assert r.status_code == 400  # reached the handler: confirm=false refused
    finally:
        local_admin.set_token_for_tests(None)
