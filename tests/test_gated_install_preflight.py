"""pyannote 401 without an HF token (review item 1d).

Installing a token-gated repo with no token must fail fast with a clear,
structured 412 (steps + fallback) and never start a download.
"""
import asyncio
import importlib

import pytest
from fastapi import HTTPException

PIPELINE = "pyannote/speaker-diarization-3.1"


@pytest.fixture
def download():
    return importlib.import_module("api.routers.setup.download")


def _spec(download, repo_id):
    return next(m for m in download.KNOWN_MODELS if m["repo_id"] == repo_id)


def test_preflight_explains_steps_and_fallback_without_token(download, monkeypatch):
    from services import token_resolver

    monkeypatch.setattr(token_resolver, "resolve", lambda *a, **k: None)
    detail = download.gated_token_preflight(_spec(download, PIPELINE))
    assert detail["code"] == "hf_token_required"
    assert detail["docs_topic"] == "PYANNOTE_LICENSE_REQUIRED"
    assert any("segmentation-3.0" in u for u in detail["access_urls"])
    assert "Settings → Hugging Face" in detail["message"]
    assert "gap-based speaker split" in detail["fallback"]
    assert "hf_" not in detail["message"], "never echo a token"


def test_preflight_passes_with_token(download, monkeypatch):
    from services import token_resolver
    from services.token_resolver import ResolvedToken

    monkeypatch.setattr(token_resolver, "resolve", lambda *a, **k: ResolvedToken(
        token="hf_x", source="app", username="u"))
    assert download.gated_token_preflight(_spec(download, PIPELINE)) is None


def test_ungated_models_skip_preflight(download, monkeypatch):
    from services import token_resolver

    monkeypatch.setattr(token_resolver, "resolve", lambda *a, **k: None)
    ungated = next(m for m in download.KNOWN_MODELS
                   if not m.get("gated") and not m.get("requires_hf_token"))
    assert download.gated_token_preflight(ungated) is None


def test_install_route_refuses_before_any_download(download, monkeypatch):
    import huggingface_hub
    from services import token_resolver

    monkeypatch.setattr(token_resolver, "resolve", lambda *a, **k: None)
    called = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda **kw: called.append(kw) or "")
    req = download.InstallModelRequest(repo_id=PIPELINE)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(download.install_model(req))
    assert exc.value.status_code == 412
    assert exc.value.detail["repo_id"] == PIPELINE
    assert called == []
