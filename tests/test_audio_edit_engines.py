"""Audio Edit engine sidecars: the edit op round-trips through a real child process."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from services import audio_edit_engines as aee

ENGINES_DIR = Path(__file__).resolve().parents[1] / "backend" / "engines"

FAKE_SIDECAR = textwrap.dedent(f"""
    import os, shutil, sys
    sys.path.insert(0, {str(ENGINES_DIR)!r})
    from _edit_protocol import serve

    def load():
        return {{"loads": 1}}

    def edit(model, msg, progress):
        if msg.get("fail"):
            raise RuntimeError("boom")
        progress(0.5)
        shutil.copyfile(msg["audio_path"], msg["out_path"])
        return {{"path": msg["out_path"], "sample_rate": 44100, "regions": [[0.1, 0.2]],
                 "loads": model["loads"]}}

    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    sys.exit(serve("fake", load, edit, os.fdopen(_frame_fd, "wb")))
""")


@pytest.fixture
def fake_engine(tmp_path):
    script = tmp_path / "fake_sidecar.py"
    script.write_text(FAKE_SIDECAR, encoding="utf-8")

    class FakeEdit(aee.EditSidecar):
        id = "audio-edit-fake"
        display_name = "Fake editor"
        engine_key = "f5tts"

        @classmethod
        def venv_python(cls) -> Path:
            return Path(sys.executable)

        @classmethod
        def sidecar_script(cls) -> Path:
            return script

    engine = FakeEdit()
    yield engine
    engine.shutdown()


def test_edit_round_trip_reports_progress(fake_engine, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF....fake")
    seen: list[float] = []
    reply = fake_engine.edit({"audio_path": str(src), "out_path": str(tmp_path / "out.wav")}, seen.append)
    assert reply["op"] == "audio" and Path(reply["path"]).read_bytes() == b"RIFF....fake"
    assert reply["regions"] == [[0.1, 0.2]]
    assert seen == [0.5]


def test_edit_error_frame_raises_engine_error(fake_engine, tmp_path):
    with pytest.raises(aee.EngineError, match="RuntimeError: boom"):
        fake_engine.edit({"fail": True, "audio_path": "x", "out_path": str(tmp_path / "o.wav")})


def test_kill_then_edit_respawns(fake_engine, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"x")
    fake_engine.edit({"audio_path": str(src), "out_path": str(tmp_path / "a.wav")})
    fake_engine.kill()
    assert fake_engine._proc is None
    reply = fake_engine.edit({"audio_path": str(src), "out_path": str(tmp_path / "b.wav")})
    assert reply["loads"] == 1 and Path(reply["path"]).is_file()


def test_missing_venv_reports_not_installed(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_F5TTS_DIR", raising=False)
    status = aee.engine_status("f5tts")
    assert status == {"installed": False, "install_id": "f5tts", "needs_gpu": False,
                      "available": False, "reason": "Install F5-TTS to use it."}
    ok, reason = aee.F5TTSEdit.is_available()
    assert ok is False and "F5-TTS" in reason
    with pytest.raises(aee.EngineError):
        aee.F5TTSEdit.venv_python()


def test_gpu_engines_need_a_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(aee, "engine_python", lambda engine_id: tmp_path / "python")
    monkeypatch.setattr(aee, "has_gpu", lambda: False)
    assert aee.engine_status("f5tts")["available"] is True
    vevo = aee.engine_status("vevo2")
    assert vevo["installed"] is True and vevo["available"] is False and "GPU" in vevo["reason"]
    monkeypatch.setattr(aee, "has_gpu", lambda: True)
    assert aee.engine_status("vevo2")["available"] is True


def test_get_edit_engine_is_a_singleton():
    assert aee.get_edit_engine("f5tts") is aee.get_edit_engine("f5tts")
    with pytest.raises(KeyError):
        aee.get_edit_engine("ace_step")


def test_f5tts_install_spec():
    from services.sidecar_install import SPECS

    spec = SPECS["f5tts"]
    assert spec.has_source is False and spec.env_var == "OMNIVOICE_F5TTS_DIR"
    assert "--python" in spec.venv_args and "f5-tts==1.1.22" in spec.install_args
    assert spec.docs_path == "docs/engines/f5tts.md"
