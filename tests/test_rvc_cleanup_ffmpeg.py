"""Applio light-cleanup ffmpeg selection (review item 1a).

Applio's ``audio_cleanup._ffmpeg()`` prefers the FFmpeg 4.3.2 bundled in its
root, whose ``afftdn`` lacks ``gs`` → "Option 'gs' not found". VoiceStudio
must probe candidates, hand a capable binary to the worker, and pre-clean
itself when none is capable.
"""
import json
import subprocess
import sys
import textwrap
import wave
from pathlib import Path

import pytest
import torch

_AFFTDN_NEW = "afftdn AVOptions:\n   tn                <boolean>    ..F.A....T. track noise\n   gs                <int>        ..F.A....T. set gain smooth radius\n"
_AFFTDN_OLD = "afftdn AVOptions:\n  tn                <boolean>    ..F.A....T track noise (default false)\n"


@pytest.fixture
def rvc(monkeypatch, tmp_path):
    import services.rvc as rvc_mod

    monkeypatch.setenv("OMNIVOICE_RVC_VOICES_DIR", str(tmp_path / "voices"))
    for var in ("OMNIVOICE_APPLIO_DIR", "OMNIVOICE_APPLIO_PYTHON", "OMNIVOICE_RVC_DEVICE",
                "OMNIVOICE_RVC_FFMPEG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rvc_mod, "_pref", lambda key, default=None: default)
    monkeypatch.setattr(rvc_mod, "managed_applio_dir", lambda: tmp_path / "no-managed-applio")
    rvc_mod._FFMPEG_PROBES.clear()
    rvc_mod.reset_service()
    yield rvc_mod
    rvc_mod.reset_service()
    rvc_mod._FFMPEG_PROBES.clear()


def _fake_run(versions: dict):
    """subprocess.run stand-in: path → (version, afftdn help text)."""
    def run(cmd, **kw):
        version, helptext = versions[str(cmd[0])]
        out = (f"ffmpeg version {version} Copyright\n" if "-version" in cmd else helptext)
        return subprocess.CompletedProcess(cmd, 0, out.encode(), b"")
    return run


def test_probe_detects_gs_support(rvc, monkeypatch, tmp_path):
    new, old = tmp_path / "new.exe", tmp_path / "old.exe"
    new.write_bytes(b"x")
    old.write_bytes(b"x")
    monkeypatch.setattr(rvc.subprocess, "run", _fake_run({
        str(new): ("9.0.1-essentials", _AFFTDN_NEW), str(old): ("n4.3.2-160", _AFFTDN_OLD)}))
    a, b = rvc.probe_ffmpeg(str(new)), rvc.probe_ffmpeg(str(old))
    assert a["supports_gs"] is True and a["version"] == "9.0.1-essentials"
    assert b["supports_gs"] is False and b["version"] == "n4.3.2-160"


def test_status_prefers_capable_ffmpeg_over_applio_bundled(rvc, monkeypatch, tmp_path):
    root = tmp_path / "Applio"
    root.mkdir()
    name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    bundled = root / name
    bundled.write_bytes(b"x")
    system = tmp_path / "bin" / name
    system.parent.mkdir()
    system.write_bytes(b"x")
    monkeypatch.setattr(rvc.shutil, "which", lambda n: str(system))
    monkeypatch.setattr("services.ffmpeg_utils.find_ffmpeg", lambda: None, raising=False)
    monkeypatch.setattr(rvc.subprocess, "run", _fake_run({
        str(system): ("9.0.1", _AFFTDN_NEW), str(bundled): ("n4.3.2", _AFFTDN_OLD)}))
    info = rvc.cleanup_ffmpeg_status(root)
    assert info["selected"]["path"] == str(system)
    assert info["applio_bundled"]["supports_gs"] is False
    assert info["fallback"] is None


def test_status_reports_fallback_when_nothing_supports_gs(rvc, monkeypatch, tmp_path):
    root = tmp_path / "Applio"
    root.mkdir()
    name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    (root / name).write_bytes(b"x")
    monkeypatch.setattr(rvc.shutil, "which", lambda n: None)
    monkeypatch.setattr("services.ffmpeg_utils.find_ffmpeg", lambda: None, raising=False)
    monkeypatch.setattr(rvc.subprocess, "run", _fake_run({str(root / name): ("n4.3.2", _AFFTDN_OLD)}))
    info = rvc.cleanup_ffmpeg_status(root)
    assert info["selected"] is None and info["fallback"] == "voicestudio_afftdn"


def _stub_applio(root: Path) -> Path:
    for pkg in ("rvc", "rvc/infer", "rvc/configs", "rvc/lib", "rvc/lib/tools"):
        (root / pkg).mkdir(parents=True, exist_ok=True)
        (root / pkg / "__init__.py").write_text("")
    (root / "rvc" / "configs" / "config.py").write_text(textwrap.dedent("""
        class _Cfg:
            device = "cpu"
            gpu_name = None
        def inference_config():
            return _Cfg()
    """))
    (root / "rvc" / "lib" / "tools" / "audio_cleanup.py").write_text(textwrap.dedent("""
        def _ffmpeg():
            return "applio-bundled-ffmpeg"
    """))
    (root / "rvc" / "infer" / "infer.py").write_text(textwrap.dedent("""
        import json, wave
        from rvc.lib.tools import audio_cleanup
        class VoiceConverter:
            def __init__(self, config=None):
                self.config = config
                self.tgt_sr = None
            def cleanup_model(self):
                pass
            def convert_audio(self, audio_input_path, audio_output_path, model_path,
                              index_path, **kw):
                self.tgt_sr = 40000
                with wave.open(audio_output_path, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(40000)
                    wf.writeframes(b"\\x00\\x00" * 400)
                with open(audio_output_path + ".json", "w") as fh:
                    json.dump({"input": audio_input_path, "ffmpeg": audio_cleanup._ffmpeg(),
                               **kw}, fh)
    """))
    (root / "core.py").write_text("")
    return root


def _wav(path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)


def _fake_pth(path):
    torch.save({"weight": {"w": torch.zeros(2)}, "config": [1, 2, 40000], "version": "v2",
                "f0": 1, "sr": "40k"}, str(path))
    return path


def test_worker_uses_selected_ffmpeg_for_light_cleanup(rvc, monkeypatch, tmp_path):
    from services import rvc_registry

    root = _stub_applio(tmp_path / "Applio")
    capable = tmp_path / "ffmpeg-9.exe"
    capable.write_bytes(b"x")
    monkeypatch.setenv("OMNIVOICE_APPLIO_DIR", str(root))
    monkeypatch.setenv("OMNIVOICE_APPLIO_PYTHON", sys.executable)
    monkeypatch.setattr(rvc, "cleanup_ffmpeg_status", lambda root=None: {
        "selected": {"path": str(capable), "supports_gs": True}, "fallback": None})
    voice = rvc_registry.import_from_paths("A", _fake_pth(tmp_path / "a.pth"))
    src, out = tmp_path / "in.wav", tmp_path / "out.wav"
    _wav(src)
    svc = rvc.get_service()
    try:
        svc.convert_voice(voice["id"], str(src), str(out), rvc.ConversionParams(clean_audio=True))
        sent = json.loads(Path(str(out) + ".json").read_text())
        assert sent["ffmpeg"] == str(capable), "worker must not fall back to Applio's ffmpeg 4.3"
        assert sent["cleanup_mode"] == "light"
    finally:
        rvc.reset_service()


def test_no_capable_ffmpeg_precleans_in_voicestudio(rvc, monkeypatch, tmp_path):
    from services import rvc_registry

    root = _stub_applio(tmp_path / "Applio")
    monkeypatch.setenv("OMNIVOICE_APPLIO_DIR", str(root))
    monkeypatch.setenv("OMNIVOICE_APPLIO_PYTHON", sys.executable)
    monkeypatch.setattr(rvc, "cleanup_ffmpeg_status", lambda root=None: {
        "selected": None, "fallback": "voicestudio_afftdn"})
    cleaned = []

    def fake_preclean(inp, outp, strength=12.0):
        Path(outp).write_bytes(Path(inp).read_bytes())
        cleaned.append((inp, outp, strength))
        return outp

    monkeypatch.setattr(rvc, "preclean_audio", fake_preclean)
    voice = rvc_registry.import_from_paths("A", _fake_pth(tmp_path / "a.pth"))
    src, out = tmp_path / "in.wav", tmp_path / "out.wav"
    _wav(src)
    svc = rvc.get_service()
    try:
        svc.convert_voice(voice["id"], str(src), str(out),
                          rvc.ConversionParams(clean_audio=True, clean_strength=9.0))
        sent = json.loads(Path(str(out) + ".json").read_text())
        assert "cleanup_mode" not in sent, "Applio cleanup must be skipped (would fail on gs)"
        assert cleaned and cleaned[0][2] == 9.0
        assert sent["input"] == cleaned[0][1]
    finally:
        rvc.reset_service()
