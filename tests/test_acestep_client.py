"""ACE-Step client against a fake acestep-api server on loopback."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from services import acestep_client as ac


class FakeAce(BaseHTTPRequestHandler):
    healthy = True
    fail = False
    polls = 0
    last_form = b""

    def log_message(self, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps({"data": data, "code": status, "error": None,
                           "timestamp": 0, "extra": None}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if FakeAce.healthy:
                self._json({"status": "ok"})
            else:
                self._json({"status": "starting"}, status=503)
        elif self.path.startswith("/v1/audio"):
            body = b"RIFFrepainted"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path == "/release_task":
            FakeAce.last_form = raw
            self._json({"task_id": "t1", "status": "queued"})
        elif self.path == "/query_result":
            assert json.loads(raw) == {"task_id_list": ["t1"]}
            FakeAce.polls += 1
            if FakeAce.fail:
                self._json([{"task_id": "t1", "status": 2, "result": "[]", "error": "cuda oom"}])
            elif FakeAce.polls < 2:
                self._json([{"task_id": "t1", "status": 0, "result": "[]"}])
            else:
                result = json.dumps([{"file": "/v1/audio?path=%2Ftmp%2Fout.wav", "status": 1}])
                self._json([{"task_id": "t1", "status": 1, "result": result}])
        else:
            self.send_error(404)


@pytest.fixture
def fake_server(monkeypatch):
    FakeAce.healthy, FakeAce.fail, FakeAce.polls, FakeAce.last_form = True, False, 0, b""
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAce)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OMNIVOICE_ACESTEP_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(ac, "POLL_INTERVAL_S", 0.01)
    yield server
    server.shutdown()


def test_loopback_only():
    assert ac.is_loopback_url("http://127.0.0.1:8001")
    assert ac.is_loopback_url("http://localhost:8001/")
    assert not ac.is_loopback_url("http://192.168.1.4:8001")
    assert not ac.is_loopback_url("https://api.example.com")


def test_remote_url_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("OMNIVOICE_ACESTEP_URL", "http://10.0.0.5:8001")
    with pytest.raises(ac.AceStepUnavailable, match="this computer"):
        ac.AceStepClient(tmp_path).ensure_running()


def test_repaint_round_trip(fake_server, tmp_path):
    src = tmp_path / "mix.wav"
    src.write_bytes(b"RIFFmix")
    client = ac.AceStepClient(tmp_path)
    out = client.repaint(src, 12.5, 18.25, "[verse]\nnew words", "pop, female vocal",
                         0.5, 7, tmp_path / "out.wav")
    assert out.read_bytes() == b"RIFFrepainted"
    form = FakeAce.last_form.decode("latin-1")
    for field, value in [("task_type", "repaint"), ("repainting_start", "12.500"),
                         ("repainting_end", "18.250"), ("chunk_mask_mode", "explicit"),
                         ("batch_size", "1"), ("use_random_seed", "false"), ("seed", "7"),
                         ("audio_format", "wav"), ("audio_cover_strength", "0.5")]:
        assert f'name="{field}"\r\n\r\n{value}\r\n' in form
    assert 'name="src_audio"; filename="mix.wav"' in form
    assert client.started_server is False


def test_failed_task_raises(fake_server, tmp_path):
    FakeAce.fail = True
    src = tmp_path / "mix.wav"
    src.write_bytes(b"x")
    with pytest.raises(ac.AceStepError, match="cuda oom"):
        ac.AceStepClient(tmp_path).repaint(src, 0, 1, "", "", 0.5, 1, tmp_path / "o.wav")


def test_unhealthy_server_times_out(fake_server, monkeypatch, tmp_path):
    FakeAce.healthy = False
    monkeypatch.setattr(ac, "STARTUP_TIMEOUT_S", 0.3)
    with pytest.raises(ac.AceStepUnavailable, match="did not answer"):
        ac.AceStepClient(tmp_path).ensure_running()


def test_cancel_stops_polling(fake_server, tmp_path):
    src = tmp_path / "mix.wav"
    src.write_bytes(b"x")
    client = ac.AceStepClient(tmp_path)
    client.cancel()
    with pytest.raises(ac.AceStepCancelled):
        client.repaint(src, 0, 1, "", "", 0.5, 1, tmp_path / "o.wav")


def test_spawns_local_server_with_safe_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OMNIVOICE_ACESTEP_URL", raising=False)
    monkeypatch.setenv("ACESTEP_API_KEY", "secret")
    monkeypatch.setattr(ac, "engine_python", lambda engine_id: tmp_path / "python")
    monkeypatch.setattr(ac, "engine_checkout", lambda engine_id: tmp_path / "ace")
    seen = {}

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            seen["terminated"] = True

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        return FakeProc()

    monkeypatch.setattr(ac.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ac.AceStepClient, "_wait_healthy", lambda self: None)
    client = ac.AceStepClient(tmp_path)
    base = client.ensure_running()
    env = seen["env"]
    assert seen["cmd"] == [str(tmp_path / "python"), "-m", "acestep.api_server"]
    assert seen["cwd"] == str(tmp_path / "ace")
    assert env["ACESTEP_API_HOST"] == "127.0.0.1" and base.endswith(env["ACESTEP_API_PORT"])
    assert env["ACESTEP_INIT_LLM"] == "false" and env["HF_HUB_OFFLINE"] == "1"
    assert "ACESTEP_API_KEY" not in env
    assert client.started_server is True
    client.stop()
    assert seen["terminated"] is True and client.started_server is False


def test_ace_step_install_spec():
    from services.sidecar_install import SPECS

    spec = SPECS["ace-step"]
    assert spec.env_var == "OMNIVOICE_ACESTEP_DIR" and spec.venv_args == ("--python", "3.11")
    assert not any(p.startswith("acestep-5Hz-lm") for p in spec.weights_allow_patterns)
    assert spec.weights_revision == "19671f406d603126926c1b7e2adc169acbcade22"
