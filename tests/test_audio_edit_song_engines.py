"""YingMusic and Vevo2: specs, editor wiring, and the pinned extra-file fetch."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from services import audio_edit_engines as aee
from services import sidecar_install as si

ENGINES_DIR = Path(__file__).resolve().parents[1] / "backend" / "engines"


def test_song_editors_are_registered():
    assert aee.EDITORS["yingmusic"].sidecar_dir == "yingmusic"
    assert aee.EDITORS["vevo2"].sidecar_dir == "vevo2"
    assert aee.EDITORS["vevo2"].engine_key == "vevo2"
    assert aee.get_edit_engine("yingmusic").display_name == "YingMusic-Singer-Plus"


def test_yingmusic_spec_pins_source_and_weights():
    spec = si.SPECS["yingmusic"]
    assert spec.source_revision == "baa409c2e7e5e775f09b4e92a220808f4827d2cc"
    assert spec.weights_revision == "9b3f444f2bccd77fbc03e32eb7c334fc96040b8d"
    assert spec.weights_config_names == ("config.json",)
    assert spec.source_manifest == "requirements.txt"
    assert spec.install_args[0] == "-r" and spec.install_args[1].endswith("requirements.txt")
    assert Path(spec.install_args[1]).is_file()


def test_vevo2_spec_fetches_whisper_once():
    spec = si.SPECS["vevo2"]
    assert spec.weights_subdir == "checkpoints/Vevo2"
    assert spec.source_required_path == "models/svc/vevo2/vevo2_utils.py"
    (whisper,) = spec.extra_files
    assert whisper.path == "checkpoints/whisper/medium.pt"
    assert whisper.sha256 in whisper.url
    assert not any("optimizer" in p or "rng_state" in p for p in spec.weights_allow_patterns)


@pytest.fixture
def extra_spec(tmp_path, monkeypatch):
    payload = b"whisper-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    spec = si.SidecarSpec(
        engine_id="extra-test", display_name="Extra", repo_url="", tarball_url="",
        checkout_dirname="extra-test", env_var="OMNIVOICE_EXTRA_TEST_DIR", probe_module="x",
        has_source=False, venv_args=("--python", "3.11"),
        extra_files=(si.ExtraFile(f"https://example.invalid/{digest}/m.pt", "ckpt/m.pt", digest),),
    )
    monkeypatch.setattr(si, "managed_checkout", lambda s: tmp_path)
    calls: list[str] = []

    def fake_stream(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(fake_stream.body)

    fake_stream.body = payload
    monkeypatch.setattr(si, "_stream_to_file", fake_stream)
    return spec, calls, fake_stream


def test_extra_file_downloads_and_verifies(extra_spec, tmp_path):
    spec, calls, _ = extra_spec
    si._ensure_extra_files(spec, {"log": []})
    assert (tmp_path / "ckpt" / "m.pt").read_bytes() == b"whisper-bytes"
    si._ensure_extra_files(spec, {"log": []})
    assert len(calls) == 1


def test_extra_file_rejects_a_bad_checksum(extra_spec, tmp_path):
    spec, _, fake_stream = extra_spec
    fake_stream.body = b"tampered"
    with pytest.raises(si._StepError, match="checksum"):
        si._ensure_extra_files(spec, {"log": []})
    assert not (tmp_path / "ckpt" / "m.pt").exists()


@pytest.mark.parametrize("name", ["yingmusic", "vevo2"])
def test_song_sidecars_never_download_at_render(name):
    source = (ENGINES_DIR / name / "main.py").read_text(encoding="utf-8")
    assert "snapshot_download" not in source and "hf_hub_download" not in source
    assert "HF_HUB_OFFLINE" in source
