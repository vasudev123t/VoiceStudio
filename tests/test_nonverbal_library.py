"""Real nonverbal library + performance transfer (review item 7)."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from services import nonverbal_library as nvl

SR = 24000


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_NONVERBAL_DIR", str(tmp_path / "nv"))
    monkeypatch.setenv("OMNIVOICE_DATASETS_DIR", str(tmp_path / "ds"))
    prefs = {}
    from core import prefs as prefs_mod

    monkeypatch.setattr(prefs_mod, "get", lambda k, d=None: prefs.get(k, d))
    monkeypatch.setattr(prefs_mod, "set_", lambda k, v: prefs.__setitem__(k, v))
    return tmp_path, prefs


def _speech(seconds, sr=SR, f0=150.0, amp=0.3):
    t = np.arange(int(sr * seconds)) / sr
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) / 3
    return (amp * tone * (0.55 + 0.45 * np.sin(2 * np.pi * 4.5 * t))).astype("float32")


def _laugh(seconds, sr=SR, rate=5.0, amp=0.4, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    bursts = (np.sin(2 * np.pi * rate * t) > 0.2).astype("float32")
    return (amp * rng.standard_normal(len(t)) * bursts).astype("float32")


def _dataset(root, name="me-abc123"):
    ds = root / "ds" / name
    (ds / "nonverbal").mkdir(parents=True)
    (ds / "clips").mkdir()
    (ds / "dataset.json").write_text(json.dumps({"name": "Me", "options": {}}))
    sf.write(ds / "clips" / "c1.wav", _laugh(1.2), SR)
    sf.write(ds / "nonverbal" / "breath_001.wav", 0.02 * _laugh(0.4, seed=3), SR)
    rows = [{"type": "laugh", "file": "clips/c1.wav", "source": "a.mp3"},
            {"type": "breath", "file": "nonverbal/breath_001.wav", "source": "a.mp3"},
            {"type": "laugh", "file": "../../../etc/passwd"}]
    (ds / "nonverbal" / "nonverbal.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return name


def test_library_from_dataset_and_activation(env):
    root, prefs = env
    lib = nvl.from_dataset(_dataset(root))
    assert lib["added"] == 2 and lib["skipped"] == 1
    assert lib["counts"] == {"laugh": 1, "breath": 1, "sigh": 0}
    assert [x["id"] for x in nvl.list_libraries()] == [lib["id"]]
    assert nvl.plan("Hello [laughter] there") is None  # nothing active yet
    nvl.set_active(lib["id"])
    assert prefs[nvl.PREF_KEY] == lib["id"]
    with pytest.raises(KeyError):
        nvl.set_active("../x")
    with pytest.raises(KeyError):
        nvl.clip_path(lib["id"], "../library.json")
    clip = lib["clips"][0]["file"]
    nvl.remove_clip(lib["id"], clip)
    assert nvl.get_library(lib["id"])["counts"]["laugh"] == 0


def test_plan_only_takes_tags_the_library_can_serve(env):
    root, _ = env
    lib = nvl.from_dataset(_dataset(root))
    nvl.set_active(lib["id"])
    assert nvl.plan("no tags here") is None
    assert nvl.plan("Oh [sigh] fine") is None  # no sigh clips → model keeps it
    p = nvl.plan("That is [Laughter] so funny [breath] and [sigh] ok")
    assert p == [("text", "That is "), ("nv", "laugh"), ("text", " so funny "), ("nv", "breath"),
                 ("text", " and [sigh] ok")]


def test_render_splices_level_matched_clips(env):
    root, _ = env
    lib = nvl.from_dataset(_dataset(root))
    nvl.set_active(lib["id"])
    calls = []

    def gen(text):
        calls.append(text)
        return torch.from_numpy(_speech(1.0))[None]

    pieces = nvl.plan("Hi [laugh] bye")
    out = nvl.render(gen, pieces, SR, key="k")
    assert calls == ["Hi ", " bye"]
    assert out.dim() == 2 and out.shape[0] == 1
    assert out.shape[-1] > 2 * SR + int(1.0 * SR)  # two spans + ~1.2 s laugh + gaps
    mid = out[0, int(2.2 * SR):int(2.8 * SR)].numpy()
    assert np.abs(mid).max() > 0.05 and np.abs(out.numpy()).max() <= 0.99
    again = nvl.render(gen, pieces, SR, key="k")
    assert torch.equal(out, again)  # deterministic clip choice


def test_parse_regions():
    assert nvl.parse_regions("1.2-2.5, 7-8.1;9-30", 10.0) == [(1.2, 2.5), (7.0, 8.1), (9.0, 10.0)]
    assert nvl.parse_regions("", 5) == []
    with pytest.raises(ValueError):
        nvl.parse_regions("abc", 5)


def test_detector_finds_laugh_not_speech():
    x = np.concatenate([_speech(3.0), _laugh(2.0), _speech(3.0)])
    found = nvl.detect_nonverbal(x, SR)
    assert found, "laugh not detected"
    assert all(2.0 <= a and b <= 6.5 for a, b, _ in found)
    assert nvl.detect_nonverbal(_speech(6.0), SR) == []


def test_restore_regions_keep_source_and_library(env, tmp_path):
    root, _ = env
    src = tmp_path / "src.wav"
    conv = tmp_path / "conv.wav"
    source = np.concatenate([_speech(1.0), _laugh(1.0), _speech(1.0)])
    sf.write(src, source, SR)
    sf.write(conv, _speech(3.0, f0=220.0), SR)
    applied = nvl.restore_regions(src, conv, mode="keep_source", regions=[(1.0, 2.0)])
    assert applied == [{"start": 1.0, "end": 2.0, "mode": "keep_source"}]
    y, _ = sf.read(conv)
    a, b = int(1.1 * SR), int(1.9 * SR)
    assert np.corrcoef(y[a:b], source[a:b])[0, 1] > 0.9  # the source laugh is back
    assert np.allclose(y[:int(0.9 * SR)], _speech(3.0, f0=220.0)[:int(0.9 * SR)], atol=1e-4)
    lib = nvl.from_dataset(_dataset(root))
    nvl.set_active(lib["id"])
    sf.write(conv, _speech(3.0, f0=220.0), SR)
    assert nvl.restore_regions(src, conv, mode="library", regions=[(1.0, 2.0)])[0]["mode"] == "library"
    with pytest.raises(ValueError):
        nvl.restore_regions(src, conv, mode="bogus", regions=[(0.5, 1.5)])


def test_empty_library_accepts_uploaded_clips(env, tmp_path):
    lib = nvl.create_library("Mine")
    wav = tmp_path / "l.wav"
    sf.write(wav, _laugh(0.8), SR)
    clip = nvl.add_clip(lib["id"], wav, "laugh", source="l.wav")
    assert clip["kind"] == "laugh" and 0.7 < clip["duration"] < 0.9
    assert nvl.get_library(lib["id"])["counts"]["laugh"] == 1
    with pytest.raises(ValueError):
        nvl.add_clip(lib["id"], wav, "cough")
    short = tmp_path / "s.wav"
    sf.write(short, _laugh(0.05), SR)
    with pytest.raises(ValueError):
        nvl.add_clip(lib["id"], short, "laugh")
