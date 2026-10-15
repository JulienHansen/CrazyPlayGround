"""Deployment-safety guards.

A policy trained with a K-step observation window expects 6*K + 3*M inputs. Flying
a checkpoint with the wrong window would produce a wrongly-shaped policy, so the
mismatch must fail at load time, on the ground.
"""

import glob
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skrl")

from check_policy import load_agent  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(REPO, "execution", "sim2real", "flight_checkpoints")


def _manifest():
    p = os.path.join(CKPT_DIR, "manifest.json")
    if not os.path.exists(p):
        pytest.skip("flight checkpoints not present")
    return json.load(open(p))


def test_every_shipped_checkpoint_loads_with_its_declared_window():
    for m in _manifest():
        ckpt = os.path.join(CKPT_DIR, m["file"])
        agent = load_agent(ckpt, torch.device("cpu"), m["k"], m["m"])
        assert agent is not None, f"{m['id']} failed to load"


def test_wrong_window_fails_loudly_rather_than_flying():
    """The safety property: a mismatched --history-len must raise, not silently
    build a differently-shaped network."""
    man = _manifest()
    windowed = next((m for m in man if m["k"] > 1), None)
    if windowed is None:
        pytest.skip("no windowed checkpoint shipped")
    ckpt = os.path.join(CKPT_DIR, windowed["file"])
    with pytest.raises(Exception):
        load_agent(ckpt, torch.device("cpu"), 1, 0)     # wrong K


def test_observation_width_matches_the_declared_window():
    """6*K + 3*M, read off the checkpoint rather than trusted from the filename."""
    for m in _manifest():
        sd = torch.load(os.path.join(CKPT_DIR, m["file"]),
                        map_location="cpu", weights_only=False)["policy"]
        sd = sd.get("state_dict", sd)
        w = sd["net_container.0.weight"]
        assert w.shape[1] == 6 * m["k"] + 3 * m["m"], (
            f"{m['id']}: checkpoint input {w.shape[1]} != 6*{m['k']} + 3*{m['m']}")


def test_manifest_hashes_match_the_files():
    """The run identity used to relabel flights comes from these hashes."""
    import hashlib
    for m in _manifest():
        h = hashlib.sha256(open(os.path.join(CKPT_DIR, m["file"]), "rb").read()).hexdigest()
        assert h.startswith(m["sha256_16"]), f"{m['id']} hash does not match manifest"
