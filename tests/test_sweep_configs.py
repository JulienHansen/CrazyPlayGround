"""Contract tests for the training sweeps. CPU only, no Isaac Lab, no GPU.

The sweeps cost ~5 h of GPU each, and a mistyped Hydra key only fails after the
environment has been built. These tests catch grid and override errors in
milliseconds.
"""

import ast
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "execution", "sim2real", "experiments")
HOVERING = os.path.join(
    ROOT, "source", "CrazyPlayGround", "CrazyPlayGround", "tasks", "direct", "hovering",
)
# RobustQuadcopterEnvCfg inherits from QuadcopterEnvCfg, so both files declare
# fields a sweep may override.
ENV_SRCS = [os.path.join(HOVERING, "vel_hovering_robust.py"),
            os.path.join(HOVERING, "vel_hovering.py")]

SWEEPS = [("configs", 24), ("configs_v2", 12), ("configs_v3", 12)]


@pytest.fixture(scope="module")
def cfg_modules():
    sys.path.insert(0, EXP)
    import importlib
    mods = {name: importlib.import_module(name) for name, _ in SWEEPS}
    yield mods
    sys.path.remove(EXP)


@pytest.fixture(scope="module")
def cfg_fields():
    """Field names declared on RobustQuadcopterEnvCfg and its base class.

    Parsed rather than imported: the classes need Isaac Lab, which is not
    installed on the CI runner.
    """
    names = set()
    for src in ENV_SRCS:
        for node in ast.walk(ast.parse(open(src).read())):
            if not (isinstance(node, ast.ClassDef) and node.name.endswith("EnvCfg")):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    assert {"history_len", "add_noise"} <= names, "parser missed config fields"
    return names


@pytest.mark.parametrize("name,expected", SWEEPS)
def test_grid_size_and_unique_tags(cfg_modules, name, expected):
    runs = cfg_modules[name].build_sweep()
    assert len(runs) == expected
    tags = [r["tag"] for r in runs]
    assert len(set(tags)) == len(tags), f"duplicate tags in {name}: {tags}"
    for r in runs:
        assert r["overrides"], f"{r['tag']} has no overrides"
        assert isinstance(r["seed"], int)


@pytest.mark.parametrize("name,_", SWEEPS)
def test_every_override_targets_a_real_config_field(cfg_modules, cfg_fields, name, _):
    """A typo'd `env.foo` override makes Hydra abort after the env is built."""
    for r in cfg_modules[name].build_sweep():
        for key in r["overrides"]:
            assert key.startswith("env."), f"{name}/{r['tag']}: unexpected scope {key}"
            field = key.split(".", 1)[1]
            assert field in cfg_fields, f"{name}/{r['tag']}: no config field '{field}'"


@pytest.mark.parametrize("name,_", SWEEPS)
def test_bools_are_hydra_strings(cfg_modules, name, _):
    """Hydra parses the override string, so a Python bool would arrive as a str."""
    for r in cfg_modules[name].build_sweep():
        for key, val in r["overrides"].items():
            assert not isinstance(val, bool), (
                f"{name}/{r['tag']}: {key} is a Python bool; use 'True'/'False'"
            )


@pytest.mark.parametrize("index,leaderboard", [
    ("sweep_index.json", "leaderboard.json"),
    ("sweep_index_v2.json", "leaderboard_v2.json"),
])
def test_leaderboard_tags_come_from_the_recorded_sweep(index, leaderboard):
    idx = json.load(open(os.path.join(EXP, index)))
    lb = json.load(open(os.path.join(EXP, leaderboard)))
    trained = set(idx["runs"])
    for row in lb["rows"]:
        assert row["tag"] in trained, f"{leaderboard}: {row['tag']} is not in {index}"
        assert row["score"] == row["score"], f"{leaderboard}: {row['tag']} scored NaN"


@pytest.mark.parametrize("index", ["sweep_index.json", "sweep_index_v2.json", "sweep_index_v3.json"])
def test_sweep_index_records_a_checkpoint_per_successful_run(index):
    idx = json.load(open(os.path.join(EXP, index)))
    assert idx["task"] == "Vel-Hovering-Robust"
    for tag, run in idx["runs"].items():
        if run.get("status") != "ok":
            continue
        assert run["checkpoint"].endswith(".pt"), f"{tag}: {run['checkpoint']}"


def test_recorded_tags_match_the_grid_that_produced_them():
    """The shipped indices must describe the grids still in the repo."""
    sys.path.insert(0, EXP)
    import importlib
    try:
        for module, index in [("configs", "sweep_index.json"),
                              ("configs_v2", "sweep_index_v2.json"),
                              ("configs_v3", "sweep_index_v3.json")]:
            grid = {r["tag"] for r in importlib.import_module(module).build_sweep()}
            recorded = set(json.load(open(os.path.join(EXP, index)))["runs"])
            assert recorded <= grid, f"{index} has tags absent from {module}: {recorded - grid}"
    finally:
        sys.path.remove(EXP)
