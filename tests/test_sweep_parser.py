"""parser.py must expand sweep files predictably and refuse ambiguous ones.

Two runs are comparable only if each was trained on what the sweep file says.
These tests pin the merge rule, the expansion, the keys the W&B check relies
on, and the errors that stop a malformed file before any GPU time is spent.
"""

import glob
import os
import re
import sys

import pytest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)
from parser import SweepError, expand, load_sweep  # noqa: E402


def sweep(*experiments, **defaults):
    base = {"trainer": "skrl", "env_name": "Vel-Hovering", "seed": [1, 2]}
    return {
        "version": 1,
        "wandb": {"project": "test"},
        "defaults": {**base, **defaults},
        "experiments": list(experiments),
    }


def test_dicts_merge_and_lists_stay_values():
    s = expand(sweep(
        {"name": "a", "env_kwargs": {"noise_std": 0.1}},
        env_kwargs={"add_noise": True, "noise_std": 0.0, "mass_range": [0.9, 1.1]},
    ))
    assert len(s.runs) == 2  # one per seed: the range is a value, not a sweep
    assert s.runs[0].config["env_kwargs"] == {"add_noise": True, "noise_std": 0.1, "mass_range": [0.9, 1.1]}


def test_sweep_expands_every_combination_times_seeds():
    s = expand(sweep({"name": "scan", "sweep": {"lr": [1e-4, 3e-4], "env_kwargs.noise_std": [0.0, 0.02]}}))
    assert len(s.runs) == 2 * 2 * 2
    assert len({r.run_key for r in s.runs}) == 8
    groups = {r.group for r in s.runs}
    assert len(groups) == 4
    for group in groups:
        members = [r for r in s.runs if r.group == group]
        assert sorted(r.seed for r in members) == [1, 2]
        assert len({r.config_key for r in members}) == 1
    run = next(r for r in s.runs if r.sweep_values == {"lr": 1e-4, "env_kwargs.noise_std": 0.02})
    assert run.group == f"scan/lr=0.0001,env_kwargs.noise_std=0.02 [{run.config_key}]"
    assert run.name == f"scan/lr=0.0001,env_kwargs.noise_std=0.02/s{run.seed}"


def test_same_label_in_two_sweep_files_gives_two_groups():
    # two hovering sweep files log to the same W&B project, both with an "lr-scan"
    a = expand(sweep({"name": "lr-scan", "sweep": {"lr": [1e-4]}}, env_kwargs={"noise_std": 0.01}))
    b = expand(sweep({"name": "lr-scan", "sweep": {"lr": [1e-4]}}, env_kwargs={"noise_std": 0.02}))
    assert a.runs[0].name == b.runs[0].name
    assert a.runs[0].group != b.runs[0].group


def test_run_shared_by_two_experiments_is_produced_once():
    s = expand(sweep({"name": "base"}, {"name": "scan", "sweep": {"lr": [5e-4, 1e-3]}}, lr=5e-4))
    assert len(s.runs) == 4  # scan's lr=5e-4 is base again
    assert [r.also_in for r in s.runs if r.experiment == "base"] == [["scan"], ["scan"]]


def test_group_follows_the_config_not_the_experiment():
    s = expand(sweep({"name": "a"}, {"name": "b", "seed": [2, 3]}))
    assert sorted(r.seed for r in s.runs) == [1, 2, 3]
    assert {r.group for r in s.runs} == {f"a [{s.runs[0].config_key}]"}


def test_run_key_ignores_spelling_but_not_meaning():
    a = expand(sweep({"name": "x", "total_timesteps": 100e6, "env_kwargs": {"a": 1, "b": 2.0}}))
    b = expand(sweep({"name": "y", "env_kwargs": {"b": 2, "a": 1}, "total_timesteps": 100000000}))
    assert [r.run_key for r in a.runs] == [r.run_key for r in b.runs]
    # values are passed on as written: Isaac Lab refuses an int for a float field
    assert isinstance(a.runs[0].config["total_timesteps"], float)
    assert isinstance(a.runs[0].config["env_kwargs"]["b"], float)

    bumped = sweep({"name": "x", "total_timesteps": 100e6, "env_kwargs": {"a": 1, "b": 2.0}})
    bumped["version"] = 2
    assert expand(bumped).runs[0].run_key != a.runs[0].run_key


@pytest.mark.parametrize(
    "data, message",
    [
        (sweep({"name": "a"}, {"name": "a"}), "another experiment already has this name"),
        (sweep({"name": "a b"}), "'name' must match"),
        (sweep({"name": "a", "sweep": {"seed": [1, 2]}}), "'seed' is always expanded"),
        (sweep({"name": "a", "lr": 1e-4, "sweep": {"lr": [1e-4, 3e-4]}}), "also set by the experiment"),
        (sweep({"name": "a", "env_kwargs": {"x": 1}, "sweep": {"env_kwargs": [{"y": 1}]}}), "also set by the experiment"),
        (sweep({"name": "a", "sweep": {"env_kwargs": [{}], "env_kwargs.x": [1]}}), "set the same setting"),
        (sweep({"name": "a", "sweep": {"lr": []}}), "non-empty list of values"),
        (sweep({"name": "a", "sweep": {"lr": 3e-4}}), "non-empty list of values"),
        (sweep({"name": "a", "sweep": {"lr": [1e-4, 0.0001]}}), "lists the same value twice"),
        (sweep({"name": "a", "sweep": {"lr.x": [1]}}, lr=1e-4), "'lr' is not an object"),
        (sweep({"name": "a", "trainer": "skrll"}), "'trainer' must be one of"),
        (sweep({"name": "a", "env_name": ""}), "'env_name' must be"),
        (sweep({"name": "a", "seed": [1, 1]}), "same seed twice"),
        (sweep({"name": "a", "seed": [True]}), "'seed' must be"),
        ({**sweep({"name": "a"}), "experiment": []}, "unknown key(s) ['experiment']"),
        ({**sweep({"name": "a"}), "version": 0}, "must be an int >= 1"),
        ({**sweep({"name": "a"}), "wandb": {"entity": "x"}}, "wandb.project"),
    ],
)
def test_malformed_sweeps_are_refused(data, message):
    with pytest.raises(SweepError, match=re.escape(message)):
        expand(data)


def test_duplicate_json_keys_are_refused(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"version": 1, "version": 2}')
    with pytest.raises(SweepError, match="appears twice"):
        load_sweep(str(path))


@pytest.mark.parametrize("path", glob.glob(os.path.join(REPO, "config", "sweep", "**", "*.json"), recursive=True))
def test_committed_sweep_files_parse(path):
    assert load_sweep(path).runs
