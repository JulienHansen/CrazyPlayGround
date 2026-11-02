"""analyze.py must pick each method's best configs the way issue #47 defines.

A wrong pick silently changes which method looks better in a report. These
tests use a fake W&B API (configs, summaries and histories made from a sweep
file), so they run offline and pin the two-stage ranking, the filters, the
summary fallback and the cache.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
import analyze  # noqa: E402
import curves  # noqa: E402
from parser import load_sweep  # noqa: E402

NUM_ENVS = 4
TIMESTEPS = np.arange(0, 101, 10)  # skrl timesteps, per env


def fast(plateau):
    """Reaches its plateau after 10 % of training: high AUC."""
    return lambda t: plateau * np.minimum(1.0, t / 10)


def slow(plateau):
    """Climbs linearly to its plateau: low AUC."""
    return lambda t: plateau * t / 100


# (shape, lr) -> learning curve. tanh: a fast config with a low plateau and a slow
# one with a high plateau. exp: two slow configs within 5 % of each other.
CURVES = {
    ("tanh", 1e-3): fast(0.8),
    ("tanh", 1e-4): slow(1.0),
    ("exp", 1e-3): slow(2.0),
    ("exp", 1e-4): fast(1.95),
}


class FakeRun:
    def __init__(self, run, summary=True, state="finished"):
        exp = run.config
        self.id = run.run_key
        self.name = run.name
        self.group = run.group
        self.state = state
        self.url = f"https://wandb.ai/runs/{run.run_key}"
        self.config = {"run_key": run.run_key, "config_key": run.config_key, "num_envs": NUM_ENVS, "exp": exp}
        curve = CURVES[(exp["env_kwargs"]["distance_reward_shape"], exp["lr"])]
        self._values = curve(TIMESTEPS) + 0.001 * exp["seed"]
        self.summary_metrics = {"completed": True}
        if summary:
            self.summary_metrics.update(curves.summarize(TIMESTEPS * NUM_ENVS, self._values))
            self.summary_metrics["return_metric"] = curves.RETURN_METRIC
        self.scan_calls = 0

    def scan_history(self, keys):
        self.scan_calls += 1
        return [{"_step": int(t), keys[1]: float(v)} for t, v in zip(TIMESTEPS, self._values)]


class FakeApi:
    """Answers like W&B for the config_key $in filter."""

    def __init__(self, runs):
        self._runs = list(runs)
        self.calls = []

    def runs(self, path, filters=None, order="+created_at", per_page=50, include_sweeps=True, lazy=True):
        self.calls.append({"path": path, "filters": filters})
        keys = (filters or {}).get("config.config_key", {}).get("$in")
        return iter([r for r in self._runs if keys is None or r.config["config_key"] in keys])


SWEEP = {
    "version": 1,
    "wandb": {"project": "hovering"},
    "defaults": {"trainer": "skrl", "env_name": "Vel-Hovering", "num_envs": NUM_ENVS, "seed": [1, 2, 3]},
    "experiments": [
        {"name": "shape", "sweep": {"env_kwargs.distance_reward_shape": ["tanh", "exp"], "lr": [1e-3, 1e-4]}}
    ],
}


@pytest.fixture
def sweep_file(tmp_path):
    path = tmp_path / "shape.json"
    path.write_text(json.dumps(SWEEP))
    return str(path)


def make_args(sweep_file, tmp_path, *extra):
    return analyze.build_cli().parse_args(
        [
            sweep_file,
            "--method-key", "env_kwargs.distance_reward_shape",
            "--out", str(tmp_path / "out"),
            "--cache", str(tmp_path / "cache"),
            "--bootstrap", "50",
            *extra,
        ]
    )


def fake_runs(sweep_file, summary=True):
    return [FakeRun(run, summary) for run in load_sweep(sweep_file).runs]


def selections(args, api):
    sweeps = [load_sweep(p) for p in args.sweep]
    cache = analyze.HistoryCache(args.cache, "crazyplayground", "hovering", args.metric)
    return {s.method: s for s in analyze.select("crazyplayground", "hovering", args, sweeps, cache, api)}


def lrs(configs):
    return [c.hps["lr"] for c in configs]


def test_threshold_drops_a_fast_config_with_a_low_plateau(sweep_file, tmp_path):
    result = selections(make_args(sweep_file, tmp_path), FakeApi(fake_runs(sweep_file)))
    # tanh lr=1e-3 has the best AUC but plateaus 20 % below lr=1e-4
    assert lrs(result["tanh"].kept) == [1e-4]
    assert lrs(result["tanh"].selected) == [1e-4]


def test_auc_ranks_the_configs_within_the_threshold(sweep_file, tmp_path):
    result = selections(make_args(sweep_file, tmp_path), FakeApi(fake_runs(sweep_file)))
    # exp: both plateaus within 5 %; lr=1e-4 is faster, so it ranks first
    assert lrs(result["exp"].selected) == [1e-4, 1e-3]
    wide = selections(make_args(sweep_file, tmp_path, "--threshold", "30"), FakeApi(fake_runs(sweep_file)))
    assert lrs(wide["tanh"].selected) == [1e-3, 1e-4]


def test_top_k_limits_the_selection(sweep_file, tmp_path):
    result = selections(make_args(sweep_file, tmp_path, "--top-k", "1"), FakeApi(fake_runs(sweep_file)))
    assert lrs(result["exp"].selected) == [1e-4]
    assert len(result["exp"].kept) == 2


def test_sweep_file_restricts_the_query_and_gives_the_search_space(sweep_file, tmp_path):
    other = dict(SWEEP, experiments=[{"name": "other", "sweep": {"env_kwargs.distance_reward_shape": ["tanh"], "lr": [1e-3]}}])
    other["defaults"] = dict(SWEEP["defaults"], entropy_coef=0.1)
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other))
    api = FakeApi(fake_runs(sweep_file) + fake_runs(str(other_path)))
    result = selections(make_args(sweep_file, tmp_path), api)

    sent = api.calls[0]["filters"]["config.config_key"]["$in"]
    assert sorted(sent) == sorted({r.config_key for r in load_sweep(sweep_file).runs})
    assert sum(len(c.runs) for s in result.values() for c in s.configs) == 12  # 4 configs x 3 seeds
    assert result["tanh"].search_space == {"lr": [1e-4, 1e-3]}
    assert result["tanh"].selected[0].hps == {"lr": 1e-4}


def test_ranking_reads_summaries_without_downloading_histories(sweep_file, tmp_path):
    runs = fake_runs(sweep_file)
    selections(make_args(sweep_file, tmp_path), FakeApi(runs))
    assert all(r.scan_calls == 0 for r in runs)


def test_runs_without_summary_are_scored_from_their_history(sweep_file, tmp_path):
    runs = fake_runs(sweep_file, summary=False)
    result = selections(make_args(sweep_file, tmp_path), FakeApi(runs))
    assert all(r.scan_calls == 1 for r in runs)
    assert lrs(result["tanh"].selected) == [1e-4]
    assert lrs(result["exp"].selected) == [1e-4, 1e-3]


def test_another_smoothing_window_recomputes_from_history(sweep_file, tmp_path):
    runs = fake_runs(sweep_file)
    selections(make_args(sweep_file, tmp_path, "--smooth", "20"), FakeApi(runs))
    assert all(r.scan_calls == 1 for r in runs)


def test_histories_are_cached(sweep_file, tmp_path):
    runs = fake_runs(sweep_file)
    analyze.analyze(make_args(sweep_file, tmp_path), FakeApi(runs))
    downloaded = sum(r.scan_calls for r in runs)
    assert downloaded == 9  # selected configs only: tanh 1 + exp 2, 3 seeds each
    analyze.analyze(make_args(sweep_file, tmp_path), FakeApi(runs))
    assert sum(r.scan_calls for r in runs) == downloaded
    analyze.analyze(make_args(sweep_file, tmp_path, "--refresh"), FakeApi(runs))
    assert sum(r.scan_calls for r in runs) == 2 * downloaded


def test_running_runs_are_not_cached(sweep_file, tmp_path):
    runs = [FakeRun(run, state="running") for run in load_sweep(sweep_file).runs]
    for run in runs:
        run.summary_metrics.pop("completed")
    args = make_args(sweep_file, tmp_path, "--state", "running")
    analyze.analyze(args, FakeApi(runs))
    analyze.analyze(args, FakeApi(runs))
    assert all(r.scan_calls in (0, 2) for r in runs) and any(r.scan_calls == 2 for r in runs)


def test_state_filter(sweep_file, tmp_path):
    runs = fake_runs(sweep_file)
    runs[0].state = "crashed"
    runs[0].summary_metrics.pop("completed")
    result = selections(make_args(sweep_file, tmp_path), FakeApi(runs))
    assert sum(len(c.runs) for s in result.values() for c in s.configs) == 11


def test_both_layouts_write_figures_and_reports(sweep_file, tmp_path):
    written = analyze.analyze(make_args(sweep_file, tmp_path), FakeApi(fake_runs(sweep_file)))
    names = sorted(os.path.basename(p) for p in written)
    assert names == sorted(
        f"hovering_{stem}.{ext}" for stem in ("Vel-Hovering", "combined") for ext in ("png", "pdf", "md")
    )
    assert all(os.path.getsize(p) > 0 for p in written)

    report = (tmp_path / "out" / "hovering_Vel-Hovering.md").read_text()
    assert "### tanh" in report and "### exp" in report
    assert "| `lr` | 0.0001, 0.001 |" in report
    assert "| 1 | lr=0.0001 |" in report
    assert "4 configs" not in report  # 2 configs per method
    assert "2 configs, 1 within 5 % of the best max return." in report


def test_main_reports_a_missing_project(capsys):
    assert analyze.main(["--entity", "x"]) == 1
    assert "--project" in capsys.readouterr().err
