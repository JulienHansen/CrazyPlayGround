"""wandb_check.py must tell which runs of a sweep are still to train.

A wrong "done" silently drops a run from a comparison; a wrong "to train"
spends GPU hours on a duplicate. These tests use a fake W&B API, so they run
offline and pin the status rules and the query sent to W&B.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from parser import expand  # noqa: E402
from wandb_check import check_sweep  # noqa: E402


class FakeRun:
    def __init__(self, run_key, state, completed=False):
        self.config = {"run_key": run_key}
        self.state = state
        self.summary_metrics = {"completed": True} if completed else {}
        self.url = f"https://wandb.ai/runs/{run_key}-{state}"


class FakeApi:
    """Answers like W&B: only the runs whose run_key is in the $in filter."""

    def __init__(self, runs=(), project_exists=True):
        self._runs = list(runs)
        self._project_exists = project_exists
        self.calls = []

    def runs(
        self,
        path,
        filters=None,
        order="+created_at",
        per_page=50,
        include_sweeps=True,
        lazy=True,
    ):
        self.calls.append({"path": path, "filters": filters, "lazy": lazy})
        return self._iterate(filters["config.run_key"]["$in"])

    def _iterate(self, keys):
        if not self._project_exists:
            raise ValueError(f"Could not find project {self.calls[-1]['path']}")
        yield from (r for r in self._runs if r.config["run_key"] in keys)


class OldApi(FakeApi):
    """A wandb version without the lazy argument."""

    def runs(
        self, path, filters=None, order="+created_at", per_page=50, include_sweeps=True
    ):
        return super().runs(path, filters, order, per_page, include_sweeps)


@pytest.fixture
def sweep():
    return expand(
        {
            "version": 1,
            "wandb": {"project": "hovering"},
            "defaults": {"trainer": "skrl", "env_name": "Vel-Hovering", "seed": [1]},
            "experiments": [{"name": "scan", "sweep": {"lr": [1, 2, 3, 4, 5, 6]}}],
        }
    )


def statuses(result):
    return [s.status for s in result.runs]


def test_each_run_gets_the_best_status_of_its_wandb_runs(sweep):
    k = [run.run_key for run in sweep.runs]
    api = FakeApi(
        [
            FakeRun(k[0], "finished"),
            FakeRun(k[1], "crashed", completed=True),  # died after training
            FakeRun(k[2], "running"),
            FakeRun(k[3], "crashed"),
            FakeRun(k[4], "failed"),
            FakeRun(k[4], "finished"),  # retried
        ]
    )
    result = check_sweep(sweep, api)
    assert statuses(result) == ["done", "done", "running", "failed", "done", "missing"]
    assert result.to_train() == [sweep.runs[3], sweep.runs[5]]
    assert result.runs[4].urls == [
        f"https://wandb.ai/runs/{k[4]}-failed",
        f"https://wandb.ai/runs/{k[4]}-finished",
    ]


def test_running_beats_failed(sweep):
    key = sweep.runs[0].run_key
    result = check_sweep(
        sweep, FakeApi([FakeRun(key, "crashed"), FakeRun(key, "running")])
    )
    assert result.runs[0].status == "running"


def test_query_asks_for_the_sweep_keys_in_the_sweep_project(sweep):
    api = FakeApi()
    check_sweep(sweep, api)
    assert api.calls == [
        {
            "path": "crazyplayground/hovering",
            "filters": {"config.run_key": {"$in": [run.run_key for run in sweep.runs]}},
            "lazy": False,
        }
    ]


def test_wandb_without_lazy_argument_still_works(sweep):
    result = check_sweep(sweep, OldApi([FakeRun(sweep.runs[0].run_key, "finished")]))
    assert statuses(result)[0] == "done"


def test_missing_project_means_every_run_is_missing(sweep):
    result = check_sweep(sweep, FakeApi(project_exists=False))
    assert not result.project_found
    assert set(statuses(result)) == {"missing"}


def test_other_wandb_errors_are_not_hidden(sweep):
    class BrokenApi(FakeApi):
        def _iterate(self, keys):
            raise ValueError("permission denied")
            yield

    with pytest.raises(ValueError, match="permission denied"):
        check_sweep(sweep, BrokenApi())
