"""Check on W&B which runs of a sweep file are already done.

Every run the sweep file expands to (see parser.py) is looked up by its
``run_key`` in the sweep's W&B project and gets one status:

    done      a W&B run with this key finished, or marked itself completed
    running   a W&B run with this key is running now, on any machine
    failed    W&B runs with this key exist, but all failed, crashed or were killed
    missing   no W&B run has this key

``failed`` and ``missing`` runs are the ones still to train. The completed
mark is ``summary["completed"] = True``, set by the training script after its
last update, so a run that crashes after training (during simulator shutdown,
say) is not trained again.

Usage (from the repo root, after ``wandb login``):
    python wandb_check.py config/sweep/hovering/test_01.json          # per-experiment counts
    python wandb_check.py config/sweep/hovering/test_01.json --runs   # one line per run
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import sys

from parser import Run, Sweep, SweepError, load_sweep

COMPLETED_KEY = "completed"
DONE, RUNNING, FAILED, MISSING = "done", "running", "failed", "missing"
TO_TRAIN = (FAILED, MISSING)


@dataclasses.dataclass
class RunStatus:
    run: Run
    status: str
    urls: list[str]  # W&B runs that carry this run_key, oldest first


@dataclasses.dataclass
class SweepStatus:
    sweep: Sweep
    runs: list[RunStatus]
    project_found: bool

    def to_train(self) -> list[Run]:
        return [s.run for s in self.runs if s.status in TO_TRAIN]


def check_sweep(sweep: Sweep, api=None) -> SweepStatus:
    """Look up every run of ``sweep`` on W&B."""
    records = fetch_records(
        sweep.entity, sweep.project, [run.run_key for run in sweep.runs], api
    )
    return SweepStatus(
        sweep=sweep,
        runs=classify(sweep.runs, records or []),
        project_found=records is not None,
    )


def fetch_records(
    entity: str, project: str, run_keys: list[str], api=None
) -> list[dict] | None:
    """The W&B runs whose ``config.run_key`` is in ``run_keys``.

    Returns None when the project does not exist on W&B yet.
    """
    if not run_keys:
        return []
    if api is None:
        import wandb

        api = wandb.Api()
    kwargs = {}
    # recent wandb fetches config and summary one run at a time unless lazy=False;
    # older versions have no lazy argument
    if "lazy" in inspect.signature(api.runs).parameters:
        kwargs["lazy"] = False
    try:
        return [
            {
                "run_key": run.config.get("run_key"),
                "state": str(run.state).lower(),
                "completed": run.summary_metrics.get(COMPLETED_KEY) is True,
                "url": run.url,
            }
            for run in api.runs(
                f"{entity}/{project}",
                filters={"config.run_key": {"$in": list(run_keys)}},
                per_page=200,
                include_sweeps=False,
                **kwargs,
            )
        ]
    except ValueError as e:
        if "Could not find project" in str(e):
            return None
        raise


def classify(runs: list[Run], records: list[dict]) -> list[RunStatus]:
    """Give each run the best status among the W&B runs that carry its key."""
    by_key: dict[str, list[dict]] = {}
    for record in records:
        by_key.setdefault(record["run_key"], []).append(record)
    statuses = []
    for run in runs:
        found = by_key.get(run.run_key, [])
        seen = {_status(record) for record in found}
        status = next((s for s in (DONE, RUNNING, FAILED) if s in seen), MISSING)
        statuses.append(
            RunStatus(run=run, status=status, urls=[record["url"] for record in found])
        )
    return statuses


def _status(record: dict) -> str:
    if record["completed"] or record["state"] == "finished":
        return DONE
    if record["state"] in ("running", "pending"):
        return RUNNING
    return FAILED


def _counts(statuses: list[RunStatus]) -> str:
    count = {
        s: sum(x.status == s for x in statuses)
        for s in (DONE, RUNNING, FAILED, MISSING)
    }
    parts = [f"{count[DONE]} done"]
    if count[RUNNING]:
        parts.append(f"{count[RUNNING]} running")
    to_train = f"{count[FAILED] + count[MISSING]} to train"
    if count[FAILED]:
        to_train += f" ({count[FAILED]} failed before)"
    parts.append(to_train)
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description="Check on W&B which runs of a sweep file are already done."
    )
    cli.add_argument("sweep_file")
    cli.add_argument("--runs", action="store_true", help="one line per run")
    args = cli.parse_args(argv)

    try:
        sweep = load_sweep(args.sweep_file)
    except (SweepError, OSError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    try:
        result = check_sweep(sweep)
    except Exception as e:  # login, network, permissions: report rather than guess
        print(f"[ERROR] W&B query failed: {e}", file=sys.stderr)
        return 1

    print(f"{sweep.path} -> {sweep.entity}/{sweep.project} (version {sweep.version})")
    if not result.project_found:
        print(
            f"  [WARN] W&B has no project {sweep.entity}/{sweep.project}: every run counts"
            " as missing. Fine for a new project, otherwise check 'wandb' in the sweep file."
        )
    if args.runs:
        for s in result.runs:
            url = f"  {s.urls[-1]}" if s.urls else ""
            print(f"  {s.status:<8} {s.run.run_key}  {s.run.name}{url}")
    else:
        width = max(len(name) for name in sweep.experiments)
        for name in sweep.experiments:
            owned = [s for s in result.runs if s.run.experiment == name]
            print(f"  {name:<{width}}  {_counts(owned)}")
    print(f"  total: {_counts(result.runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
