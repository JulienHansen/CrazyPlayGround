"""Compare methods on W&B: pick each method's best configs, plot IQM curves with CIs.

W&B can group runs and plot a group's mean, but it cannot pick the best
hyperparameters of each method, nor plot the interquartile mean (IQM) across
seeds with confidence intervals (Agarwal et al., 2021). This script does both,
independently for every (environment, method) pair:

1. Final performance: the IQM over seeds of each run's ``max_smoothed_return``
   (summary). Configs more than ``--threshold`` % below the best are dropped.
2. Sample efficiency: the remaining configs are sorted by the IQM of their
   ``auc_return`` (summary), and the ``--top-k`` best are kept.

Both summary keys are written by scripts/skrl/train.py at the end of training
(see curves.py). A run without them, or logged with another smoothing window
or metric, gets them computed from its history instead. Ranking needs only run
summaries; full histories are downloaded for the selected configs only, and
cached under ``--cache``. Nothing is written to W&B.

A method is the value of ``--method-key`` in the run's resolved settings
(``config.exp``), e.g. ``algo`` or ``env_kwargs.distance_reward_shape``; a
config is one ``config_key`` (all seeds of one setting). The figures plot the
IQM across seeds against env steps (summed over envs), with a bootstrap CI that
resamples seeds; each figure gets a ``.md`` report with the hyperparameter
search space and the selected configs of every method.

Usage (from the repo root, after ``wandb login``):
    python experiments/analyze.py experiments/config/sweep/hovering/reward_shape_01.json \\
        --method-key env_kwargs.distance_reward_shape
    python experiments/analyze.py --project hovering --env Vel-Hovering --tags paper --layout combined
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import math
import os
import re
import subprocess
import sys

import numpy as np

import curves
from parser import DEFAULT_ENTITY, Sweep, SweepError, load_sweep
from wandb_check import COMPLETED_KEY, query_runs

LINESTYLES = ("-", "--", ":", "-.")
_MISSING = object()


@dataclasses.dataclass
class RunScore:
    run: object  # a wandb public-API run
    seed: int
    auc: float
    max_return: float


@dataclasses.dataclass
class Config:
    """All seeds of one config_key, with its scores across seeds."""

    env: str
    method: str
    config_key: str
    group: str
    hps: dict  # hyperparameter -> value, the method key excluded
    runs: list[RunScore]
    score_max: float = math.nan  # IQM over seeds of max_smoothed_return
    score_auc: float = math.nan  # IQM over seeds of auc_return


@dataclasses.dataclass
class Selection:
    """The ranking of one (env, method) pair."""

    env: str
    method: str
    configs: list[Config]  # every config, best max return first
    kept: list[Config]  # within the threshold, best AUC first
    selected: list[Config]  # the top-k of ``kept``
    search_space: dict  # hyperparameter -> values tried


class HistoryCache:
    """Downloads one metric's history per run, cached as ``<root>/<entity>/<project>/<run id>__<metric>.npz``.

    A finished run's history no longer changes, so it is downloaded once unless
    ``refresh``; a run that is still going is never cached.
    """

    def __init__(self, root: str, entity: str, project: str, metric: str, refresh: bool = False):
        self.dir = os.path.join(root, entity, project)
        self.metric = metric
        self.refresh = refresh
        self.downloads = 0

    def history(self, run, metric: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(env steps, values) of ``run``, sorted by step."""
        key = metric or self.metric
        path = os.path.join(self.dir, f"{run.id}__{_slug(key)}.npz")
        if not self.refresh and os.path.exists(path):
            with np.load(path) as data:
                timesteps, values = data["timesteps"], data["values"]
        else:
            rows = [
                row
                for row in run.scan_history(keys=["_step", key])
                if row.get(key) is not None and math.isfinite(float(row[key]))
            ]
            self.downloads += 1
            rows.sort(key=lambda row: row["_step"])
            timesteps = np.array([row["_step"] for row in rows], dtype=float)
            values = np.array([row[key] for row in rows], dtype=float)
            if _is_done(run):
                os.makedirs(self.dir, exist_ok=True)
                np.savez(path, timesteps=timesteps, values=values)
        # train.py logs skrl's timestep, which counts steps per env
        return timesteps * _num_envs(run), values


def analyze(args: argparse.Namespace, api=None) -> list[str]:
    """Fetch, rank, plot and report. Returns the paths written."""
    sweeps = [load_sweep(path) for path in args.sweep]
    entity, project = _target(args, sweeps)
    cache = HistoryCache(args.cache, entity, project, args.metric, args.refresh)
    selections = select(entity, project, args, sweeps, cache, api)

    plot_metrics = list(getattr(args, "plot_metrics", None) or [args.metric])
    curve_of = {}
    for selection in selections:
        for config in selection.selected:
            for metric in plot_metrics:
                curve_of[(config.config_key, metric)] = config_curve(config, cache, args, metric)
            curve_of[config.config_key] = curve_of[(config.config_key, plot_metrics[0])]

    os.makedirs(args.out, exist_ok=True)
    meta = _meta(args, entity, project, sweeps)
    methods = sorted({s.method for s in selections})
    envs = sorted({s.env for s in selections})
    written = []
    if args.layout in ("per-env", "both"):
        for env in envs:
            stem = os.path.join(args.out, f"{_slug(project)}_{_slug(env)}")
            chosen = [s for s in selections if s.env == env]
            written += plot([env], chosen, curve_of, methods, args, stem)
            written.append(write_report(stem + ".md", chosen, meta))
    if args.layout in ("combined", "both"):
        stem = os.path.join(args.out, f"{_slug(project)}_combined")
        written += plot(envs, selections, curve_of, methods, args, stem)
        written.append(write_report(stem + ".md", selections, meta))
    return written


def select(entity: str, project: str, args, sweeps: list[Sweep], cache: HistoryCache, api=None) -> list[Selection]:
    """The ranking of every (env, method) pair, sorted by env then method."""
    runs = fetch_runs(entity, project, args, sweeps, api)
    if not runs:
        raise SweepError(f"no run of {entity}/{project} matches the filters")
    configs = group_configs(runs, args, sweeps, cache)
    selections = [
        rank(group, args.threshold, args.top_k, search_space(group, args.method_key, sweeps))
        for group in configs.values()
    ]
    return sorted(selections, key=lambda s: (s.env, s.method))


def fetch_runs(entity: str, project: str, args, sweeps: list[Sweep], api=None) -> list:
    """The W&B runs that pass every filter; only config and summary are fetched."""
    filters: dict = {}
    if sweeps:
        filters["config.config_key"] = {"$in": sorted({r.config_key for s in sweeps for r in s.runs})}
    if args.tags:
        filters["tags"] = {"$in": args.tags}
    runs = []
    for run in query_runs(f"{entity}/{project}", filters, api):
        exp = run.config.get("exp") or {}
        if "any" not in args.state and not (
            str(run.state).lower() in args.state or ("finished" in args.state and _is_done(run))
        ):
            continue
        if args.env and exp.get("env_name") not in args.env:
            continue
        method = _get(exp, args.method_key)
        if method is _MISSING or (args.methods and str(method) not in args.methods):
            continue
        runs.append(run)
    return runs


def group_configs(runs: list, args, sweeps: list[Sweep], cache: HistoryCache) -> dict:
    """{(env, method): [Config]}, each run scored from its summary (or its history)."""
    sweep_values = {r.config_key: r.sweep_values for s in sweeps for r in s.runs}
    by_key: dict[str, Config] = {}
    for run in runs:
        exp = run.config["exp"]
        key = run.config["config_key"]
        if key not in by_key:
            by_key[key] = Config(
                env=exp["env_name"],
                method=str(_get(exp, args.method_key)),
                config_key=key,
                group=run.group or key,
                hps={},
                runs=[],
            )
        auc, max_return = run_scores(run, args, cache)
        if math.isfinite(auc) and math.isfinite(max_return):
            by_key[key].runs.append(RunScore(run, exp.get("seed"), auc, max_return))

    groups: dict = {}
    for config in by_key.values():
        if config.runs:
            groups.setdefault((config.env, config.method), []).append(config)
    for group in groups.values():
        varying = _varying(group, args.method_key)
        for config in group:
            if config.config_key in sweep_values:
                config.hps = {
                    k: v for k, v in sweep_values[config.config_key].items() if k != args.method_key
                }
            else:
                leaves = dict(_leaves(config.runs[0].run.config["exp"]))
                config.hps = {k: leaves.get(k) for k in varying}
    return groups


def run_scores(run, args, cache: HistoryCache) -> tuple[float, float]:
    """(auc_return, max_smoothed_return) from the summary when it was logged the same way."""
    summary = run.summary_metrics
    smooth_frac = args.smooth / 100
    if (
        summary.get("auc_return") is not None
        and summary.get("max_smoothed_return") is not None
        and summary.get("return_metric", curves.RETURN_METRIC) == args.metric
        and math.isclose(summary.get("smooth_frac", math.nan), smooth_frac)
    ):
        return float(summary["auc_return"]), float(summary["max_smoothed_return"])
    steps, values = cache.history(run)
    if len(steps) == 0:
        return math.nan, math.nan
    return curves.auc(steps, values), curves.max_smoothed(steps, values, smooth_frac)


def rank(configs: list[Config], threshold_pct: float, top_k: int, space: dict | None = None) -> Selection:
    """Two-stage ranking of one (env, method) pair's configs.

    Keeps the configs whose IQM max return is within ``threshold_pct`` % of the
    best one, sorts them by IQM AUC, and selects the first ``top_k``.
    """
    for config in configs:
        config.score_max = float(curves.iqm([r.max_return for r in config.runs]))
        config.score_auc = float(curves.iqm([r.auc for r in config.runs]))
    configs = sorted(configs, key=lambda c: -c.score_max)
    best = configs[0].score_max
    # |best|, so the margin still lies below the best when returns are negative
    floor = best - threshold_pct / 100 * abs(best)
    kept = sorted((c for c in configs if c.score_max >= floor), key=lambda c: -c.score_auc)
    return Selection(
        env=configs[0].env,
        method=configs[0].method,
        configs=configs,
        kept=kept,
        selected=kept[:top_k],
        search_space=space or {},
    )


def search_space(configs: list[Config], method_key: str, sweeps: list[Sweep]) -> dict:
    """Hyperparameter -> values tried by this method: from the sweep files, else the varying settings."""
    keys = {c.config_key for c in configs}
    space: dict[str, list] = {}
    for sweep in sweeps:
        for run in sweep.runs:
            if run.config_key not in keys:
                continue
            for k, v in run.sweep_values.items():
                if k != method_key and v not in space.setdefault(k, []):
                    space[k].append(v)
    if not space:
        for config in configs:
            for k, v in config.hps.items():
                if v not in space.setdefault(k, []):
                    space[k].append(v)
    return {k: _sorted(vs) for k, vs in space.items()}


def config_curve(
    config: Config, cache: HistoryCache, args, metric: str | None = None
) -> tuple[np.ndarray, ...]:
    """(grid, IQM, CI low, CI high) of one config: each seed smoothed, then aligned."""
    steps_list, values_list = [], []
    for score in config.runs:
        steps, values = cache.history(score.run, metric)
        steps_list.append(steps)
        values_list.append(curves.smooth(steps, values, args.smooth / 100))
    grid, matrix = curves.align(steps_list, values_list, args.grid_points)
    low, high = curves.bootstrap_ci(matrix, args.bootstrap, args.ci)
    return grid, curves.iqm(matrix), low, high


def plot(envs: list[str], selections: list[Selection], curve_of: dict, methods: list[str], args, stem: str) -> list[str]:
    """One subplot per env (or env x metric); ``<stem>.png`` and ``<stem>.pdf``. Colours follow ``methods``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(methods)}
    plot_metrics = list(getattr(args, "plot_metrics", None) or [args.metric])
    raw_log_y = getattr(args, "log_y", None)
    log_y = set(plot_metrics if raw_log_y == [] else (raw_log_y or ()))
    multi_metric = len(plot_metrics) > 1
    combined = len(envs) > 1
    if multi_metric:
        ncols = len(plot_metrics)
        nrows = len(envs)
        pairs = [(env, m) for env in envs for m in plot_metrics]
    else:
        ncols = min(len(envs), 3)
        nrows = math.ceil(len(envs) / ncols)
        pairs = [(env, plot_metrics[0]) for env in envs]
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    for ax, (env, metric) in zip(axes.flat, pairs):
        use_log_y = metric in log_y
        for selection in (s for s in selections if s.env == env):
            for i, config in enumerate(selection.selected):
                grid, center, low, high = curve_of.get((config.config_key, metric)) or curve_of[config.config_key]
                color = colors[selection.method]
                label = selection.method
                if (args.top_k > 1 or multi_metric) and config.hps:
                    label += f" ({_hp_text(config.hps)})"
                ax.plot(grid, center, color=color, linestyle=LINESTYLES[i % len(LINESTYLES)], label=label)
                ci_low = np.maximum(low, 1e-12) if use_log_y else low
                ax.fill_between(grid, ci_low, high, color=color, alpha=0.2, linewidth=0)
        if use_log_y:
            ax.set_yscale("log")
        ax.set_title(f"{env} — {metric}" if (multi_metric and combined) else (metric if multi_metric else env))
        ax.set_xlabel("environment steps")
        ax.set_ylabel(metric)
        ax.grid(True, which="both" if use_log_y else "major", alpha=0.3)
        if not combined and not multi_metric:
            ax.legend(fontsize="small")
    if combined:
        # labels with hyperparameters differ between envs, so the shared legend shows methods and ranks
        handles = [Line2D([], [], color=colors[m], label=m) for m in methods]
        if args.top_k > 1:
            handles += [
                Line2D([], [], color="grey", linestyle=LINESTYLES[i % len(LINESTYLES)], label=f"rank {i + 1}")
                for i in range(args.top_k)
            ]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    elif multi_metric:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        ncol = 1 if any(len(lbl) > 35 for lbl in labels) else min(len(handles), 3)
        bottom = 0.04 * math.ceil(len(handles) / ncol) + 0.02
        fig.suptitle(envs[0])
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize="small")
        fig.tight_layout(rect=(0, bottom, 1, 0.95))
    else:
        fig.tight_layout()
    paths = [stem + ".png", stem + ".pdf"]
    for path in paths:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return paths


def write_report(path: str, selections: list[Selection], meta: dict) -> str:
    """The hyperparameter search space and the selected configs of every (env, method)."""
    lines = [f"# {meta['title']}", ""]
    lines += [f"- **{k}:** {v}" for k, v in meta["items"].items()]
    lines += [
        "",
        "Configs are ranked per environment and method: those whose IQM max smoothed return"
        f" is within {meta['threshold']} % of the best are kept, then sorted by IQM AUC"
        " (the mean return over training). IQM drops the lowest and highest 25 % of seeds;"
        " with fewer than 4 seeds it equals the mean.",
    ]
    for env in sorted({s.env for s in selections}):
        lines += ["", f"## {env}"]
        for s in (s for s in selections if s.env == env):
            lines += ["", f"### {s.method}", ""]
            lines.append(
                f"{len(s.configs)} configs, {len(s.kept)} within {meta['threshold']} % of the best max return."
            )
            if s.search_space:
                lines += ["", "| hyperparameter | values tried |", "|---|---|"]
                lines += [f"| `{k}` | {', '.join(_fmt(v) for v in vs)} |" for k, vs in s.search_space.items()]
            lines += [
                "",
                "| rank | hyperparameters | IQM max smoothed return | IQM AUC | seeds | W&B group |",
                "|---|---|---|---|---|---|",
            ]
            for i, c in enumerate(s.selected):
                lines.append(
                    f"| {i + 1} | {_hp_text(c.hps) or '-'} | {c.score_max:.4g} | {c.score_auc:.4g}"
                    f" | {len(c.runs)} | `{c.group}` |"
                )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _target(args, sweeps: list[Sweep]) -> tuple[str, str]:
    projects = {(s.entity, s.project) for s in sweeps}
    if len(projects) > 1:
        raise SweepError(f"the sweep files point to several W&B projects: {sorted(projects)}")
    entity, project = next(iter(projects), (DEFAULT_ENTITY, None))
    entity, project = args.entity or entity, args.project or project
    if project is None:
        raise SweepError("give a sweep file or --project")
    return entity, project


def _meta(args, entity: str, project: str, sweeps: list[Sweep]) -> dict:
    try:
        commit = subprocess.run(
            ["git", "describe", "--always", "--dirty"],  # -dirty: the figures may not match the commit
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    plot_metrics = list(getattr(args, "plot_metrics", None) or [args.metric])
    items = {
        "date": datetime.date.today().isoformat(),
        "commit": f"`{commit}`",
        "sweep files": ", ".join(f"`{s.path}`" for s in sweeps) or "none (all matching runs)",
        "method key": f"`{args.method_key}`",
        "metric": f"`{args.metric}`",
    }
    if plot_metrics != [args.metric]:
        items["plot metrics"] = ", ".join(f"`{m}`" for m in plot_metrics)
    items.update(
        {
            "ranking": f"threshold {_fmt(args.threshold)} %, top-{args.top_k}, smoothing {_fmt(args.smooth)} % of training",
            "curves": f"IQM across seeds, {_fmt(args.ci * 100)} % bootstrap CI ({args.bootstrap} resamples)",
        }
    )
    return {
        "title": f"{entity}/{project}: best configs per method",
        "threshold": _fmt(args.threshold),
        "items": items,
    }


def _varying(configs: list[Config], method_key: str) -> list[str]:
    """Settings that differ between the configs, the method key and seed excluded."""
    values: dict[str, set] = {}
    for config in configs:
        for k, v in _leaves(config.runs[0].run.config["exp"]):
            values.setdefault(k, set()).add(repr(v))
    return [k for k, vs in values.items() if len(vs) > 1 and k not in (method_key, "seed")]


def _leaves(obj: dict, prefix: str = ""):
    for key, value in obj.items():
        if isinstance(value, dict):
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def _get(obj: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return _MISSING
        obj = obj[part]
    return obj


def _is_done(run) -> bool:
    return str(run.state).lower() == "finished" or run.summary_metrics.get(COMPLETED_KEY) is True


def _num_envs(run) -> int:
    return run.config.get("num_envs") or run.config["exp"]["num_envs"]


def _sorted(values: list) -> list:
    try:
        return sorted(values)
    except TypeError:  # mixed types (e.g. None and numbers): keep the order seen
        return values


def _hp_text(hps: dict) -> str:
    return ", ".join(f"{k.split('.')[-1]}={_fmt(v)}" for k, v in hps.items())


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def build_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Pick each method's best configs on W&B and plot IQM curves with CIs."
    )
    cli.add_argument("sweep", nargs="*", help="sweep files; restrict the runs to their configs")
    cli.add_argument("--entity", help=f"W&B entity (default: the sweep's, else {DEFAULT_ENTITY})")
    cli.add_argument("--project", help="W&B project (default: the sweep's)")
    cli.add_argument("--env", nargs="+", help="env_name(s) to keep")
    cli.add_argument("--method-key", default="algo", help="setting that names the method (dotted path in exp)")
    cli.add_argument("--methods", nargs="+", help="method values to keep")
    cli.add_argument("--tags", nargs="+", help="keep runs with any of these W&B tags")
    cli.add_argument(
        "--state", nargs="+", default=["finished"],
        help="W&B run states to keep, or 'any'; 'finished' also takes runs marked completed",
    )
    cli.add_argument("--threshold", type=float, default=5.0, help="%% below the best max return a config may be")
    cli.add_argument("--top-k", type=int, default=2, help="configs plotted per method")
    cli.add_argument("--smooth", type=float, default=curves.SMOOTH_FRAC * 100, help="smoothing window, %% of training")
    cli.add_argument("--metric", default=curves.RETURN_METRIC, help="W&B key of the return")
    cli.add_argument(
        "--plot-metrics", nargs="+",
        help="W&B keys to plot side by side for the selected configs (default: --metric)",
    )
    cli.add_argument(
        "--log-y", nargs="*",
        help="metrics to plot with a log Y-axis (all metrics if given with no values)",
    )
    cli.add_argument("--grid-points", type=int, default=200)
    cli.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    cli.add_argument("--ci", type=float, default=0.95, help="confidence level")
    cli.add_argument("--layout", choices=("per-env", "combined", "both"), default="both")
    cli.add_argument("--out", default="outputs/analysis")
    cli.add_argument("--cache", default="outputs/analysis/cache")
    cli.add_argument("--refresh", action="store_true", help="download cached histories again")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = build_cli().parse_args(argv)

    try:
        written = analyze(args)
    except (SweepError, OSError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:  # login, network, permissions: report rather than guess
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
