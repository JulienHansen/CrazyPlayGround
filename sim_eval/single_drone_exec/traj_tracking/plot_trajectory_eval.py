"""Figures and tables of tracking rollouts recorded by play.py --eval_csv: the path, the reference, the error.

    python scripts/skrl/play.py --task Vel-Tracking --headless --checkpoint <agent.pt> --eval_csv a.csv \
        [env.<knob>=<value> ...]
    python sim_eval/single_drone_exec/traj_tracking/plot_trajectory_eval.py a.parquet [b.parquet ...] \
        [--labels A B ...] [--out DIR] [--name NAME] [--env I] [--max-traces 64] [--overlay]

Files are grouped by label (--labels, one per file; default: the file name). Files that share a
label are pooled, e.g. the 5 seeds of one config. Writes, in --out (default: the first file's folder):
    top_<name>_<label>.png   top view (x, y): reference dashed, IQM path of the episodes still flying
                             with the 10-90 % band of their sideways offset from it, reference
                             start ●, each episode's first position ○, crashes as ×
                             (episodes with different references: up to --max-traces of them faint);
    time_<name>_<label>.png  x/y/z(t), tracking error (log scale) and speed
                             (actual vs velocity setpoint): IQM over the episodes still flying with
                             the 10-90 % band (10th to 90th percentile of the episodes);
                             one pair per label (--overlay: top_<name>.png and time_<name>.png with
                             every label); each label keeps its colour across figures;
    summary_<name>.png   per-episode RMSE: IQM and 10-90 % range per label, over the episodes;
    summary_<name>.csv / .md   one row per label (also the along- and cross-track errors).

The IQM path is, per step, the interquartile mean of the episodes still flying. It needs every
episode to fly the same reference (env.traj_*_range set to one value, env.traj_yaw_range=[0.0,0.0],
env.traj_phase_range=[0.0,0.0]); otherwise the top view shows each episode against its own
reference and the time rows show the error components instead of x/y/z.
--env I draws episode I of each label instead of the IQM.

Input (.parquet that play.py --eval_csv writes next to its CSV, one row per (episode, step) as the flight recordings;
positions relative to the env origin):
    episode, step                            episode id, policy step 1..length
    pos_*, target_*, vel_*                   drone position, scored reference [m], world velocity [m/s]
    vx, vy, vz                               velocity setpoint after latency [m/s] (velocity mode only)
    action_*                                 the policy's commanded action (for the effort)
    footer: dt, env_cfg (JSON: reset_thres, ...), died (JSON: per episode, crashed or not)
numpy, pyarrow and matplotlib: no Isaac Lab needed.
"""

import argparse
import csv
import json
import os
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BAND = (10, 90)        # percentiles of the shaded band
MAX_TRACES = 64        # faint episode lines per label in the top view, without a shared reference
SHARED_TOL = 1e-4      # m: references closer than this count as the same

# Colours: categorical slots of the dataviz reference palette in fixed order (the first three are
# validated for all pairs); the reference is ink, not a series.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a85", "#e6e5e0"
REF_STYLE = dict(color=INK_2, lw=1.4, ls=(0, (4, 3)))

PER_EPISODE = ("pos", "ref", "vel_w", "setpoint", "length", "died",
               "rmse", "mean_err", "max_err", "effort")


# ── loading and grouping ──────────────────────────────────────────────────────────────────────


def load_run(path: str, label: str | None = None) -> dict:
    """Read one play.py --eval_csv .parquet file into (N, T, ...) arrays; the label defaults to the file name.

    The per-episode metrics are computed from the recorded steps, as the env computes them:
    ``rmse``, ``mean_err``, ``max_err`` of the distance to the scored reference, and ``effort``, the
    mean squared norm of the commanded action.
    """
    table = pq.read_table(path)
    footer = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    cfg = json.loads(footer.get("env_cfg", "{}"))
    column = {name: table.column(name).to_numpy() for name in table.column_names}
    row, slot = column["episode"].astype(int), column["step"].astype(int) - 1
    died = np.array(json.loads(footer["died"]), dtype=bool)
    n_episodes, n_steps = len(died), int(slot.max()) + 1 if slot.size else 0

    def grid(names):
        out = np.zeros((n_episodes, n_steps, len(names)), dtype=np.float32)
        out[row, slot] = np.stack([column[n] for n in names], axis=-1)
        return out

    length = np.bincount(row, minlength=n_episodes)  # one row per step of the episode
    run = {
        "pos": grid(["pos_x", "pos_y", "pos_z"]), "ref": grid(["target_x", "target_y", "target_z"]),
        "vel_w": grid(["vel_x", "vel_y", "vel_z"]), "length": length, "died": died,
        "dt": float(footer["dt"]), "reset_thres": float(cfg.get("reset_thres", np.nan)),
        "ref_lead": float(cfg.get("traj_step_size", 0.0)),
        "label": label or os.path.splitext(os.path.basename(path))[0], "files": [path],
    }
    if "vx" in column:  # velocity mode: the setpoint the controller got [m/s]
        run["setpoint"] = grid(["vx", "vy", "vz"])
    steps = np.maximum(length, 1)
    error = np.linalg.norm(np.stack([column[f"pos_{a}"] - column[f"target_{a}"] for a in "xyz"], axis=-1), axis=-1)
    run["rmse"] = np.sqrt(np.bincount(row, error ** 2, n_episodes) / steps)
    run["mean_err"] = np.bincount(row, error, n_episodes) / steps
    run["max_err"] = np.zeros(n_episodes)
    np.maximum.at(run["max_err"], row, error)
    action = [n for n in column if n.startswith("action_") and not n.startswith("action_sent_")]
    effort = np.sum([column[n].astype(float) ** 2 for n in action], axis=0) if action else np.zeros(len(row))
    run["effort"] = np.bincount(row, effort, n_episodes) / steps
    return run


def group_by_label(runs: list[dict]) -> dict[str, dict]:
    """Pool the episodes of the runs that share a label, keeping the first-seen label order.

    Raises:
        ValueError: if pooled runs differ in policy step or number of steps.
    """
    groups: dict[str, dict] = {}
    for run in runs:
        group = groups.get(run["label"])
        if group is None:
            groups[run["label"]] = dict(run)
            continue
        if group["dt"] != run["dt"] or group["pos"].shape[1] != run["pos"].shape[1]:
            raise ValueError(f"label {run['label']!r}: files differ in dt or episode length, can't pool them")
        for key in PER_EPISODE:
            if key in group and key in run:
                group[key] = np.concatenate([group[key], run[key]])
            else:
                group.pop(key, None)  # a velocity setpoint only if every pooled file has one
        group["files"] = group["files"] + run["files"]
    return groups


def spawn_point(ref: np.ndarray, lead: float, fit: int = 8) -> np.ndarray:
    """(3,) reference start point, where the drone spawns (before its random offset).

    The scored reference at step t (slot t - 1) is the trajectory ``lead`` policy steps ahead of t,
    and the spawn point is the trajectory at step 0, so it lies ``lead + 1`` steps before slot 0;
    it is extrapolated back with a quadratic fit of the first ``fit`` slots.

    Args:
        ref: (T, 3) scored reference of one episode [m].
        lead: look-ahead of the scored reference, policy steps (env ``traj_step_size``).
    """
    n = min(fit, len(ref))
    if n < 3:
        return ref[0]
    coef = np.polyfit(np.arange(n, dtype=float), ref[:n], 2)  # (3 powers, 3 axes)
    x = -(lead + 1.0)
    return np.array([x * x, x, 1.0]) @ coef


def _longest(g: dict) -> int:
    return int(np.argmax(g["length"]))


def shared_reference(groups: dict[str, dict], tol: float = SHARED_TOL) -> tuple[bool, bool]:
    """(within, across): every episode of a label flies its longest episode's reference on the
    steps it lasts, and all labels fly the same one on the steps they share."""
    within = True
    for g in groups.values():
        ref = g["ref"][_longest(g)]
        dev = np.abs(g["ref"] - ref[None]).max(axis=-1)  # (N, T)
        within &= bool((dev[valid_mask(g["length"], g["ref"].shape[1])] <= tol).all())
    first = next(iter(groups.values()))
    ref0, n0 = first["ref"][_longest(first)], first["length"].max()
    across = within
    for g in groups.values():
        n = min(n0, g["length"].max(), ref0.shape[0], g["ref"].shape[1])
        across &= bool(np.abs(g["ref"][_longest(g), :n] - ref0[:n]).max() <= tol) if n else True
    return within, across


# ── per-step statistics ───────────────────────────────────────────────────────────────────────


def valid_mask(length: np.ndarray, steps: int) -> np.ndarray:
    """(N, T) bool: step t (slot t - 1 of the arrays) belongs to episode n."""
    return np.arange(steps)[None, :] < np.asarray(length)[:, None]


def masked(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``x`` (N, T, ...) as float with NaN on the steps outside each episode."""
    x = np.array(x, dtype=float)
    x[~valid] = np.nan
    return x


def _nan_stat(fn, x, **kwargs):
    with warnings.catch_warnings():  # all-NaN columns (every episode over) are expected, give NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(x, **kwargs)


def band(x: np.ndarray, valid: np.ndarray, q=BAND) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(10th percentile, IQM, 90th percentile) over the episodes still flying, per step: each (T, ...)."""
    lo, hi = _nan_stat(np.nanpercentile, masked(x, valid), q=[q[0], q[1]], axis=0)
    return lo, iqm_alive(x, valid), hi


def iqm_alive(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interquartile mean over the episodes still flying, per step: (T, ...), NaN once all are over.

    Per step, the mean of the central 50 % of the n episodes still flying (sorted, n // 4 dropped
    at each end), as ``iqm`` does for one set of values."""
    s = np.sort(masked(x, valid), axis=0)  # NaN (episode over) sorts last
    n = np.sum(np.isfinite(s), axis=0)
    q = n // 4
    rank = np.arange(s.shape[0]).reshape((-1,) + (1,) * (s.ndim - 1))
    keep = (rank >= q) & (rank < n - q)
    total = np.where(keep, s, 0.0).sum(axis=0)
    count = keep.sum(axis=0)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def mean_alive(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean over the episodes still flying, per step: (T, ...), NaN once all are over."""
    return _nan_stat(np.nanmean, masked(x, valid), axis=0)


def lateral_band(pos_xy: np.ndarray, valid: np.ndarray, tangent_xy: np.ndarray,
                 q=BAND) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IQM path in the plane and the edges of its 10-90 % band across the direction of travel.

    Per step, each episode's offset from the IQM path is projected on the left normal of
    ``tangent_xy``; the band edges are the IQM path moved by the 10th and 90th percentiles of that
    offset. Along the direction of travel a top view can't show a spread, so the band is sideways only.

    Args:
        pos_xy: (N, T, 2) drone positions in the plane [m].
        valid: (N, T) bool, steps that belong to each episode.
        tangent_xy: (T, 2) direction of travel (any length); NaN band where it is 0.

    Returns:
        (lo, mid, hi): (T, 2) each [m]; mid is the IQM path.
    """
    mid = iqm_alive(pos_xy, valid)
    norm = np.linalg.norm(tangent_xy, axis=-1, keepdims=True)
    unit = np.divide(tangent_xy, norm, out=np.full(tangent_xy.shape, np.nan), where=norm > 1e-9)
    normal = np.stack([-unit[:, 1], unit[:, 0]], axis=-1)
    offset = np.sum((pos_xy - mid) * normal, axis=-1)  # (N, T), > 0 = left of the IQM path
    lo, _, hi = band(offset, valid, q)
    return mid + lo[:, None] * normal, mid, mid + hi[:, None] * normal


def track_errors(pos: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the error e = pos - ref into along-track and cross-track parts.

    The track direction is the reference's direction of travel (finite difference over the steps).
    Along-track is signed: < 0 means the drone is behind the reference (lag), > 0 ahead. Where the
    reference doesn't move, along-track is 0 and the whole error counts as cross-track.

    Args:
        pos, ref: (N, T, 3) drone and reference positions [m].

    Returns:
        (along, cross): (N, T) each [m]; cross >= 0.
    """
    e = pos - ref
    tangent = np.gradient(ref, axis=1) if ref.shape[1] > 1 else np.zeros_like(ref)
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    unit = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 1e-9)
    along = np.sum(e * unit, axis=-1)
    cross = np.linalg.norm(e - along[..., None] * unit, axis=-1)
    return along, cross


def episode_mean(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(N,) mean of ``x`` (N, T) over each episode's own steps."""
    return _nan_stat(np.nanmean, masked(x, valid), axis=1)


def iqm(values) -> float:
    """Interquartile mean: the mean of the central 50 % of the finite values."""
    x = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if x.size == 0:
        return float("nan")
    q = x.size // 4
    return float(x[q:x.size - q].mean())


def spread(values) -> tuple[float, float, float]:
    """(10th percentile, IQM, 90th percentile) of the finite values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(x, BAND)
    return float(lo), iqm(x), float(hi)


def episode_values(g: dict) -> dict[str, np.ndarray]:
    """(N,) per-episode RMSE, mean along-track and mean cross-track error [m]."""
    valid = valid_mask(g["length"], g["pos"].shape[1])
    along, cross = track_errors(g["pos"], g["ref"])
    return {"rmse": np.asarray(g["rmse"], dtype=float), "along": episode_mean(along, valid),
            "cross": episode_mean(cross, valid)}


def summarize(groups: dict[str, dict]) -> list[dict]:
    """One row per label: crash rate, and the IQM (with 10th/90th percentiles for the RMSE) of the
    per-episode metrics."""
    rows = []
    for label, g in groups.items():
        values = episode_values(g)
        p10, rmse_iqm, p90 = spread(values["rmse"])
        rows.append({
            "label": label,
            "files": len(g["files"]),
            "episodes": len(g["length"]),
            "crash_rate": float(np.mean(g["died"])),
            "rmse_iqm": rmse_iqm,
            "rmse_p10": p10,
            "rmse_p90": p90,
            "mean_err_iqm": iqm(g["mean_err"]),
            "max_err_iqm": iqm(g["max_err"]),
            "along_iqm": iqm(values["along"]),
            "cross_iqm": iqm(values["cross"]),
            "effort_iqm": iqm(g["effort"]),
        })
    return rows


# ── figures ───────────────────────────────────────────────────────────────────────────────────


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_3, "ytick.color": INK_3, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "font.size": 9, "legend.frameon": False, "legend.labelcolor": INK,
    })


def _slug(label: str) -> str:
    """A label as a file-name part: letters, digits, '.', '-' and '_' kept, anything else -> '-'."""
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in label).strip("-") or "label"


def _colors(groups):
    if len(groups) > len(SERIES):
        raise ValueError(f"{len(groups)} labels, at most {len(SERIES)} fit one figure: split the files")
    return dict(zip(groups, SERIES))


def _episode(g: dict, index: int) -> dict:
    if not 0 <= index < len(g["length"]):
        raise ValueError(f"--env {index}: label {g['label']!r} has {len(g['length'])} episodes")
    return {k: (g[k][index:index + 1] if k in PER_EPISODE else g[k]) for k in g}


def trim_to_episodes(groups: dict[str, dict]) -> dict[str, dict]:
    """Cut the per-step arrays at the longest episode of any label.

    A timed-out episode lasts max_episode_length - 1 steps, so the last recorded slot belongs to no
    episode; drawing it would show the zero-filled buffer (a reference jumping to the origin)."""
    T = int(max(g["length"].max() for g in groups.values()))
    steps = ("pos", "ref", "vel_w", "setpoint")
    return {label: {**g, **{k: g[k][:, :T] for k in steps if k in g}} for label, g in groups.items()}


def _ref_line(g: dict) -> np.ndarray:
    """(T, 3) scored reference of the label's longest episode, NaN past its end."""
    n = int(np.argmax(g["length"]))
    return masked(g["ref"][n], np.arange(g["ref"].shape[1]) < g["length"][n])


def _prepare(groups, env, colors):
    """Shared set-up of the top-view and time-series figures."""
    _style()
    if env is not None:
        groups = {label: _episode(g, env) for label, g in groups.items()}
    groups = trim_to_episodes(groups)
    colors = {label: (colors or _colors(groups))[label] for label in groups}
    within, across = shared_reference(groups)
    first = next(iter(groups.values()))
    # one reference per label when every episode of a label shares it (once if all labels do)
    ref_groups = ([first] if across else list(groups.values())) if within else []
    return groups, colors, within, ref_groups, first


def _title(fig, groups, env, within, width: int = 200):
    who = next(iter(groups)) if len(groups) == 1 else "trajectory tracking"
    n = max(len(g["length"]) for g in groups.values())
    if env is not None:
        what = f"episode {env}"
    elif within:
        what = f"IQM of the {n} episodes still flying, band {BAND[0]}–{BAND[1]} %"
    else:
        what = "each episode against its own reference (grey); rows: error components"
    fig.suptitle("\n".join(textwrap.wrap(f"{who}: {what}", width)), color=INK, fontsize=10.5)


def _save(fig, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _legend(fig, groups, colors, ref_groups, setpoint=False, start=False):
    handles = [Line2D([], [], **REF_STYLE, label="reference")] if ref_groups else []
    if start:
        if ref_groups:
            handles.append(Line2D([], [], color=INK, lw=0, marker="o", ms=7, label="spawn point"))
        handles.append(Line2D([], [], color=INK_2, lw=0, marker="o", ms=6, mfc="none", mew=1.3,
                              label="drone, first step"))
    handles += [Line2D([], [], color=c, lw=2.2, label=f"{label} ({len(groups[label]['length'])} ep.), IQM")
                for label, c in colors.items()]
    handles.append(Line2D([], [], color=INK_2, lw=0, marker="x", ms=7, mew=2, label="crash"))
    if setpoint and any("setpoint" in g for g in groups.values()):
        handles.append(Line2D([], [], color=INK_2, lw=1.2, ls=(0, (1, 1.5)), label="velocity setpoint, IQM"))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4))


def plot_topview(groups: dict[str, dict], out_path: str, env: int | None = None,
                 max_traces: int = MAX_TRACES, colors: dict | None = None) -> str:
    """Write the top view (x, y): reference dashed, IQM path with its sideways 10–90 % band, crashes as ×.

    When the episodes fly different references (no IQM path), up to ``max_traces`` of them are drawn
    faint instead, each against its own reference.

    ``colors`` maps labels to colours, so a label keeps its colour in its own figure; default: the
    palette in label order."""
    groups, colors, within, ref_groups, first = _prepare(groups, env, colors)
    T = first["pos"].shape[1]
    fig, ax = plt.subplots(figsize=(7.6, 8.0), layout="constrained")
    for g in ref_groups:  # on top of the paths: they hug it within centimetres
        ref = g["ref"][_longest(g), :g["length"].max()]  # the scored reference of the longest episode
        ax.plot(ref[:, 0], ref[:, 1], **{**REF_STYLE, "lw": 1.1}, zorder=6)
        start = spawn_point(ref, g["ref_lead"])
        ax.plot(start[0], start[1], "o", color=INK, ms=7, zorder=8)
        k = max(1, min(len(ref) // 10, len(ref) - 2))
        ax.annotate("", xy=tuple(ref[k + 1, :2]), xytext=tuple(ref[k - 1, :2]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=14))
    traced = {label: g for label, g in groups.items() if not within or len(g["length"]) == 1}
    n_traces = sum(min(max_traces, len(g["length"])) for g in traced.values())
    faint = float(np.clip(4.0 / max(n_traces, 1), 0.06, 0.22))  # many episodes (laps) stack up
    for label, g in traced.items():
        single = len(g["length"]) == 1
        for n in range(min(max_traces, len(g["length"]))):
            L = g["length"][n]
            if not within:
                ax.plot(g["ref"][n, :L, 0], g["ref"][n, :L, 1], color=INK_3, lw=0.5, alpha=0.5)
            ax.plot(g["pos"][n, :L, 0], g["pos"][n, :L, 1], color=colors[label],
                    lw=1.8 if single else 0.6, alpha=0.9 if single else faint, zorder=4)
    for label, g in groups.items():
        if label in traced:
            continue
        tangent = np.gradient(g["ref"][_longest(g), :T, :2], axis=0)  # the shared reference's direction
        lo, path, hi = lateral_band(g["pos"][..., :2], valid_mask(g["length"], T), tangent)
        ok = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
        ax.fill(np.r_[lo[ok, 0], hi[ok, 0][::-1]], np.r_[lo[ok, 1], hi[ok, 1][::-1]],
                color=colors[label], alpha=0.25, lw=0, zorder=3)
    for label, g in groups.items():  # IQM paths after every label's band, so no band hides one
        if label not in traced:
            path = iqm_alive(g["pos"][..., :2], valid_mask(g["length"], T))
            ax.plot(path[:, 0], path[:, 1], color=colors[label], lw=1.6, zorder=5)
    for label, g in groups.items():  # each episode's first recorded position (spawn is randomised)
        ax.plot(g["pos"][:, 0, 0], g["pos"][:, 0, 1], "o", color=colors[label], ms=5, mfc="none",
                mew=1.1, alpha=0.8, zorder=6)
    for label, g in groups.items():
        crashed = np.flatnonzero(g["died"])
        if crashed.size:
            ends = g["pos"][crashed, g["length"][crashed] - 1]
            ax.plot(ends[:, 0], ends[:, 1], "x", color=colors[label], ms=11, mew=2.6, zorder=7,
                    path_effects=[patheffects.withStroke(linewidth=5, foreground=SURFACE)])
    ax.set(xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    _legend(fig, groups, colors, ref_groups, start=True)
    _title(fig, groups, env, within, width=90)
    return _save(fig, out_path)


def plot_timeseries(groups: dict[str, dict], out_path: str, env: int | None = None,
                    colors: dict | None = None) -> str:
    """Write the time rows: x/y/z(t) (or the error components without a shared reference), the
    tracking error and the speed; IQM with a 10–90 % band."""
    groups, colors, within, ref_groups, first = _prepare(groups, env, colors)
    T, dt = first["pos"].shape[1], first["dt"]
    t = (np.arange(T) + 1) * dt
    fig, rows = plt.subplots(5, 1, figsize=(11, 11), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.3, 1.3]})

    for axis, ax in enumerate(rows[:3]):
        name = "xyz"[axis]
        for g in ref_groups:
            ax.plot(t, _ref_line(g)[:, axis], **REF_STYLE)
        for label, g in groups.items():
            valid = valid_mask(g["length"], T)
            series = g["pos"][..., axis] if within else (g["pos"] - g["ref"])[..., axis]
            lo, mid, hi = band(series, valid)
            ax.fill_between(t, lo, hi, color=colors[label], alpha=0.18, lw=0)
            ax.plot(t, mid, color=colors[label], lw=1.8)
        ax.set_ylabel(f"{name} [m]" if within else f"{name} − {name}_ref [m]")

    ax_err, ax_speed = rows[3], rows[4]
    for label, g in groups.items():
        valid = valid_mask(g["length"], T)
        lo, mid, hi = band(np.linalg.norm(g["pos"] - g["ref"], axis=-1), valid)
        ax_err.fill_between(t, lo, hi, color=colors[label], alpha=0.18, lw=0)
        ax_err.plot(t, mid, color=colors[label], lw=1.8)

        lo, mid, hi = band(np.linalg.norm(g["vel_w"], axis=-1), valid)
        ax_speed.fill_between(t, lo, hi, color=colors[label], alpha=0.18, lw=0)
        ax_speed.plot(t, mid, color=colors[label], lw=1.8)
        if "setpoint" in g:
            setpoint = np.linalg.norm(g["setpoint"], axis=-1)
            ax_speed.plot(t, iqm_alive(setpoint, valid), color=colors[label], lw=1.0, ls=(0, (1, 1.5)))
    reset = first["reset_thres"]
    ax_err.axhline(reset, color=INK_3, lw=1, ls=":")
    ax_err.annotate(f"reset at {reset:g} m", (t[0], reset), xytext=(2, 3), textcoords="offset points",
                    color=INK_2, fontsize=8)
    ax_err.set_ylabel(f"‖e‖ [m] (log)\nIQM, {BAND[0]}–{BAND[1]} %")
    ax_err.set_yscale("log")  # zeros (e.g. a band edge at a reset) are masked, not drawn
    for g in ref_groups:
        if T > 1:
            ax_speed.plot(t, np.linalg.norm(np.gradient(_ref_line(g), dt, axis=0), axis=-1), **REF_STYLE)
    ax_speed.set(ylabel="speed [m/s]", xlabel="time [s]")
    top = max((np.nanmax(line.get_ydata()) for line in ax_speed.get_lines() if np.isfinite(line.get_ydata()).any()),
              default=1.0)
    ax_speed.set_ylim(0, 1.15 * top if top > 0 else 1.0)
    _legend(fig, groups, colors, ref_groups, setpoint=True)
    _title(fig, groups, env, within)
    return _save(fig, out_path)


def _tick(label: str, g: dict) -> str:
    return "\n".join(textwrap.wrap(label, 18)) + f"\nn={len(g['length'])}\ncrash {100 * np.mean(g['died']):.1f} %"


def plot_summary(groups: dict[str, dict], out_path: str, seed: int = 0, max_points: int = 400) -> str:
    """Write the per-episode RMSE summary figure and return its path.

    Per label: the IQM (black bar) and the 10–90 % range (grey line) of the per-episode RMSE, over
    up to ``max_points`` of the episodes as points (× = crashed)."""
    _style()
    colors = _colors(groups)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(groups) + 1.5), 4.2), layout="constrained")
    for x, (label, g) in enumerate(groups.items()):
        values = g["rmse"]
        shown = rng.permutation(len(values))[:max_points]
        jitter = x + rng.uniform(-0.2, 0.2, size=len(shown))
        ok = ~g["died"][shown]
        ax.plot(jitter[ok], values[shown][ok], "o", ms=3, color=colors[label], alpha=0.35, mew=0)
        ax.plot(jitter[~ok], values[shown][~ok], "x", ms=6, color=colors[label], mew=1.6)
        lo, mid, hi = spread(values)
        if np.isfinite(mid):
            ax.vlines(x, lo, hi, color=INK_2, lw=2, zorder=5)
            ax.hlines([lo, hi], x - 0.08, x + 0.08, color=INK_2, lw=2, zorder=5)
            ax.hlines(mid, x - 0.28, x + 0.28, color=INK, lw=2.6, zorder=6)
    ax.set_xticks(range(len(groups)), [_tick(label, g) for label, g in groups.items()])
    ax.set_ylabel("RMSE [m]")
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"Per-episode RMSE: IQM (bar), {BAND[0]}–{BAND[1]} % range\n"
                 "(●: full episode, ×: crashed)", color=INK, fontsize=10.5)
    return _save(fig, out_path)


def write_tables(rows: list[dict], csv_path: str, md_path: str) -> tuple[str, str]:
    """Write the per-label summary rows as CSV and as a Markdown table."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, md_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help=".parquet files written by play.py --eval_csv")
    parser.add_argument("--labels", nargs="+", default=None, help="one label per file; equal labels are pooled")
    parser.add_argument("--out", default=None, help="output folder (default: the first file's)")
    parser.add_argument("--name", default="eval", help="figure/table name suffix")
    parser.add_argument("--env", type=int, default=None, help="draw episode I of each label instead of the IQM")
    parser.add_argument("--max-traces", type=int, default=MAX_TRACES, help="faint episode lines per label in the top view when the episodes fly different references")
    parser.add_argument("--overlay", action="store_true",
                        help="one top-view and one time-series figure with every label overlaid, instead of one per label")
    args = parser.parse_args(argv)
    if args.labels and len(args.labels) != len(args.files):
        parser.error(f"--labels needs one label per file ({len(args.files)}), got {len(args.labels)}")

    labels = args.labels or [None] * len(args.files)
    groups = group_by_label([load_run(path, label) for path, label in zip(args.files, labels)])
    within, across = shared_reference(groups)
    if not within:
        print("[WARN] episodes fly different references (--random_traj?): no mean path, "
              "each episode is drawn against its own reference")
    elif not across:
        print("[WARN] the labels fly different references: each label's reference is drawn, "
              "and their mean paths are not directly comparable")

    out = args.out or os.path.dirname(os.path.abspath(args.files[0]))
    os.makedirs(out, exist_ok=True)
    rows = summarize(groups)
    sets = [("", groups)] if args.overlay else [(f"_{_slug(label)}", {label: g}) for label, g in groups.items()]
    colors = _colors(groups)
    tracking = []
    for suffix, subset in sets:
        tracking.append(plot_topview(subset, os.path.join(out, f"top_{args.name}{suffix}.png"),
                                     args.env, args.max_traces, colors))
        tracking.append(plot_timeseries(subset, os.path.join(out, f"time_{args.name}{suffix}.png"), args.env, colors))
    saved = [
        *tracking,
        plot_summary(groups, os.path.join(out, f"summary_{args.name}.png")),
        *write_tables(rows, os.path.join(out, f"summary_{args.name}.csv"), os.path.join(out, f"summary_{args.name}.md")),
    ]
    for row in rows:
        print(f"  {row['label']}: {row['episodes']} episodes, crash {100 * row['crash_rate']:.1f} %, "
              f"RMSE IQM {row['rmse_iqm']:.3f} m ({BAND[0]}–{BAND[1]} %: {row['rmse_p10']:.3f}–{row['rmse_p90']:.3f}), "
              f"along {row['along_iqm']:+.3f} m, cross {row['cross_iqm']:.3f} m")
    for path in saved:
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
