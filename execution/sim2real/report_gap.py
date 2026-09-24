"""Tabulate the CL-1 hover gap per policy, sim against hardware.

Runs are matched to a policy by the SHA-256 of the checkpoint recorded in each
metadata.json, not by the folder name. Folder names have been wrong before: the
2026-09-24 session mislabelled four runs, swapping A0 and A1 tags, and only the
hash caught it.

    python execution/sim2real/report_gap.py --markdown gap.md
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_hover import compute_metrics, load_flight  # noqa: E402

# (key, label, lower is better)
METRICS = [
    ("mean_pos_err_m", "mean position error [m]", True),
    ("rms_pos_err_m", "RMS position error [m]", True),
    ("max_drift_m", "max drift [m]", True),
    ("mean_horiz_err_m", "horizontal error [m]", True),
    ("mean_vert_err_m", "vertical error [m]", True),
    ("cmd_smoothness_mps2", "command chatter [m/s^2]", True),
    ("cmd_smoothness_mps_per_step", "command chatter [m/s/step]", True),
    ("mean_speed_mps", "mean speed [m/s]", True),
    ("effective_rate_hz", "control rate [Hz]", False),
]


def policy_of(run_dir, by_sha):
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_path):
        sha = json.load(open(meta_path)).get("checkpoint_sha256") or ""
        if sha[:16] in by_sha:
            return by_sha[sha[:16]]
    for pid in by_sha.values():                       # fall back to the folder name
        if f"_{pid}_" in os.path.basename(run_dir):
            return pid
    return None


def collect(paths, by_sha, skip_s=0.0):
    out = defaultdict(list)
    for d in sorted(paths):
        if not os.path.isdir(d) or "ABORTED" in d or "chirp" in d:
            continue
        pid = policy_of(d, by_sha)
        if pid is None:
            continue
        try:
            out[pid].append(compute_metrics(load_flight(d), skip_s=skip_s))
        except FileNotFoundError:
            continue
    return out


def agg(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std())) if v.size else (float("nan"), float("nan"))


def main():
    p = argparse.ArgumentParser(description="Per-policy sim-to-real hover gap.")
    p.add_argument("--sim", default=os.path.join(HERE, "data_sim_gap", "*"))
    p.add_argument("--real", default=os.path.join(HERE, "data", "2026-09-24*"))
    p.add_argument("--markdown", default=None, help="Write the tables to this file.")
    p.add_argument("--json-out", default=None)
    p.add_argument("--skip-s", type=float, default=3.0,
                   help="Ignore the first N seconds of every flight. The approach transient is not "
                        "matched between simulator and drone, so it does not belong in a hover gap.")
    args = p.parse_args()

    manifest = json.load(open(os.path.join(HERE, "flight_checkpoints", "manifest.json")))
    by_sha = {e["sha256_16"]: e["id"] for e in manifest}
    note = {e["id"]: e["note"] for e in manifest}

    sim = collect(glob.glob(args.sim), by_sha, args.skip_s)
    real = collect(glob.glob(args.real), by_sha, args.skip_s)
    shared = sorted(set(sim) & set(real))
    if not shared:
        print("No policy has both a sim and a real flight.")
        sys.exit(1)

    lines, report = [], {}
    lines.append("| policy | trials sim / real | " + " | ".join(
        f"{lab} sim → real (ratio)" for _, lab, _ in METRICS) + " |")
    lines.append("|" + "---|" * (2 + len(METRICS)))
    for pid in shared:
        cells, entry = [], {"note": note.get(pid), "n_sim": len(sim[pid]), "n_real": len(real[pid])}
        for key, label, _ in METRICS:
            s, _ss = agg(sim[pid], key)
            r, _rs = agg(real[pid], key)
            # A ratio is only informative while the simulator value is a real
            # quantity. On the steady-state window a stacked policy converges to a
            # fixed point in the nominal simulator, so its chatter falls to ~1e-5
            # and any ratio against it measures the division, not the gap.
            meaningful = np.isfinite(s) and s != 0 and abs(s) > 0.01 * abs(r)
            ratio = (r / s) if meaningful else float("nan")
            shown = f"{ratio:.2f}x" if meaningful else "sim~0"
            cells.append(f"{s:.4g} → {r:.4g} ({shown})")
            entry[key] = {"sim": s, "real": r, "abs_gap": r - s,
                          "real_over_sim": ratio if meaningful else None}
        lines.append(f"| {pid} | {len(sim[pid])} / {len(real[pid])} | " + " | ".join(cells) + " |")
        report[pid] = entry

    text = "\n".join(lines)
    print(text)
    print("\nRatio is real / sim. Above 1 means hardware is worse.")
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("# CL-1 hover gap, per policy\n\n" + text +
                    "\n\nRatio is real / sim. Above 1 means hardware is worse.\n")
        print(f"\n[INFO] markdown -> {args.markdown}")
    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=2)
        print(f"[INFO] json -> {args.json_out}")


if __name__ == "__main__":
    main()
