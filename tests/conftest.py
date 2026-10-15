"""Shared fixtures for the sim2real pipeline tests.

The tests are CPU-only and must not require Isaac Sim. Modules that pull heavy
optional dependencies (torch, skrl, cflib) are imported through `importorskip`
inside the tests that need them, so the core metric tests still run anywhere.
"""

import csv
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM2REAL = os.path.join(REPO, "execution", "sim2real")
sys.path.insert(0, SIM2REAL)


def write_flight(path, rows, columns, duration_s=None):
    """Write a run directory in the schema the real collector produces."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "flight.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    meta = {"controller": "velocity", "source": "synthetic"}
    if duration_s is not None:
        meta["duration_s"] = duration_s
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f)
    return path


@pytest.fixture
def columns():
    from analyze_hover import load_flight  # noqa: F401  (ensures path is importable)
    # the canonical column list, taken from the collector so the two cannot drift
    cflib = pytest.importorskip("cflib", reason="collector needs cflib")  # noqa: F841
    pytest.importorskip("torch")
    pytest.importorskip("skrl")
    from collect_hover_vel import CSV_COLUMNS
    return CSV_COLUMNS


@pytest.fixture
def hover_rows():
    """A 10 s, 100 Hz flight that sits exactly on target with no command motion."""
    def _make(columns, n=1000, dt=0.01, pos=(0.0, 0.0, 1.0), tgt=(0.0, 0.0, 1.0), cmd=0.0):
        rows = []
        for i in range(n):
            r = {c: 0.0 for c in columns}
            r.update(t_mono=i * dt, step=i,
                     pos_x=pos[0], pos_y=pos[1], pos_z=pos[2],
                     tgt_x=tgt[0], tgt_y=tgt[1], tgt_z=tgt[2],
                     cmd_vx=cmd, cmd_vy=0.0, cmd_vz=0.0,
                     qw=1.0, vbat=3.8, m1=30000, m2=30000, m3=30000, m4=30000)
            rows.append(r)
        return rows
    return _make
