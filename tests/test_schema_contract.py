"""The CSV contract between the collector, the simulator evaluator and the analyser.

These three must agree or a sim-vs-real comparison silently compares different
things. The columns are defined once, in the collector.
"""

import csv
import os

import pytest

from conftest import write_flight


def test_analyser_consumes_every_collector_column(tmp_path, columns, hover_rows):
    """analyze_hover casts every column to float, so any column the collector
    writes must be numeric - a string would break the whole analysis."""
    from analyze_hover import load_flight
    d = write_flight(tmp_path / "run", hover_rows(columns), columns, duration_s=10.0)
    data = load_flight(str(d))["data"]
    assert set(data) == set(columns)
    for c in columns:
        assert data[c].dtype.kind == "f", f"column {c} is not numeric"


def test_metric_inputs_are_present_in_the_schema(columns):
    """Every column the metrics actually read must exist in the collector schema."""
    needed = {
        "t_mono",
        "pos_x", "pos_y", "pos_z", "tgt_x", "tgt_y", "tgt_z",
        "vel_x", "vel_y", "vel_z",
        "cmd_vx", "cmd_vy", "cmd_vz",
        "act_x", "act_y", "act_z",
        "m1", "m2", "m3", "m4", "vbat",
        "varPX", "varPY", "varPZ",
    }
    assert needed <= set(columns), f"schema is missing {needed - set(columns)}"


def test_simulator_evaluator_writes_the_same_columns():
    """The in-sim evaluator imports the column list rather than restating it, so the
    two cannot drift apart. Guard that it still does."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "execution", "sim2real", "eval_hover_vel_sim.py")).read()
    assert "from collect_hover_vel import CSV_COLUMNS" in src, (
        "the sim evaluator must import the schema from the collector, not redefine it")


def test_a_run_missing_metadata_still_loads(tmp_path, columns, hover_rows):
    """duration_s drives the ended_early flag; without metadata the run must still
    be analysable rather than crashing the batch."""
    from analyze_hover import compute_metrics, load_flight
    d = write_flight(tmp_path / "run", hover_rows(columns), columns, duration_s=None)
    os.remove(os.path.join(str(d), "metadata.json"))
    m = compute_metrics(load_flight(str(d)))
    assert m["ended_early"] is False


def test_blank_row_covers_the_whole_schema(columns):
    """Producers build on blank_row, so it must stay in step with CSV_COLUMNS."""
    from collect_hover_vel import blank_row
    row = blank_row()
    assert set(row) == set(columns)
    assert all(isinstance(v, float) for v in row.values())


def test_writing_a_short_row_names_the_missing_columns(tmp_path, columns, hover_rows):
    """A new schema column must fail loudly in the producer, not as a bare KeyError.

    The sim evaluator once wrote every column but the four loop-timing ones, and
    the failure surfaced as KeyError('dt_infer_ms') from inside pyarrow.
    """
    from flight_io import write_flight
    rows = hover_rows(columns)
    for r in rows:
        del r[columns[-1]]
    with pytest.raises(ValueError) as e:
        write_flight(str(tmp_path / "run"), rows, columns)
    assert columns[-1] in str(e.value)
    assert "blank_row" in str(e.value)
