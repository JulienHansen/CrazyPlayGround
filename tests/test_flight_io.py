"""Parquet and CSV recordings must be interchangeable.

The collector writes Parquet, older flights on disk are CSV, and the analyser has
to read both without the caller knowing which it is holding.
"""

import numpy as np
import pytest

from flight_io import PARQUET_NAME, CSV_NAME, read_flight, resolve, write_flight, parquet_available

pytestmark = pytest.mark.skipif(not parquet_available(), reason="pyarrow not installed")


def test_parquet_round_trips_every_column(tmp_path, columns, hover_rows):
    rows = hover_rows(columns)
    path = write_flight(str(tmp_path / "run"), rows, columns)
    assert path.endswith(PARQUET_NAME)
    data = read_flight(path)
    assert set(data) == set(columns)
    for c in columns:
        assert data[c].dtype == np.float64
        np.testing.assert_allclose(data[c], [r[c] for r in rows])


def test_csv_and_parquet_give_identical_arrays(tmp_path, columns, hover_rows):
    rows = hover_rows(columns)
    run = tmp_path / "run"
    write_flight(str(run), rows, columns, also_csv=True)
    par = read_flight(str(run / PARQUET_NAME))
    csvd = read_flight(str(run / CSV_NAME))
    assert set(par) == set(csvd)
    for c in columns:
        np.testing.assert_allclose(par[c], csvd[c])


def test_parquet_is_preferred_but_csv_still_resolves(tmp_path, columns, hover_rows):
    run = tmp_path / "run"
    write_flight(str(run), hover_rows(columns), columns, also_csv=True)
    assert resolve(str(run)).endswith(PARQUET_NAME)
    (run / PARQUET_NAME).unlink()
    assert resolve(str(run)).endswith(CSV_NAME), "a pre-Parquet flight must still load"


def test_csv_fallback_when_parquet_is_not_wanted(tmp_path, columns, hover_rows):
    path = write_flight(str(tmp_path / "run"), hover_rows(columns), columns, prefer_parquet=False)
    assert path.endswith(CSV_NAME)
    assert set(read_flight(path)) == set(columns)


def test_the_analyser_reads_a_parquet_run(tmp_path, columns, hover_rows):
    from analyze_hover import load_flight, compute_metrics
    run = tmp_path / "run"
    write_flight(str(run), hover_rows(columns), columns)
    m = compute_metrics(load_flight(str(run)))
    assert m["n_samples"] > 0
    assert m["mean_pos_err_m"] == m["mean_pos_err_m"]      # not NaN


def test_parquet_is_substantially_smaller_than_csv(tmp_path, columns, hover_rows):
    """The reason for the switch. A regression here means the compression broke."""
    rows = hover_rows(columns, n=2000)
    run = tmp_path / "run"
    write_flight(str(run), rows, columns, also_csv=True)
    par = (run / PARQUET_NAME).stat().st_size
    csvb = (run / CSV_NAME).stat().st_size
    assert par < csvb / 2, f"parquet {par} B vs csv {csvb} B"
