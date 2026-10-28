"""plot_step_response.py on a synthetic excite_hover recording: the figure is written, the IQM is right."""

import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "sim2real"))
import plot_step_response  # noqa: E402

from test_delay_and_noise import write_flight


def test_iqm_is_the_mean_of_the_central_half():
    windows = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [100.0]])
    assert plot_step_response.iqm(windows) == pytest.approx([3.5])  # mean of 2, 3, 4, 5


def test_main_writes_the_figure(tmp_path, capsys):
    flights = [write_flight(tmp_path / f"alternate_{a}.parquet", alternate=a) for a in (False, True)]
    sim = pq.read_table(write_flight(tmp_path / "sim.parquet"))
    pq.write_table(sim.replace_schema_metadata({"source": "sim"}), tmp_path / "sim.parquet")
    assert [plot_step_response.source(p) for p in (flights[0], tmp_path / "sim.parquet")] == ["real", "sim"]
    out = tmp_path / "step.png"
    plot_step_response.main(flights + [str(tmp_path / "sim.parquet"), "--out", str(out)])
    assert out.stat().st_size > 0 and f"saved {out}" in capsys.readouterr().out


def test_main_refuses_recordings_without_excitation(tmp_path):
    with pytest.raises(SystemExit):
        plot_step_response.main([write_flight(tmp_path / "hover.parquet", excited=False)])
