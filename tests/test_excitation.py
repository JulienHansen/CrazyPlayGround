"""excitation.py, shared by excite_hover.py and excite_hover_sim.py: the step signal, the command law,
the CLI and the recording that measure_delay_noise.py reads.

No cflib and no Isaac Lab: the simulated flight is replayed here on a first-order plant with the
timeline of excite_hover_sim.py (row k = state before command k, echo = command after the latency).
"""

import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "execution", "sim2real"))
import excitation  # noqa: E402
import measure_delay_noise  # noqa: E402

AMP, DT = 0.2, 0.01


def test_step_signal_shape():
    start, step_s, recover_s = excitation.EXC_START_S, 1.0, 4.0
    sig = lambda axis, t: excitation.step_signal(axis, AMP, step_s, recover_s, start + t)
    assert sig(0, -0.01) == ([0.0, 0.0, 0.0], True)
    # one cycle at 100 Hz: REST, PUSH (hold off), then RECOVER (hold on)
    cycle = [sig(1, 0.005 + 0.01 * i) for i in range(600)]
    assert all(e[0] == e[2] == 0.0 for e, _ in cycle)
    assert [e[1] for e, _ in cycle] == [0.0] * 100 + [AMP] * 100 + [0.0] * 400
    assert [hold for _, hold in cycle] == [False] * 200 + [True] * 400
    # every PUSH goes the way of amp's sign
    assert sig(0, 6 + 1.5)[0][0] == sig(0, 12 + 1.5)[0][0] == AMP
    assert excitation.step_signal(0, -AMP, step_s, recover_s, start + 1.5)[0][0] == -AMP


def test_command_is_a_pure_step_on_the_excited_axis_and_a_clamped_hold_elsewhere():
    err = torch.tensor([0.1, -0.05, 1.0])
    hold, cmd = excitation.excitation_command(err, 0, [AMP, 0.0, 0.0], hold_on=False)
    assert hold.tolist() == pytest.approx([0.0, -0.05, excitation.HOLD_MAX])
    assert cmd.tolist() == pytest.approx([AMP, -0.05, excitation.HOLD_MAX])
    hold, cmd = excitation.excitation_command(err, 0, [0.0, 0.0, 0.0], hold_on=True)
    assert hold[0].item() == cmd[0].item() == pytest.approx(0.1)
    _, cmd = excitation.excitation_command(torch.tensor([5.0, 0.0, 0.0]), 0, [1.0, 0.0, 0.0], hold_on=True)
    assert cmd[0].item() == excitation.MAX_VEL
    # batched [N, 3], as the simulator's tensors are
    hold, cmd = excitation.excitation_command(err.repeat(4, 1), 1, [0.0, -AMP, 0.0], hold_on=False)
    assert hold.shape == cmd.shape == (4, 3)
    assert (hold[:, 1] == 0).all() and torch.allclose(cmd[:, 1], torch.tensor(-AMP))


def parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None)
    excitation.add_excitation_args(parser)
    args = parser.parse_args(argv)
    excitation.check_excitation_args(parser, args)
    return args


def test_cli_refuses_excitations_the_estimator_cannot_use():
    args = parse(["--exc-axis", "y"])
    meta = excitation.excitation_metadata(args)
    assert (args.exc_axis, meta["excitation"], meta["kp"]) == ("y", "push_recover", str(excitation.KP))
    for bad in (["--duration", "2"], ["--exc-amp", "1.1"], ["--exc-amp", "0"], ["--exc-step", "0"],
                ["--exc-recover", "-1"]):
        with pytest.raises(SystemExit):
            parse(bad)


def simulated_flight(path, latency_steps, plant_dead_steps=2, tau=0.15, duration=40.0, seed=0):
    """excite_hover_sim.py's loop on a first-order velocity plant on x, recorded with write_recording.

    Each row holds the state read before its command; the controller acts on the command sent
    `latency_steps` earlier (the env's action latency, echoed as echo_v*), and the velocity follows
    it `plant_dead_steps` later with time constant `tau`, plus a little sensor noise."""
    rng = np.random.default_rng(seed)
    a = np.exp(-DT / tau)
    target = torch.tensor([0.0, 0.0, 1.0])
    pos, vel = target.clone(), torch.zeros(3)
    sent, rows = [], []
    for k in range(int(duration / DT)):
        t = k * DT
        err = target - pos
        exc, hold_on = excitation.step_signal(0, AMP, 1.0, 2.0, t)
        hold, cmd = excitation.excitation_command(err, 0, exc, hold_on)
        sent.append(cmd)
        echo = sent[k - latency_steps] if k >= latency_steps else torch.zeros(3)
        applied = sent[k - latency_steps - plant_dead_steps] if k >= latency_steps + plant_dead_steps else torch.zeros(3)
        measured = vel + torch.from_numpy(1e-3 * rng.standard_normal(3)).float()
        rows.append(excitation.excitation_row(k, t, pos.tolist(), measured.tolist(), [1.0, 0.0, 0.0, 0.0],
                                              target.tolist(), err.tolist(), hold.tolist(), cmd.tolist(),
                                              exc, echo.tolist()))
        vel = a * vel + (1 - a) * applied
        pos = pos + vel * DT
    excitation.write_recording(str(path), rows, {"source": "sim", "action_latency_steps": latency_steps})
    return str(path)


def test_recording_carries_the_real_flight_columns_and_footer(tmp_path):
    path = simulated_flight(tmp_path / "sim.parquet", latency_steps=0, duration=5.0)
    schema = pq.read_schema(path)
    for name in ["step", "control_time", "obs_age_s", "pos_x", "vel_x", "quat_w", "target_z", "err_x",
                 "hold_vx", "vx", "vy", "vz", "exc_vx", "echo_vz"]:
        assert name in schema.names
    assert str(schema.field("step").type) == "int64" and str(schema.field("vel_x").type) == "double"
    assert schema.metadata[b"source"] == b"sim"
    assert pq.read_table(path).column("step").to_pylist() == list(range(500))


@pytest.mark.parametrize("latency_steps", [0, 3])
def test_estimator_reads_the_latency_as_radio_delay_not_dead_time(tmp_path, latency_steps):
    res = measure_delay_noise.analyse(simulated_flight(tmp_path / "sim.parquet", latency_steps))
    assert set(res) == {"x"}  # only the excited axis
    x = res["x"]
    assert x["d"] == latency_steps and x["radio_ms"] == pytest.approx(latency_steps * DT * 1e3)
    # 2 steps of plant dead time + 1 step for the row's state read; the first moving sample is
    # already past 5% (1 - exp(-DT/tau) = 6.4%), whatever the latency
    assert x["dead_ms"] == pytest.approx(30.0)
    assert x["tau_ms"] == pytest.approx(150.0, rel=0.15)
