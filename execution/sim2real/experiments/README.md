# Hover robustness sweeps

Training sweeps on `Vel-Hovering-Robust`. They answer one question: which training
condition reduces the sim-to-real gap measured in `execution/sim2real/`?

The measured gap is asymmetric. Position error transfers to hardware almost
unchanged, but the commanded velocity chatters about ten times more on the drone
than in simulation. Every sweep below therefore scores chatter explicitly.

## Layout

| file | role |
|---|---|
| `configs.py`, `configs_v2.py`, `configs_v3.py` | run grids for sweeps 1-3 |
| `sweep_train.py` | launches a grid sequentially, writes `sweep_index*.json` |
| `eval_sweep.py` | replays every checkpoint under one fixed condition, writes a leaderboard |
| `eval_dr_generalization.py` | evaluates sweep-3 policies on a held-out randomisation range |
| `eval_dr_sweep.py` | sweeps one randomised parameter for a single policy |
| `sweep_index*.json` | tag -> checkpoint path, produced by training |
| `leaderboard*.json` | evaluation results |

## Factors

Sweep 1 varies observation noise, external disturbance, observation window `K`, and
an action-rate penalty. Sweep 2 fixes `K=8` with disturbances on, then varies the
action history `M`, loop latency, and the action-rate penalty. Sweep 3 randomises
the body itself: mass, inertia and actuation effectiveness, with latency and
correlated noise held on.

All three use 4096 environments and 600 iterations (19.2k timesteps). The first,
48k-timestep runs over-trained: reward peaked near timestep 10k and then fell by
about 35%.

## Running

```bash
python execution/sim2real/experiments/sweep_train.py --configs configs_v2
python execution/sim2real/experiments/eval_sweep.py  --index sweep_index_v2.json \
    --out leaderboard_v2.json
```

`sweep_train.py` runs one training at a time. Isaac Lab cannot build a second
environment inside one process, so each evaluation also spawns its own subprocess.

The leaderboard score is `pos_err + 5 * chatter`, averaged over two targets and
evaluated under a condition harsher than any used for training.

## Results

**Sweep 1 (24 runs).** The observation window is the only clear win. Stacking eight
frames roughly halves chatter. The action-rate penalty does not help: the policy
cannot see its own previous command, so the penalty scores a quantity it cannot
perceive. Best score 1.666, worst 3.814 (`n0.0_k1_d0_s123`, the unstacked
noise-free reference).

**Sweep 2 (12 runs).** Latency-aware training is the dominant factor. The four best
runs all train with a randomised 2-6 step dead time; the four worst all train
without it. Best score 1.477 (`m0_laton_aroff`: `K=8`, `M=0`, latency on, no
action-rate penalty), against 1.626 for the best sweep-1 policy re-evaluated under
the same condition. Action history `M` and the action-rate penalty change little.

**Sweep 3 (12 runs, negative result).** Physical randomisation bought no
generalisation. On a held-out range wider than training, the three best position
errors belong to policies trained with randomisation *off* or *narrow*, and the
widest range is uniformly worse. The cause is structural: the policy commands a
velocity setpoint that the onboard cascade PID absorbs, so mass and thrust
scale are nearly unobservable from the observation. Adaptation has nothing to infer
from. A policy acting at thrust or body-rate level would be the setting where this
factor can matter.

## Hardware follow-up

Five checkpoints were flown (`execution/sim2real/flight_checkpoints/`). The ranking
transferred. The two latency-aware policies cut hardware chatter by 1.6x and 2.1x,
and the window-only policy was no better than the baseline, exactly as simulation
predicted.
