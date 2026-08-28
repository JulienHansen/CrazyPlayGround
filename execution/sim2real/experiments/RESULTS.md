# Hover robustness sweep — results

24 policies trained on `Vel-Hovering-Robust` (600 iters = 19.2k timesteps, 4096 envs,
~26 min each, 10.7 h total on an RTX 3090), then each evaluated on two targets
[(0,0,1), (1,1,1)] under a **common held-out condition**: obs noise 0.03, constant
force/torque bias, and gusts at 2% per step. W&B: project `crazyplayground-hover`,
group `hover-robustness-sweep`. Raw data: `sweep_index.json`, `leaderboard.json`.

## Why this experiment

Measured sim-to-real gap on the real Crazyflie (see `../README.md`): position
accuracy transfers well (~3 cm), but the deployed policy's **command chatter was
~10x worse on hardware than in sim** (0.069 vs 0.0065). It had been trained with no
observation noise and no disturbances, so it never learned to tolerate either.

## Headline result

Under the held-out (realistic) condition, all three factors independently reduce
chatter, and they compose:

| factor | levels | chatter | pos error |
|---|---|---|---|
| observation window K | 1 → 4 → 8 | **0.395 → 0.203 → 0.186** | 0.114 → 0.153 → 0.125 |
| training noise std | 0 → 0.02 → 0.05 | **0.322 → 0.257 → 0.208** | ~flat |
| disturbances | off → on | **0.270 → 0.200** | 0.134 → 0.123 |

No policy crashed under disturbances (crash rate 0/24).

Against the deployed baseline, same condition:

| policy | chatter | vs baseline | pos err |
|---|---|---|---|
| baseline `n0.0_k1_d0_s42` (K=1, clean-trained) | 0.53925 | 1.0x | 0.0824 |
| **`n0.02_k8_d1_s42`** | 0.17422 | **3.1x better** | 0.0827 (equal) |
| `n0.05_k8_d1_ar_s42` | 0.15341 | 3.5x better | 0.1206 |
| `n0.05_k4_d0_s42` (top by score) | 0.08532 | 6.3x better | 0.2391 (worst) |

**`n0.02_k8_d1_s42` is the deploy candidate**: 3.1x less chatter at *identical*
position accuracy — a strict improvement, no trade-off.

## The trap: clean-sim evaluation is misleading

Under the **nominal (clean)** condition the ranking **inverts** — K=1 has the lowest
chatter (0.0037) and K=8 the highest (0.0099). The observation window only pays off
when there is noise/disturbance to filter.

This is exactly how the original policy fooled us: pristine in clean sim (0.0065),
chattering on hardware (0.069). Evaluate sim-to-real candidates under the disturbed
condition, not the clean one.

## Caveats

- **`max_drift` is inert**: ~1.000 for every policy (it only measures the initial
  approach distance from the fixed start pose), so it added a constant to every
  score and discriminated nothing. Read the raw columns, not just the score.
- The composite score weights chatter 5x, which is why its #1 pick trades away
  position accuracy. That weight is a judgement call, not a derived quantity.
- Nothing here is hardware-validated yet: sim says these chatter less; only a real
  flight confirms it.
- Reproducibility check: the sweep's `n0.0_k1_d0_s42` checkpoint is **bitwise
  identical** (sha256 `e169bc0a…`) to the originally deployed `best_agent.pt` —
  same seed, and `Vel-Hovering-Robust` with all knobs off reproduces `Vel-Hovering`
  exactly. Good evidence the new env is a faithful superset.

## Reproduce

```bash
python execution/sim2real/experiments/sweep_train.py               # 24 runs
python execution/sim2real/experiments/eval_sweep.py --nominal      # rank them
```

## Deploying the winner

`n0.02_k8_d1_s42` uses K=8, so the real-drone scripts need the matching window:

```bash
python execution/sim2real/check_policy.py   --checkpoint <ckpt> --device cpu --history-len 8
python execution/sim2real/collect_hover_vel.py --checkpoint <ckpt> --history-len 8 \
    --target 0 0 1.0 --duration 20 --tag k8_robust
```
A mismatched `--history-len` fails at load time with a state_dict size mismatch
rather than flying a wrongly-shaped policy.

---

# Sweep 2 — action history and loop latency

Sweep 1 left out the two factors most likely to explain the hardware chatter: the
policy could not see its own previous command, and there was no loop latency at all.
Sweep 2 fixes both. 12 runs (~24 min each, 4.8 h), K=8 + disturbances + AR(1)
correlated noise held fixed, varying:

    action_hist_len M       0, 1, 4
    latency                 none  /  2-6 steps randomised per episode
    action-rate penalty     0, -0.05

W&B group `hover-robustness-sweep-v2`; raw records in `sweep_index_v2.json`,
`leaderboard_v2.json`, `leaderboard_v1refs.json`.

## What motivated it: the latency diagnostic

Evaluating the *deployed* policy with everything clean except loop dead time:

| delay | 0 | 1 (10 ms) | 2 (20 ms) | 3 (30 ms) | 4 (40 ms) | real hardware |
|---|---|---|---|---|---|---|
| chatter | 0.0060 | 0.0147 | 0.0300 | 0.0422 | 0.0517 | **0.0690** |
| pos err | 0.039 | 0.040 | 0.043 | 0.049 | 0.056 | 0.097 |

Latency alone reproduces ~75% of the chatter gap *and* its signature — chatter
explodes 8.6x while tracking barely moves — which noise and disturbances never
explained. This partly supersedes the sweep-1 framing of the gap as a
noise-robustness problem.

## Marginal effects (v2 condition)

| factor | level | chatter | pos err |
|---|---|---:|---:|
| trained latency | 0 | 0.1217 | 0.1366 |
| | **2-6** | **0.0925** | **0.1273** |
| action history M | 0 | 0.1064 | 0.1461 |
| | **1** | **0.0879** | 0.1207 |
| | 4 | 0.1269 | 0.1291 |
| action-rate penalty | off | **0.0972** | **0.1231** |
| | on | 0.1169 | 0.1409 |

## Cross-sweep ranking, one identical condition

All policies re-evaluated under 40 ms latency + correlated noise + disturbances:

| # | source | policy | pos err | chatter | vs baseline |
|---:|---|---|---:|---:|---:|
| 1 | sweep 2 | `m1_laton_aroff` | 0.1336 | 0.0732 | 2.21x |
| 2 | sweep 2 | **`m0_laton_aroff`** | **0.0960** | **0.0763** | **2.12x** |
| 3 | sweep 2 | `m1_laton_aron` | 0.1127 | 0.0766 | 2.11x |
| 5 | sweep 1 | `n0.05_k8_d1_ar_s42` | 0.1504 | 0.0951 | 1.70x |
| 7 | sweep 1 | `n0.02_k8_d1_s42` | 0.1087 | 0.1060 | 1.53x |
| 14 | sweep 1 | `n0.0_k1_d0_s42` (deployed) | 0.1013 | 0.1616 | 1.00x |

**Deploy candidate: `m0_laton_aroff`** (K=8, M=0, latency-trained, no penalty) --
2.12x less chatter than the deployed policy *and* slightly better position error
(0.0960 vs 0.1013). A strict improvement on both axes, and it beats sweep 1's best
by ~20%. sha256 `f5878876...`, offline gate PASS.

## Conclusions

1. **Training under latency is the real win** -- the only factor with consistent
   support, improving both chatter and tracking. It came from asking what sweep 1
   had not tested.
2. **One step of action history helps; four hurt.** M=4 is worse than none, likely
   because 12 extra inputs on a [32,32] net at 19.2k timesteps cost more than they
   inform.
3. **The action-rate penalty backfired.** Predicted to help once `a_{t-1}` was
   observable, it made chatter *and* tracking worse at this magnitude -- it competes
   with the distance reward. Hypothesis rejected.

## Caveats

- **n=4 per level, single seed.** Only the latency effect has clean support; the
  M=0 vs M=1 gap is within plausible seed noise, and the best balanced policy
  (`m0_laton_aroff`) uses no action history at all.
- **The v2 condition overshoots reality.** The deployed policy scores 0.1616 there
  against 0.0690 measured on hardware, so v2 is a stress test for *relative*
  ranking, not an absolute predictor.
- **The latency value is bracketed, not measured.** 2-6 steps came from matching a
  simulated curve. It cannot be identified from closed-loop logs (see
  `measure_delay_noise.py`); the open-loop chirp flight would pin it down.
- Still no hardware validation of any sweep policy.
