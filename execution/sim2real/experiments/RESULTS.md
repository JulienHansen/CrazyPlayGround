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
