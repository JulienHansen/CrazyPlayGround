# Sim-to-Real Report — Crazyflie Velocity Hover

**The reality gap is chatter, not accuracy.**

A trained hover policy tracked position within 3 cm of simulation on real hardware — while
shaking its velocity command ten times harder. This report covers what was built, measured
and tested to find that out, and what closes the gap in simulation.

| | |
|---|---|
| Branch | `feature/sim2real-hover-velocity` (6 commits, +3,255 lines) |
| Hardware | Crazyflie 2.1, Crazyradio PA |
| Controller | velocity (`send_velocity_world_setpoint`, 100 Hz) |
| Real flights | 6 (2026-07-24) |
| Sim flights | 52 |
| Policies trained | 25 (1 baseline + 24 sweep) |
| GPU time | ~11.5 h (RTX 3090) |
| Tracking | W&B project `crazyplayground-hover`, group `hover-robustness-sweep` |

---

## 1. What was done

1. **Trained a deployable velocity hover policy.** skrl PPO on `Vel-Hovering`, 4096 parallel
   envs, 48,000 timesteps ≈ **196.6 M** transitions (~546 h of aggregate flight experience) in
   25 min. Reward peaked near timestep 10k then declined 35%, so `best_agent.pt` — not the
   final checkpoint — is the usable policy.
2. **Built the data-collection tooling.** Nothing in the repo persisted flight data before
   this; the deploy scripts streamed telemetry and discarded it.
3. **Flew and logged 6 real flights** at 100 Hz with full telemetry, then computed hover metrics.
4. **Measured the gap** by running the same policy in simulation on the same targets, emitting
   an identical CSV schema, and differencing the metrics.
5. **Ran a 24-policy robustness experiment** over observation noise × observation-window length
   × disturbances × seed, ranked under a common held-out disturbance condition.

## 2. What was built

| Module | Role | Lines |
|---|---|---|
| `collect_hover_vel.py` | Flies the policy on hardware and records full telemetry | 685 |
| `eval_hover_vel_sim.py` | Same flight in sim, identical CSV schema | 237 |
| `analyze_hover.py` | Hover metrics + multi-trial aggregation | 245 |
| `compare_sim2real.py` | Per-metric sim − real gap table | 96 |
| `check_policy.py` | Offline checkpoint sanity gate (no drone, no Isaac Lab) | 121 |
| `vel_hovering_robust.py` | New env: noise, disturbances, observation window | 203 |
| `experiments/` | Sweep design, training driver, ranking evaluator | 355 |

The recorded schema — position, velocity, quaternion, gyro, accelerometer, `motor.m1–m4`,
battery, Kalman variance, plus the observation, raw action and command actually sent — is what
feeds protocol tests OL-1, OL-3, OL-4 and CL-1.

## 3. The measured gap

Same policy, same two targets: 5 real flights vs 2 sim flights. Ratio < 1 means the real drone
is worse.

| Metric | Sim | Real | Gap | Ratio |
|---|---:|---:|---:|---:|
| mean position error [m] | 0.0683 | 0.0974 | −0.0291 | 0.70 |
| RMS position error [m] | 0.1930 | 0.2093 | −0.0163 | 0.92 |
| horizontal error [m] | 0.0432 | 0.0616 | −0.0184 | 0.70 |
| vertical error [m] | 0.0443 | 0.0637 | −0.0194 | 0.70 |
| max drift [m] | 1.0012 | 0.9628 | +0.0384 | 1.04 |
| mean speed while hovering [m/s] | 0.0945 | 0.3416 | −0.2471 | 0.28 |
| **command chatter [m/s/step]** | **0.0065** | **0.0690** | **−0.0625** | **0.09** |

Estimator quality was excellent in every real flight (Kalman variance ≈ 0.0002), so these are
genuine control differences, not sensor noise. Motor PWM, battery and accelerometer have no sim
counterpart and are excluded.

**Position tracking transfers well; everything that went wrong went wrong in smoothness.** In
simulation the command decays to zero and the drone parks. On hardware it never stops hunting,
saturating the ±1 m/s limit for the whole flight — the drone only stays put because its inertia
and the firmware PID low-pass the thrashing.

Side finding from the same logs: the real control loop ran at **85.7 Hz**, not the 100 Hz the
simulator assumes — inference, radio and five log streams cost ~11.7 ms per step.

## 4. The experiment

The deployed policy was trained on noiseless observations and undisturbed dynamics, so it never
learned to tolerate either. `Vel-Hovering-Robust` adds three knobs (all off by default, so
`Vel-Hovering` is unchanged):

- **observation noise** — reuses `add_noise` / `noise_std`
- **external disturbances** — per-episode constant force/torque bias plus decaying intra-episode
  gusts, injected on top of the controller wrench
- **observation window** — the last K observations concatenated (`observation_space = 6·K`),
  giving the policy the temporal context to infer what it is subject to
- plus an optional action-rate penalty aimed at the chatter directly

24 policies (600 iters = 19.2k timesteps each, ~26 min each), then each flown on two targets
under one **common held-out condition** harder than most saw in training.

### Marginal effects (held-out condition)

| Factor | Level | Chatter | Position error |
|---|---|---:|---:|
| observation window K | 1 | 0.395 | 0.114 |
| | 4 | 0.203 | 0.153 |
| | 8 | **0.186** | 0.125 |
| training noise σ | 0.00 | 0.322 | 0.129 |
| | 0.02 | 0.257 | 0.118 |
| | 0.05 | **0.208** | 0.143 |
| disturbances | off | 0.270 | 0.134 |
| | on | **0.200** | 0.123 |

All three factors reduce chatter independently and compose. No policy crashed (0/24).

### Against the deployed baseline

| Policy | K | Chatter | vs baseline | Position error |
|---|---:|---:|---:|---:|
| baseline `n0.0_k1_d0_s42` — the deployed policy | 1 | 0.53925 | 1.0× | 0.0824 |
| **`n0.02_k8_d1_s42`** — deploy candidate | 8 | **0.17422** | **3.1×** | 0.0827 |
| `n0.05_k8_d1_ar_s42` | 8 | 0.15341 | 3.5× | 0.1206 |
| `n0.05_k4_d0_s42` — top by composite score | 4 | 0.08532 | 6.3× | 0.2391 |

`n0.02_k8_d1_s42` cuts chatter **3.1× at identical position accuracy** — a strict improvement.
The score's own top pick is *not* recommended: it buys smoothness by giving up 3× the position
error and never saw disturbances in training.

### The finding that matters most

Under a **clean** simulation the ranking **inverts**: K=1 shows the lowest chatter (0.0037) and
K=8 the highest (0.0099). The observation window only pays off when there is something to
filter. This is precisely the trap the original policy fell into — pristine in clean sim,
thrashing on hardware. **Clean-sim evaluation does not predict real behaviour**; candidates must
be judged under the disturbed condition.

## 5. What was tested

| Result | Check | Detail |
|---|---|---|
| PASS | Checkpoint loads on CPU | Forced `--device cpu` to mirror the MacBook deploy path; CUDA-trained weights load and act correctly |
| **CAUGHT** | Sanity gate rejected a degraded checkpoint | Final `agent_48000.pt` commanded 0.75 m/s sideways while on target → WARN. `best_agent.pt` passed. Over-training confirmed against the reward curve |
| PASS | Metrics pipeline end-to-end | Synthetic 20 s flight → metrics, plot, JSON report, 2-trial aggregate, all verified before touching real data |
| PASS | Sim evaluator emits the real schema | 1000 and 1500 rows at exactly 100 Hz; `analyze_hover.py` consumed them unmodified |
| PASS | Frame stacking dimensions | Read the trained checkpoint's first layer directly: 32×**24** = 6 obs × K=4 — not inferred from config |
| PASS | Wrong window length fails loudly | K=1 against a K=4 policy raises a state_dict size mismatch at load; it cannot silently fly a misshapen policy |
| PASS | New env reproduces the old one | With all knobs off, the sweep's baseline checkpoint is **bitwise identical** (sha256 `e169bc0a…`) to the originally deployed policy |
| PASS | W&B logging online | Verified project, run name and group reach `wandb.init` before committing 24 runs |
| PASS | Full sweep integrity | 24/24 trainings succeeded; 48 evaluation flights + 2 baseline flights, 0 crashes |

## 6. Defects found

1. **`scripts/skrl/play.py` is broken against the installed skrl 2.1.0** — two removed/changed
   APIs (`set_running_mode`; `act()` missing its `states` argument). Pre-existing and unrelated
   to this work; worked around in the new scripts rather than silently patching the repo.
   **Still open.**
2. **Randomized episode counter truncated recordings.** The env randomizes
   `episode_length_buf` on reset to de-correlate training envs; in evaluation that ended flights
   at a random point (first run cut at step 957/1000). Fixed in the evaluator.
3. **Observation size silently mis-shaped the policy.** `__post_init__` runs before Hydra
   applies `env.history_len`, so skrl reinterpreted a `[256, 24]` batch as `[1024, 6]`. Now
   computed in the env constructor. Fixed.
4. **W&B config was malformed and pointed at another account** — a stray comma in
   `wandb: True,` plus a hardcoded foreign entity that would have failed for this user. Fixed.
5. **Hydra cannot override absent keys** — run naming needed `++` (set-or-append). Caught by a
   single-run test before the overnight sweep.

## 7. Limitations

- **No hardware validation of the new policies.** Simulation says they chatter 3× less; only a
  real flight confirms it. This is the biggest open question.
- **`max_drift` was inert in the ranking** — ≈1.000 for every policy, because it only measures
  the initial approach distance from a fixed start pose. It added a constant and discriminated
  nothing, so the composite score is effectively `position error + 5 × chatter`.
- **The 5× chatter weight is a judgement call**, not a derived quantity — which is why the
  score's top pick sacrifices accuracy. Read the raw columns, not the ranking.
- **Disturbance configurations used a single seed** (only the no-disturbance grid has two), so
  those rows carry more variance than the marginals imply.
- **Sim has no motor PWM, battery, accelerometer or estimator noise**, so those columns are
  written as zero and excluded from every comparison.
- **Rate mismatch is unresolved**: 85.7 Hz real vs 100 Hz sim. It plausibly feeds the chatter
  and was corrected on neither side.
- Sim runs start from a commanded hover pose rather than the real takeoff transient, so
  steady-state metrics are the fair comparison.

## 8. Recommended next step

Fly `n0.02_k8_d1_s42` with `--history-len 8` against the same targets as the six existing
flights. That closes the loop: it either confirms the 3.1× simulated improvement on hardware, or
shows the remaining gap lives somewhere the disturbance model does not yet reach.

```bash
# offline gate first — no drone
python execution/sim2real/check_policy.py --checkpoint <ckpt> --device cpu --history-len 8

# then a short real flight
python execution/sim2real/collect_hover_vel.py --checkpoint <ckpt> --history-len 8 \
    --target 0 0 1.0 --duration 20 --tag k8_robust

# compare against the existing flights
python execution/sim2real/analyze_hover.py execution/sim2real/data/*k8_robust --json-out new.json
```

## Reproducing

```bash
python scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096 --headless --seed 42
python execution/sim2real/eval_hover_vel_sim.py --checkpoint <ckpt> --target 0 0 1.0 --duration 10 --headless
python execution/sim2real/compare_sim2real.py sim.json real.json
python execution/sim2real/experiments/sweep_train.py        # 24 runs
python execution/sim2real/experiments/eval_sweep.py --nominal
```

Raw records: `experiments/leaderboard.json`, `experiments/sweep_index.json`.
Factor analysis and sweep detail: [`experiments/RESULTS.md`](experiments/RESULTS.md).
Tooling usage and protocol mapping: [`README.md`](README.md).
