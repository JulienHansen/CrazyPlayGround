# Sim-to-real gap, velocity hover controller

Measured on 2026-09-24. Four policies, flown on a Crazyflie 2.1 with Lighthouse
positioning, then re-flown in the nominal simulator under the **same two trial
conditions** as the hardware.

```bash
python execution/sim2real/run_sim_gap.py --headless
python execution/sim2real/report_gap.py --skip-s 3      # steady-state segment
python execution/sim2real/report_gap.py --skip-s 0      # whole flight
```

## Policies

| id | training condition | K | M |
|---|---|---|---|
| A0 | baseline, no randomisation | 1 | 0 |
| A1 | observation noise + disturbance, observation window | 8 | 0 |
| A3 | as A1, plus per-episode delay of 2-6 steps | 8 | 0 |
| A4 | as A3, plus one past action | 8 | 1 |

A2 was trained as a sweep-2 reference but never flown, so it has no gap.

## Trial conditions

Both sides fly the same two trials. Matching them matters: an earlier version of
this measurement flew the simulator at (0,0,1) for both trials while the hardware
long trial used (1,1,1), which charged the drone for a 1.41 m traverse the
simulator never made.

| trial | goal | duration |
|---|---|---|
| short | (0, 0, 1) | 10 s |
| long | (1, 1, 1) | 15 s |

## Protocol coverage

| item | status |
|---|---|
| CL-1 hover gap | measured, below |
| CL-2 rank correlation | measured, below |
| OL-2 loop delay | instrumented, not yet flown open-loop |
| OL-4 state estimation | Kalman variance logged, not yet analysed |
| OL-1 replay, OL-3 actuator | not started |

## CL-1, steady-state segment

First 3 s dropped. The approach is not the hover being measured and is not matched:
the simulator starts 0.50 m below the goal, the drone took off to 0.36 m under a
1.00 m goal.

| policy | position error sim → hw | chatter sim → hw [m/s²] | loop rate |
|---|---|---|---|
| A0 | 0.017 → 0.073 m | 0.257 → 5.52 | 100 → 84.0 Hz |
| A1 | 0.029 → 0.058 m | 0.031 → 6.32 | 100 → 85.0 Hz |
| A3 | 0.030 → **0.027** m | 0.008 → **2.47** | 100 → 84.1 Hz |
| A4 | 0.069 → 0.065 m | 0.039 → 2.58 | 100 → 84.1 Hz |

## Single-factor contrasts, hardware only

Split by trial condition, because the policies have unequal short/long mixes.

| policy | chatter long / short | position error long / short |
|---|---|---|
| A0 | 6.36 / 5.25 | 0.063 / 0.077 |
| A1 | 5.82 / 6.81 | 0.061 / 0.055 |
| A3 | **2.55 / 2.40** | **0.026 / 0.027** |
| A4 | 2.92 / 2.25 | 0.063 / 0.067 |

- **A1 → A3** isolates the delay. Both conditions agree: ~2.5x less chatter, ~2.2x
  better position error.
- **A3 → A4** isolates the action history. Chatter unchanged, position error 2.4x
  worse.
- **A0 → A1** varies window, noise and disturbance together, so it isolates nothing.
  No consistent effect either way.

## CL-2, does the ranking transfer?

| metric | window | SRCC | concordant pairs |
|---|---|---|---|
| chatter | whole flight | **+1.00** | **6/6** |
| chatter | steady state | +0.40 | 4/6 |
| position error | whole flight | +0.20 | 3/6 |
| position error | steady state | −0.40 | 2/6 |

## Findings

**The gap is an excitation gap, not a tracking gap.** With the approach removed,
position error transfers almost unchanged, and for both delay-trained policies it
closes entirely: A3 hovers better on the drone (0.027 m) than in simulation
(0.030 m). What does not transfer is chatter. In the nominal simulator a policy with
an observation window settles almost completely — A3 reaches 0.008 m/s², ~300x below
its hardware value — while the drone never stops moving.

**Delay-aware training more than halves hardware chatter** and improves accuracy.
This is the cleanest single-factor result: A1 → A3 changes only the delay.

**Action history does not help and costs accuracy** — 2.4x worse position error on
both trial conditions.

**The observation window has no detectable effect on hardware**, though it roughly
halves chatter in simulation.

**The inner loop absorbs most of the chatter.** The cascade PID attenuates commanded
oscillation by 2.14-2.32x, identically for every policy. What differs is what the
policy asks for: the delay-trained policies ask for ~4x less. This is why accuracy
survives a large chatter gap.

**Simulated chatter ranks policies perfectly over the whole flight and barely at all
in steady state**, where the simulated values collapse toward zero. Select on
simulated chatter scored over the whole flight; do not select on simulated position
error.

**The deployment loop ran at 84.4 Hz, not the commanded 100.** Median period
11.96 ms. Caused by the control loop's relative sleep, not the radio; fixed since,
but every measurement here predates the fix.

## Caveats

- Two simulator trials and two to four hardware flights per policy. Rank
  correlations over four policies. Read every ratio as a direction.
- Hardware ran at ~84 Hz, the simulator at 100. Chatter is reported per second for
  that reason.
- One session, one airframe, one battery state. Between-session variation unmeasured.
- The simulator is nominal: every randomisation used in training is off, so this
  measures the gap to the model, not robustness.
