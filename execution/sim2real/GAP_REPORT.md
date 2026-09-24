# Sim-to-real gap, velocity hover controller

Measured on 2026-09-24. Four policies, flown on a Crazyflie 2.1 with Lighthouse
positioning, then re-flown in the nominal simulator with the same frozen goal at
(0, 0, 1) m.

Reproduce with:

```bash
python execution/sim2real/run_sim_gap.py --headless
python execution/sim2real/report_gap.py --skip-s 3 --markdown gap_steady.md
```

## Policies

| id | training condition | K | M |
|---|---|---|---|
| A0 | baseline, no randomisation | 1 | 0 |
| A1 | observation noise + disturbance, frame stacking | 8 | 0 |
| A3 | as A1, plus randomised 2-6 step loop latency | 8 | 0 |
| A4 | as A3, plus one past action in the observation | 8 | 1 |

A2 was trained but never flown, so it has no gap.

## Protocol coverage

| item | status |
|---|---|
| CL-1 hover gap | measured, below |
| CL-2 rank correlation | measured, below |
| OL-2 loop delay | instrumented, not yet flown open-loop |
| OL-4 state estimation | Kalman variance logged, not yet analysed |
| OL-1 replay, OL-3 actuator | not started |

## CL-1, steady state

The first 3 s of every flight are dropped. The approach is not the hover being
measured and it is not matched between the two sides: the simulator starts 0.50 m
below the goal, the drone took off to 0.36 m under a 1.00 m goal.

| policy | position error sim → real | command chatter sim → real [m/s²] | control rate |
|---|---|---|---|
| A0 | 0.017 → 0.073 m (4.2x) | 0.175 → 5.52 (31.7x) | 100 → 84.0 Hz |
| A1 | 0.029 → 0.058 m (2.0x) | 0.006 → 6.32 (sim≈0) | 100 → 85.0 Hz |
| A3 | 0.030 → **0.027** m (0.88x) | 0.002 → **2.48** (sim≈0) | 100 → 84.1 Hz |
| A4 | 0.069 → 0.065 m (0.94x) | 0.034 → 2.58 (77x) | 100 → 84.1 Hz |

## CL-1, whole flight

| policy | position error sim → real | command chatter sim → real [m/s²] |
|---|---|---|
| A0 | 0.036 → 0.129 m (3.6x) | 0.505 → 5.06 (10.0x) |
| A1 | 0.046 → 0.127 m (2.7x) | 0.724 → 6.08 (8.4x) |
| A3 | 0.051 → 0.098 m (1.9x) | 0.445 → 3.16 (7.1x) |
| A4 | 0.084 → 0.133 m (1.6x) | 0.499 → **2.37** (4.7x) |

## CL-2, does the ranking transfer?

| metric | SRCC | concordant pairs | sim ranking | real ranking |
|---|---|---|---|---|
| position error | +0.20 | 3/6 | A0 < A1 < A3 < A4 | A3 < A1 < A0 < A4 |
| command chatter | **+0.80** | **5/6** | A3 < A4 < A0 < A1 | A4 < A3 < A0 < A1 |

## Findings

**The gap is an excitation gap, not a tracking gap.** Once the approach is
removed, position error transfers almost unchanged, and for the two
latency-trained policies it closes entirely: A3 hovers *better* on the drone than
in simulation (0.027 m against 0.030 m). What does not transfer is motion. In the
nominal simulator a frame-stacked policy converges to a fixed point and its
steady-state chatter falls to ~1e-5 m/s². The drone never stops moving. The
simulator is simply too quiet to contain the behaviour that dominates hardware.

**Latency-aware training halves hardware chatter.** A3 and A4 command 2.48 and
2.58 m/s² against 5.52 and 6.32 for A0 and A1. The two policies trained with a
randomised 2-6 step dead time are the two quietest on hardware, and the ordering
is the one simulation predicted.

**Frame stacking alone does not help.** A1 differs from A0 only by the observation
window and the training noise, and it is marginally *worse* on hardware for both
chatter and max drift. The sweep predicted this; hardware confirms it.

**The simulator ranks smoothness well and accuracy badly.** Chatter ranking
transfers with 5 of 6 pairs ordered correctly. Position-error ranking does not
(3 of 6, and it inverts on the steady-state window). Training selection on
simulated chatter is therefore justified; selection on simulated position error
is not.

**The deployment loop ran at 84 Hz, not the commanded 100.** Every flight shows
it. The cause was the control loop's relative sleep, not the radio, and it is
fixed; these numbers were all collected before the fix.

## Caveats

- Two simulator trials and two to four hardware flights per policy. The rank
  correlation is over four policies. Read every ratio as a direction.
- Hardware flights ran at 84 Hz while the simulator ran at 100. Chatter is
  reported per second for that reason; the per-step column is kept for reference
  but should not be compared across the two.
- Hardware runs came from one session, one airframe and one battery state.
  Between-session variation is unmeasured.
- The simulator is nominal. Every randomisation used during training is off, so
  this measures the gap to the model, not robustness.
