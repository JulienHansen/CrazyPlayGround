# Roadmap — Crazyflie + cflib + Lighthouse

Scope: velocity-control flights with different policy configs, then get CTBR working.
No new hardware. FPV/Jetson and the adaptive-policy work are parked.

---

## 0. What changed since your 2026-07-24 flights

You flew with commit `773c16b`. Everything since is listed below, split by whether it
affects flying.

**Affects how you fly (read these):**

| commit | date | what it means for you |
|---|---|---|
| `ab862c6` | 07-27 | `collect_hover_vel.py` and `check_policy.py` take **`--history-len K`**. Required for every policy below except the baseline. A wrong K fails at load, not in the air. |
| `78457c3` | 08-28 | same two scripts take **`--action-hist-len M`** (past commanded actions). |
| `da2cf89` | 08-28 | `collect_hover_vel.py --excite chirp\|step` = fly a known sweep with the policy OFF, to measure loop delay (OL-2). New tool: `measure_delay_noise.py`. |

**Sim-side only (does not change flying):** `d686faf` in-sim eval + gap table,
`d07dfb9` `Vel-Hovering-Robust` env + sweep harness, `6a09c19` W&B fix, `a9bcb00`
sweep-1 results, `5a84bda` report, `d907f90`/`36b0052` sweep 2, `88f9614`/`904611e`
physical DR + asymmetric critic, `c338fd5`/`b7fb970`/`5db3d34`/`2e4ed6e` DR sweep 3.

**Nothing removed or changed in the flight path you already used.** The old command
still works unchanged for the baseline policy.

---

## Phase A — velocity flight campaign (do this first)

Five policies, all velocity control, chosen to isolate one factor at a time. Metric
that matters: `cmd_smoothness` (chatter). Baseline measured **0.0690** on 07-24.

| # | policy | K | M | tests | sim chatter vs baseline |
|---|---|---:|---:|---|---|
| A0 | `n0.0_k1_d0_s42` | 1 | 0 | baseline, re-fly for same-day reference | 1.00x |
| A1 | `n0.02_k8_d1_s42` | 8 | 0 | observation window + noise + disturbance | 1.53x better |
| A2 | `m0_latoff_aroff` | 8 | 0 | same but **no latency** in training | 1.40x better |
| A3 | `m0_laton_aroff` | 8 | 0 | **latency-trained** (the sim winner) | 2.12x better |
| A4 | `m1_laton_aroff` | 8 | 1 | + one past action in the observation | 2.11x better |

A2 vs A3 is the key comparison: identical except for latency during training.

Checkpoint paths are in `experiments/flight_set.md`.

### Per policy

```bash
# 1. offline gate on the laptop, no drone. Must print [PASS].
python execution/sim2real/check_policy.py --checkpoint <CKPT> \
    --device cpu --history-len <K> --action-hist-len <M>

# 2. short flight first, hand on Ctrl-C
python execution/sim2real/collect_hover_vel.py --checkpoint <CKPT> \
    --history-len <K> --action-hist-len <M> \
    --target 0 0 1.0 --duration 8 --tag <TAG>_short

# 3. if stable, three 15 s runs for statistics
python execution/sim2real/collect_hover_vel.py --checkpoint <CKPT> \
    --history-len <K> --action-hist-len <M> \
    --target 0 0 1.0 --duration 15 --tag <TAG>_r1     # repeat r2, r3
```

### After the campaign

```bash
python execution/sim2real/analyze_hover.py execution/sim2real/data/*_r? --json-out real_new.json
python execution/sim2real/compare_sim2real.py sim.json real_new.json
```

The question being answered: **does the 2.12x simulated chatter reduction show up on
hardware?** If A3 beats A0 by ~2x on real chatter, the latency finding is confirmed and
the sim is now predictive. If it does not, the remaining gap is something the
disturbance/latency model still misses.

### Two 25-second flights worth doing while the drone is out

```bash
# measured loop delay (replaces the guessed 2-6 steps). Policy is OFF; a known chirp
# is sent instead, so the response lag is a valid delay estimate.
python execution/sim2real/collect_hover_vel.py --excite chirp --excite-axis x \
    --excite-amp 0.3 --duration 25 --tag ol2_x
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*ol2_x --open-loop

# measured estimator noise. Drone STATIONARY on the floor, Lighthouse on, motors off:
# true velocity is 0 so everything logged is estimator error.
python execution/sim2real/collect_hover_vel.py --excite step --excite-amp 0.0 \
    --duration 60 --tag static
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*static --static
```

Note on Lighthouse: it *is* the position source feeding the EKF, so there is no
independent ground truth and the `--mocap` path does not apply. The static test is the
route that works here.

---

## Phase B — CTBR on the Crazyflie

### What the API actually sends

`Commander.send_setpoint_manual(roll, pitch, yawrate, thrust_percentage, rate)`
(`cflib/crazyflie/commander.py:237`) packs

```
struct.pack('<BfffHB', TYPE_MANUAL, roll, -pitch, yawrate, thrust_16, rate)
```

| argument | type sent | unit | notes |
|---|---|---|---|
| `roll` | float32 | **deg** if `rate=False`, **deg/s** if `rate=True` | |
| `pitch` | float32 | same | **negated inside the API** — pass your own sign, do not pre-negate |
| `yawrate` | float32 | **deg/s** always | |
| `thrust_percentage` | float 0..100 | percent | converted to uint16: `10001 + 0.01*pct*49999` |
| `rate` | uint8 bool | | `True` = roll/pitch are **rates** (this is CTBR) |

So the integer that reaches the drone is `thrust_16 ∈ [10001, 60000]`, and
`pct=0 -> 10001`, `pct=100 -> 60000`. The older `send_setpoint(roll, pitch, yawrate,
thrust)` takes that **uint16 directly** (`<fffH`) and has no rate flag — with it you
must set the `flightmode.stabMode*` params instead.

### The part that bites: two different nonlinearities

1. `thrust_percentage -> uint16` is **linear** (`10001 + 499.99*pct`).
2. `uint16 PWM -> thrust in newtons` is **not linear**: thrust goes as rotor speed
   squared, the PWM-to-speed curve is nonlinear, and the whole curve shifts as the
   battery sags.

That second one is why you cannot compute hover thrust from first principles.

### A hover-thrust value from data you already have

The five 07-24 flights logged mean motor PWM at hover = **50054**. Inverting the
commander mapping:

```
hover_thrust_pct = (50054 - 10001) / 49999 * 100 = 80.1 %
```

whereas `exec_rate.py`'s built-in default is the theoretical
`100 * weight_N / max_thrust_N = 100 * 0.265 / 0.638 = 41.5 %` — **1.9x too low**, i.e.
with the default the drone will not lift. Start from `--hover-thrust-pct 80`.

Caveat: this assumes the logged `motor.mX` PWM at level hover is close to the
commanded collective PWM. It is an estimate to confirm in step B2, not a measurement.

### Steps

**B1 — bench check of conventions, props OFF.** Run `exec_rate.py` with a tiny thrust
and command a single axis at a time; log `gyro.x/y/z` while you rotate the drone by
hand. You are only checking that rate mode is engaged and the signs match, not that it
tracks. Confirm: `+roll` command corresponds to `+gyro.x`, `+pitch` to `+gyro.y`.
Also worth resolving here whether the `flightmode.stabMode*` params that `exec_rate.py`
sets are needed at all, given `rate=True` is already in the packet — they may be
redundant.

**B2 — confirm hover thrust, props ON, tethered or over a net.** Ramp thrust with zero
rate commands until it just lifts; read the pct. Expect ~80%. Update
`--hover-thrust-pct`.

**B3 — short CTBR hover.** `Rate-Hovering` needs an 18-dim observation
`[lin_vel_b(3), desired_pos_b(3), rot_mat(9), ang_vel_b(3)]`, so `exec_rate.py` must
supply attitude and body rates — it already logs gyro, and the rotation matrix comes
from the quaternion (`quat_to_rotmat_flat` is in that file). Verify the sim/real
conventions match: `max_body_rate = pi rad/s = 180 deg/s` in sim vs
`MAX_BODY_RATE_DEG = 180.0` in the script (they agree), and the thrust map
`[-1,1] -> [0.5, 1.8] x hover`.

**B4 — only then** consider adding a thrust/acceleration channel to the observation.
Sweep 3 showed mass and actuation are close to unobservable through velocity control;
CTBR with attitude and body rates is much better, and adding `acc_z` plus the commanded
thrust would make it comparable to what the adaptation literature uses. That is the
door to the adaptive-policy work, later.

---

## Parked

FPV drone, Jetson, onboard inference, RMA/asymmetric-critic training. The asymmetric
critic infrastructure is committed and working (`skrl_asym_cfg_entry_point`) whenever
you come back to it.
