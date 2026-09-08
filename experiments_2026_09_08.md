# Crazyflie sim-to-real flight experiments — 2026-09-08

**Hardware:** Crazyflie 2.1, Crazyradio PA, Lighthouse positioning, MacBook running the
policy off-board over cflib.
**Scope:** velocity control only (`send_velocity_world_setpoint`). CTBR is a separate
follow-up, see the end.
**Repo:** `CrazyPlayGround`, branch `feature/sim2real-hover-velocity`.

This document is self-contained: it can be handed to an assistant or a colleague to run
the campaign. Everything needed is in the repo after a `git pull`, including the policy
checkpoints.

---

## Background: what we already know

A velocity hover policy was trained and flown on 2026-07-24 (5 successful flights). The
gap was measured by running the *same* policy in simulation on the same targets:

| metric | sim | real | ratio |
|---|---:|---:|---|
| mean position error | 0.068 m | 0.097 m | 0.70 |
| **command chatter** | **0.0065** | **0.0690** | **10.6x worse on hardware** |
| mean speed while hovering | 0.094 m/s | 0.342 m/s | 3.6x |

**Position tracking transfers; control smoothness does not.** In simulation the
commanded velocity decays to zero and the drone parks. On hardware it oscillates at
1.4-1.7 Hz, saturating +/-1 m/s for the whole flight, and only stays in place because
inertia and the firmware PID low-pass the thrashing.

A simulation study then showed **unmodelled loop latency reproduces ~75% of that gap**
and, importantly, its *signature* — chatter explodes while position error barely moves:

| simulated loop delay | 0 | 1 (10 ms) | 2 (20 ms) | 3 (30 ms) | 4 (40 ms) | real |
|---|---|---|---|---|---|---|
| chatter | 0.0060 | 0.0147 | 0.0300 | 0.0422 | 0.0517 | **0.0690** |
| position error | 0.039 | 0.040 | 0.043 | 0.049 | 0.056 | 0.097 |

Noise and disturbance randomization never explained that asymmetry. Latency does.

**So the campaign tests one hypothesis:** *policies trained with simulated loop latency
should chatter substantially less on the real drone.* Simulation predicts a 2.1x
reduction. That has never been checked on hardware — every number above except the
2026-07-24 column is simulated.

---

## Setup on the Mac (once)

```bash
git clone git@github.com:JulienHansen/CrazyPlayGround.git   # or cd an existing clone
cd CrazyPlayGround
git fetch origin
git checkout feature/sim2real-hover-velocity
git pull

python3 -m venv ~/cf-deploy && source ~/cf-deploy/bin/activate
pip install "cflib>=0.1.32" "skrl>=2.1.0" torch numpy matplotlib
brew install libusb        # Crazyradio needs libusb on macOS
```

The five policy checkpoints are committed in `execution/sim2real/flight_checkpoints/`
(279 KB) — no scp needed. `manifest.json` there lists each one's `K`/`M` and sha256.

**Before every session:** close cfclient (it holds the radio), fresh battery, clean
props, Lighthouse base stations on, and let the drone sit still ~5 s so the EKF
converges. Clear ~2x2 m. The scripts emergency-land on stale position, Kalman variance
> 0.5, `z < 0.1 m` after a 3 s grace, or `z > 2.5 m`. Ctrl-C triggers a safe land.

---

## The policies

All are velocity-control policies for the same task; they differ only in how they were
trained. `K` = observation window length (stacked frames), `M` = number of past
commanded actions in the observation.

| id | file | K | M | what it tests | predicted vs baseline |
|---|---|---:|---:|---|---|
| A0 | `A0_baseline_k1_m0.pt` | 1 | 0 | the policy already flown on 07-24 | 1.00x (reference) |
| A1 | `A1_noise_disturb_k8_m0.pt` | 8 | 0 | observation window + noise + disturbance | 1.53x less chatter |
| A2 | `A2_no_latency_k8_m0.pt` | 8 | 0 | same, **no latency** during training | 1.40x |
| A3 | `A3_latency_trained_k8_m0.pt` | 8 | 0 | **latency-trained** — the simulation winner | **2.12x** |
| A4 | `A4_latency_acthist_k8_m1.pt` | 8 | 1 | + one past action in the observation | 2.11x |

**A2 vs A3 is the experiment.** They are matched in every respect except that A3 saw a
randomized 2-6 step (20-60 ms) command delay during training. Fly them back to back at
similar battery level.

`--history-len` and `--action-hist-len` **must** match the table. A mismatch raises a
`state_dict` size error at load, so it fails on the ground rather than in the air.

---

## Step 1 — offline gate, no drone

Run all five. Each should print `[PASS]`.

```bash
cd CrazyPlayGround
C=execution/sim2real/flight_checkpoints

python execution/sim2real/check_policy.py --checkpoint $C/A0_baseline_k1_m0.pt        --device cpu --history-len 1 --action-hist-len 0
python execution/sim2real/check_policy.py --checkpoint $C/A1_noise_disturb_k8_m0.pt   --device cpu --history-len 8 --action-hist-len 0
python execution/sim2real/check_policy.py --checkpoint $C/A2_no_latency_k8_m0.pt      --device cpu --history-len 8 --action-hist-len 0
python execution/sim2real/check_policy.py --checkpoint $C/A3_latency_trained_k8_m0.pt --device cpu --history-len 8 --action-hist-len 0
python execution/sim2real/check_policy.py --checkpoint $C/A4_latency_acthist_k8_m1.pt --device cpu --history-len 8 --action-hist-len 1
```

A `[WARN]` is not automatically disqualifying: it means the policy commands more than
0.3 m/s while sitting exactly on target in a *static* probe, which is out of
distribution for a window policy (it is fed K identical frames, i.e. "perfectly steady
forever", which never happens in flight). Record which ones warn, fly those last, and
watch altitude on the short flight.

---

## Step 2 — flights E0..E4

For every policy: **one 8 s flight first**, and only if it was stable, three 15 s runs.
Three runs give enough spread to compare against the 07-24 statistics.

```bash
C=execution/sim2real/flight_checkpoints
COLLECT="python execution/sim2real/collect_hover_vel.py"

# ---- E0  baseline re-fly (same-day reference; makes the comparison valid) ----
$COLLECT --checkpoint $C/A0_baseline_k1_m0.pt --history-len 1 --action-hist-len 0 \
    --target 0 0 1.0 --duration 8  --tag A0_short
$COLLECT --checkpoint $C/A0_baseline_k1_m0.pt --history-len 1 --action-hist-len 0 \
    --target 0 0 1.0 --duration 15 --tag A0_r1      # then A0_r2, A0_r3

# ---- E1  window + noise + disturbance ----
$COLLECT --checkpoint $C/A1_noise_disturb_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 8  --tag A1_short
$COLLECT --checkpoint $C/A1_noise_disturb_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 15 --tag A1_r1      # then A1_r2, A1_r3

# ---- E2  no latency in training ----
$COLLECT --checkpoint $C/A2_no_latency_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 8  --tag A2_short
$COLLECT --checkpoint $C/A2_no_latency_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 15 --tag A2_r1      # then A2_r2, A2_r3

# ---- E3  latency-trained  (the prediction under test) ----
$COLLECT --checkpoint $C/A3_latency_trained_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 8  --tag A3_short
$COLLECT --checkpoint $C/A3_latency_trained_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 15 --tag A3_r1      # then A3_r2, A3_r3

# ---- E4  + one past action ----
$COLLECT --checkpoint $C/A4_latency_acthist_k8_m1.pt --history-len 8 --action-hist-len 1 \
    --target 0 0 1.0 --duration 8  --tag A4_short
$COLLECT --checkpoint $C/A4_latency_acthist_k8_m1.pt --history-len 8 --action-hist-len 1 \
    --target 0 0 1.0 --duration 15 --tag A4_r1      # then A4_r2, A4_r3
```

Each run writes `execution/sim2real/data/<timestamp>_vel_<tag>/` containing
`flight.csv` (one row per 100 Hz control step), `flight.npz`, and `metadata.json` with
the checkpoint hash, git commit, rates and frame conventions.

Optional, if time allows: repeat E0 and E3 at `--target 1 1 1.0` to check the result is
not specific to a hover directly above the takeoff point.

---

## Step 3 — two measurement flights (25 s and 60 s)

These are worth more than any additional policy flight, because they replace guessed
simulation parameters with measured ones.

### E5 — loop delay (policy OFF)

The policy is replaced by a known frequency sweep, so the command does not depend on the
measured state and the response lag is a valid delay estimate. **This cannot be obtained
from closed-loop policy flights**: there the policy reacts to the state it caused, and
the apparent lag collapses onto the oscillation geometry (on the 07-24 data the fit
returned ~150 ms, which is just a quarter period of the 1.5 Hz limit cycle).

```bash
$COLLECT --excite chirp --excite-axis x --excite-amp 0.3 --excite-f0 0.2 --excite-f1 4.0 \
    --duration 25 --tag E5_chirp_x
# optional other axes
$COLLECT --excite chirp --excite-axis y --excite-amp 0.3 --duration 25 --tag E5_chirp_y
$COLLECT --excite chirp --excite-axis z --excite-amp 0.2 --duration 25 --tag E5_chirp_z
```

The drone hovers and oscillates back and forth along one axis. Keep the space clear.

### E6 — estimator noise (drone stationary on the floor, motors off)

True velocity is 0 and position is constant, so everything logged **is** estimator
error. This substitutes for the usual motion-capture comparison, which is impossible
here: Lighthouse *is* the position source feeding the EKF, so there is no independent
ground truth to difference against.

```bash
$COLLECT --excite step --excite-amp 0.0 --duration 60 --tag E6_static
```

Optional second variant: same, but with the drone held down / secured and motors at
hover thrust, to capture vibration coupling that the static test misses.

---

## Step 4 — analysis

```bash
# per policy
python execution/sim2real/analyze_hover.py execution/sim2real/data/*A0_r? --json-out A0.json
python execution/sim2real/analyze_hover.py execution/sim2real/data/*A3_r? --json-out A3.json --plot

# all together
python execution/sim2real/analyze_hover.py execution/sim2real/data/*_r? --json-out real_all.json

# measured delay and noise -> parameters for the next training round
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*E5_chirp_x --open-loop
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*E6_static  --static
```

### The metric that matters

**`cmd_smoothness_mps_per_step`** — mean per-step change of the commanded velocity.
Reference from 07-24: **0.0690** for A0.

| observation | interpretation |
|---|---|
| A3 chatter ~0.03 or lower, position error not worse | the 2.12x simulated gain transferred; latency was the gap and the simulator is now predictive |
| A3 ~= A2 | training latency did not help on hardware; the simulation latency model is wrong and needs the E5 measurement |
| all ~0.069 | the gap is something neither latency nor noise captures — look at the thrust curve and the 85 Hz vs 100 Hz rate mismatch |
| A4 clearly better than A3 | action history matters more on hardware than in simulation |

Also record `mean_pos_err_m` (A0 was 0.097) — a policy that smooths by tracking worse is
not an improvement — plus `crashed`, `min_z_after_grace_m` and `effective_rate_hz`
(07-24 gave 84-86 Hz, not the 100 Hz the simulator assumes).

---

## Known issues and gotchas

| symptom | cause / fix |
|---|---|
| `Connection timeout after 10s` | cfclient still open, or wrong URI. Default `radio://0/80/2M/E7E7E7E7E8`, override with `--uri` |
| `size mismatch for net_container.0.weight` | wrong `--history-len` / `--action-hist-len` for that checkpoint (this is the intended safe failure) |
| emergency land immediately | Kalman variance > 0.5 -> Lighthouse tracking poor. Check base stations, lighting, and that the deck is seated |
| `Position data stale` | radio dropouts. Retry with `--log-rate-hz 50` to halve telemetry load |
| steady climb or drift | land. Note the policy; the static gate flags some for a residual climb command at target |
| effective rate ~85 Hz | expected, not a fault: inference + radio + five log streams cost ~11.7 ms per step. Worth recording per flight |

---

## Follow-up: CTBR (not part of this campaign)

For later. The API details, since they are the usual source of confusion:

`Commander.send_setpoint_manual(roll, pitch, yawrate, thrust_percentage, rate)` packs
`struct.pack('<BfffHB', TYPE_MANUAL, roll, -pitch, yawrate, thrust_16, rate)`
(`cflib/crazyflie/commander.py:237`).

- `roll`/`pitch`: float32, **deg/s when `rate=True`** (that flag is what makes it CTBR),
  degrees when `False`. **`pitch` is negated inside the API** — do not pre-negate.
- `yawrate`: float32, deg/s always.
- `thrust_percentage`: float 0..100, converted to the integer the drone receives:
  `thrust_16 = 10001 + 0.01 * pct * 49999`, so 0% -> 10001 and 100% -> 60000.

Two separate nonlinearities: percent -> PWM is **linear**, but PWM -> newtons is **not**
(thrust goes as rotor speed squared, the PWM-to-speed curve is nonlinear, and it shifts
with battery sag). That is why hover thrust cannot be computed from first principles.

Useful starting value: the 07-24 flights logged mean hover motor PWM = 50054, which
inverts to **hover_thrust_pct ~= 80%**. `exec_rate.py`'s built-in theoretical default is
`100 * weight_N / max_thrust_N = 41.5%`, **1.9x too low — the drone would not lift**.
Start from `--hover-thrust-pct 80` and confirm on the bench. (This assumes logged
`motor.mX` at level hover approximates the commanded collective; verify it.)

Bring-up order: props off, check sign conventions against `gyro.*`; then confirm hover
percent tethered; then a short CTBR hover with `Rate-Hovering` (18-dim observation
`[lin_vel_b(3), desired_pos_b(3), rot_mat(9), ang_vel_b(3)]`, `max_body_rate` = 180 deg/s
in both simulator and script).

---

## Honest status of the predictions in this document

Everything except the 2026-07-24 column is **simulation**. The 2.12x figure for A3 comes
from evaluating in a simulated condition (40 ms latency + correlated noise +
disturbance) that was itself calibrated to reproduce the observed hardware chatter — so
it is a self-consistent prediction, not an independent one. The simulated latency is
*bracketed* (2-6 steps) rather than measured, which is what E5 fixes. The training
noise (0.05 white) is likely too large and of the wrong character: the 07-24 flights
suggest nearer 0.007 with AR(1) ~0.26, which E6 will settle.

A separate simulation study also found that randomizing mass/inertia/thrust bought no
generalization in velocity control, with the likely reason being structural: through a
velocity setpoint absorbed by the firmware cascade PID, and with an observation of only
`[lin_vel_b, desired_pos_b]`, mass and actuation are close to unobservable. That is a
reason to expect this campaign to say something about *latency and smoothness*, and not
to expect it to demonstrate adaptation.
