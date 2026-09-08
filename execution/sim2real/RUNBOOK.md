# Hardware runbook — run this on the Mac

Crazyflie 2.1 + Crazyradio PA + Lighthouse. Velocity control. Everything you need is
in the repo; the five policy checkpoints are committed under
`execution/sim2real/flight_checkpoints/` (279 KB total), so no scp is required.

---

## 0. Setup on the Mac (once)

```bash
git clone git@github.com:JulienHansen/CrazyPlayGround.git      # or: cd existing clone
cd CrazyPlayGround
git fetch origin
git checkout feature/sim2real-hover-velocity
git pull

python3 -m venv ~/cf-deploy && source ~/cf-deploy/bin/activate
pip install "cflib>=0.1.32" "skrl>=2.1.0" torch numpy matplotlib
brew install libusb          # Crazyradio needs it on macOS
```

Before every session: **close cfclient** (it holds the radio), fresh battery, clean
props, Lighthouse base stations powered, and let the drone sit still ~5 s so the EKF
converges. The scripts abort on stale position or Kalman variance > 0.5.

Re-pull later with just `git pull` — checkpoints come with it.

---

## 1. Offline gate — no drone, do all five now

Each must print `[PASS]`. This catches a wrong `--history-len` on the ground rather
than in the air (a mismatch raises a state_dict size error at load).

```bash
cd CrazyPlayGround
C=execution/sim2real/flight_checkpoints

python execution/sim2real/check_policy.py --checkpoint $C/A0_baseline_k1_m0.pt        --device cpu --history-len 1 --action-hist-len 0
python execution/sim2real/check_policy.py --checkpoint $C/A1_noise_disturb_k8_m0.pt   --device cpu --history-len 8 --action-hist-len 0
python execution/sim2real/check_policy.py --checkpoint $C/A2_no_latency_k8_m0.pt      --device cpu --history-len 8 --action-hist-len 0
python execution/sim2real/check_policy.py --checkpoint $C/A3_latency_trained_k8_m0.pt --device cpu --history-len 8 --action-hist-len 0
python execution/sim2real/check_policy.py --checkpoint $C/A4_latency_acthist_k8_m1.pt --device cpu --history-len 8 --action-hist-len 1
```

A `[WARN]` here is not automatically a blocker — it means the policy commands > 0.3 m/s
while sitting exactly on target in a *static* probe, which is out-of-distribution for a
window policy. Note which ones warn, fly them last, and watch altitude.

---

## 2. Experiment list

Fly in this order. `--duration 8` first for every policy, then three 15 s runs only if
the short one was stable. Keep a hand on Ctrl-C — it triggers a safe land.

### E0 — baseline re-fly (same-day reference)

The policy you already flew on 07-24. Re-flying it on the same battery/room as the
others is what makes the comparison valid.

```bash
C=execution/sim2real/flight_checkpoints
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A0_baseline_k1_m0.pt \
    --history-len 1 --action-hist-len 0 --target 0 0 1.0 --duration 8  --tag A0_short
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A0_baseline_k1_m0.pt \
    --history-len 1 --action-hist-len 0 --target 0 0 1.0 --duration 15 --tag A0_r1
# repeat with --tag A0_r2 and --tag A0_r3
```

### E1 — observation window + noise + disturbance

```bash
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A1_noise_disturb_k8_m0.pt \
    --history-len 8 --action-hist-len 0 --target 0 0 1.0 --duration 8  --tag A1_short
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A1_noise_disturb_k8_m0.pt \
    --history-len 8 --action-hist-len 0 --target 0 0 1.0 --duration 15 --tag A1_r1   # r2, r3
```

### E2 / E3 — the key pair: identical except training latency

These two differ **only** in whether loop latency was simulated during training. Fly
them back to back, same battery level if you can.

```bash
# E2: no latency in training
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A2_no_latency_k8_m0.pt \
    --history-len 8 --action-hist-len 0 --target 0 0 1.0 --duration 8  --tag A2_short
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A2_no_latency_k8_m0.pt \
    --history-len 8 --action-hist-len 0 --target 0 0 1.0 --duration 15 --tag A2_r1   # r2, r3

# E3: latency-trained (predicted best: 2.12x less chatter than baseline)
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A3_latency_trained_k8_m0.pt \
    --history-len 8 --action-hist-len 0 --target 0 0 1.0 --duration 8  --tag A3_short
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A3_latency_trained_k8_m0.pt \
    --history-len 8 --action-hist-len 0 --target 0 0 1.0 --duration 15 --tag A3_r1   # r2, r3
```

### E4 — + one past action in the observation

```bash
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A4_latency_acthist_k8_m1.pt \
    --history-len 8 --action-hist-len 1 --target 0 0 1.0 --duration 8  --tag A4_short
python execution/sim2real/collect_hover_vel.py --checkpoint $C/A4_latency_acthist_k8_m1.pt \
    --history-len 8 --action-hist-len 1 --target 0 0 1.0 --duration 15 --tag A4_r1   # r2, r3
```

### E5 — measured loop delay (25 s, policy OFF)

A known chirp is sent instead of the policy, so the command is exogenous and the
response lag is a valid delay estimate. Closed-loop policy flights cannot give this.

```bash
python execution/sim2real/collect_hover_vel.py --excite chirp --excite-axis x \
    --excite-amp 0.3 --excite-f0 0.2 --excite-f1 4.0 --duration 25 --tag E5_chirp_x
# optional, other axes:
python execution/sim2real/collect_hover_vel.py --excite chirp --excite-axis y --excite-amp 0.3 --duration 25 --tag E5_chirp_y
python execution/sim2real/collect_hover_vel.py --excite chirp --excite-axis z --excite-amp 0.2 --duration 25 --tag E5_chirp_z
```

### E6 — measured estimator noise (60 s, drone stationary on the floor, motors off)

True velocity is 0 and position is constant, so everything logged is estimator error.
This is the OL-4 substitute for Lighthouse, which cannot be differenced against itself.

```bash
python execution/sim2real/collect_hover_vel.py --excite step --excite-amp 0.0 \
    --duration 60 --tag E6_static
```

---

## 3. Analysis (Mac or Linux box)

```bash
# per-policy metrics + plots
python execution/sim2real/analyze_hover.py execution/sim2real/data/*A0_r? --json-out A0.json
python execution/sim2real/analyze_hover.py execution/sim2real/data/*A3_r? --json-out A3.json --plot

# all policies side by side
python execution/sim2real/analyze_hover.py execution/sim2real/data/*_r? --json-out real_all.json

# measured delay and noise -> the numbers to put back into training
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*E5_chirp_x --open-loop
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*E6_static  --static
```

### What to look at

The metric is **`cmd_smoothness_mps_per_step`** (chatter). Reference from 07-24:
**0.0690** for the baseline.

| outcome | reading |
|---|---|
| A3 chatter ≈ 0.03 or below | the 2.12x sim gain transferred; latency was the gap |
| A3 ≈ A2 | training latency did not help on hardware; the sim latency model is wrong |
| all ≈ 0.069 | the gap is something neither latency nor noise captures |

Also watch `mean_pos_err_m` (baseline 0.097) — a policy that smooths by tracking worse
is not a win — and `crashed` / `min_z_after_grace_m`.

`measure_delay_noise.py` on E5/E6 prints the `env.action_latency_steps` and
`env.noise_std` / `env.noise_corr` to use in the next training round. My current values
are guesses: latency bracketed at 2-6 steps, noise 0.02 white, while the 07-24 flights
suggest noise nearer 0.007 with AR(1) ~0.26.

---

## 4. If something goes wrong

| symptom | fix |
|---|---|
| `Connection timeout` | cfclient still open, or wrong URI (`--uri radio://0/80/2M/E7E7E7E7E8`) |
| `size mismatch` at load | wrong `--history-len` / `--action-hist-len` for that checkpoint |
| immediate emergency land | Kalman variance > 0.5 -> Lighthouse tracking poor; check base stations |
| `Position data stale` | radio dropouts; try `--log-rate-hz 50` |
| drifts / climbs steadily | land. Note which policy; the static gate flagged some for a residual climb command |
