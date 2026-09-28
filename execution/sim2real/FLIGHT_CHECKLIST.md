# Flight checklist — run this, top to bottom

Single source of truth for the hardware session. Crazyflie 2.1 + Crazyradio + Lighthouse,
policy running on the Mac over cflib.

`README.md` in this directory documents the pipeline itself. If the two disagree
about a command, this file wins.

Question being answered: **does training with simulated loop latency reduce chatter on
the real drone?** Baseline chatter measured 2026-07-24 = **0.0690**.

---

## 0. Setup (once per machine)

- [ ] Python **3.11** — the macOS system 3.9.6 cannot work (cflib and skrl both need ≥3.10)

```bash
brew install python@3.11 libusb
rm -rf ~/cf-deploy
/opt/homebrew/bin/python3.11 -m venv ~/cf-deploy     # Intel Mac: /usr/local/bin/python3.11
source ~/cf-deploy/bin/activate
python --version                                      # must print 3.11.x
pip install --upgrade pip
pip install "cflib>=0.1.32" "skrl>=2.1.0" torch numpy matplotlib
```

- [ ] Repo up to date

```bash
cd CrazyPlayGround
git fetch origin && git checkout feature/sim2real-hover-velocity && git pull
```

Checkpoints ship with the repo in `execution/sim2real/flight_checkpoints/` (279 KB).

---

## 1. Before every session

- [ ] **cfclient closed** (it holds the radio)
- [ ] Fresh battery, props clean, Lighthouse deck seated
- [ ] Base stations powered, drone sitting still ~5 s so the EKF converges
- [ ] ~2 x 2 m clear
- [ ] Ctrl-C = safe land. Scripts also auto-land on stale position, kalman variance > 0.5,
      z < 0.1 m after 3 s, or z > 2.5 m

```bash
cd CrazyPlayGround && source ~/cf-deploy/bin/activate
export C=execution/sim2real/flight_checkpoints
export COLLECT="python execution/sim2real/collect_hover_vel.py"
```

---

## 2. TEST 0 — offline gate (no drone)

Each must print `[PASS]`. Catches a wrong `--history-len` on the ground instead of in the air.

- [ ] A0
```bash
python execution/sim2real/check_policy.py --checkpoint $C/A0_baseline_k1_m0.pt        --device cpu --history-len 1 --action-hist-len 0
```
- [ ] A2
```bash
python execution/sim2real/check_policy.py --checkpoint $C/A2_no_latency_k8_m0.pt      --device cpu --history-len 8 --action-hist-len 0
```
- [ ] A3
```bash
python execution/sim2real/check_policy.py --checkpoint $C/A3_latency_trained_k8_m0.pt --device cpu --history-len 8 --action-hist-len 0
```

`[WARN]` is not a hard stop — it means the policy commands >0.3 m/s while sitting exactly
on target in a *static* probe, which is out of distribution for a window policy. Note which
ones warn, fly them last, watch altitude.

Optional extras: `A1_noise_disturb_k8_m0.pt` (K=8, M=0), `A4_latency_acthist_k8_m1.pt` (K=8, **M=1**).

---

## 3. TEST 1 — A0 baseline (your current policy)

Same-day reference. Without it the comparison is not valid.

- [ ] short
```bash
$COLLECT --checkpoint $C/A0_baseline_k1_m0.pt --history-len 1 --action-hist-len 0 \
    --target 0 0 1.0 --duration 8 --tag A0_short
```
- [ ] r1  - [ ] r2  - [ ] r3
```bash
$COLLECT --checkpoint $C/A0_baseline_k1_m0.pt --history-len 1 --action-hist-len 0 \
    --target 0 0 1.0 --duration 15 --tag A0_r1        # then A0_r2, A0_r3
```

## 4. TEST 2 — A3 latency-trained (the prediction under test)

- [ ] short
```bash
$COLLECT --checkpoint $C/A3_latency_trained_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 8 --tag A3_short
```
- [ ] r1  - [ ] r2  - [ ] r3
```bash
$COLLECT --checkpoint $C/A3_latency_trained_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 15 --tag A3_r1        # then A3_r2, A3_r3
```

## 5. TEST 3 — A2 control (identical to A3, no training latency)

Without this you cannot tell whether an improvement came from latency training or just
from the observation window.

- [ ] short
```bash
$COLLECT --checkpoint $C/A2_no_latency_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 8 --tag A2_short
```
- [ ] r1  - [ ] r2  - [ ] r3
```bash
$COLLECT --checkpoint $C/A2_no_latency_k8_m0.pt --history-len 8 --action-hist-len 0 \
    --target 0 0 1.0 --duration 15 --tag A2_r1        # then A2_r2, A2_r3
```

---

## 6. TEST 4 — measured loop delay (25 s, policy OFF)

A known chirp replaces the policy, so the command does not depend on the measured state
and the lag is a valid delay estimate. **This cannot be obtained from the policy flights** —
in closed loop the estimate collapses onto the limit-cycle geometry.

- [ ] chirp x
```bash
$COLLECT --excite chirp --excite-axis x --excite-amp 0.3 --excite-f0 0.2 --excite-f1 4.0 \
    --duration 25 --tag E5_chirp_x
```
- [ ] optional: `--excite-axis y` (amp 0.3), `--excite-axis z` (amp 0.2)

The drone hovers and sweeps back and forth along one axis. Keep the space clear.

## 7. TEST 5 — measured estimator noise (60 s, motors OFF, drone on the floor)

True velocity is 0 and position constant, so everything logged is estimator error. This
replaces the usual motion-capture comparison, which is impossible here because Lighthouse
*is* the source feeding the EKF.

- [ ] static
```bash
$COLLECT --excite step --excite-amp 0.0 --duration 60 --tag E6_static
```

---

## 8. Analysis (any machine)

```bash
python execution/sim2real/analyze_hover.py execution/sim2real/data/*A0_r? --json-out A0.json
python execution/sim2real/analyze_hover.py execution/sim2real/data/*A2_r? --json-out A2.json
python execution/sim2real/analyze_hover.py execution/sim2real/data/*A3_r? --json-out A3.json --plot

python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*E5_chirp_x --open-loop
python execution/sim2real/measure_delay_noise.py execution/sim2real/data/*E6_static  --static
```

### Reading it — `cmd_smoothness_mps_per_step`, baseline 0.0690

| result | meaning |
|---|---|
| A3 ≈ 0.03 or lower, position error not worse | latency was the gap; the simulator is predictive and A3 is a better policy |
| A3 ≈ A2 | the window helped, latency training did not — the sim latency model is wrong |
| all ≈ 0.069 | neither latency nor noise explains it; next suspects are the thrust curve and the 85 Hz vs 100 Hz rate mismatch |
| A4 ≫ A3 (if flown) | action history matters more on hardware than in sim |

Also record `mean_pos_err_m` (baseline 0.097 — smoothing by tracking worse is not a win),
`crashed`, and `effective_rate_hz` (expect 84-86 Hz; that is normal, not a fault).

`measure_delay_noise.py` prints the `env.action_latency_steps` / `env.noise_std` /
`env.noise_corr` to use in the next training round. Current values are guesses: latency
bracketed at 2-6 steps, noise 0.05 white, while the 07-24 flights suggest noise nearer
0.007 with AR(1) ≈ 0.26.

---

## 9. If something goes wrong

| symptom | fix |
|---|---|
| `Connection timeout after 10s` | cfclient still open, or wrong URI (`--uri radio://0/80/2M/E7E7E7E7E8`) |
| `size mismatch for net_container.0.weight` | wrong `--history-len` / `--action-hist-len` — this is the intended safe failure |
| immediate emergency land | kalman variance > 0.5 → Lighthouse tracking poor; check base stations and deck |
| `Position data stale` | radio dropouts → add `--log-rate-hz 50` |
| steady climb or drift | land. Note which policy; the static gate flags some for a residual climb command |

---

## Not part of this session

Asymmetric actor-critic, the MemoryStateCritic port, CTBR bring-up. None of the five flight
policies use any of it (all are plain PPO, `privileged_state=False`). Ignore it today.
