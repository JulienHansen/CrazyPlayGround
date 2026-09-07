# Flight set — checkpoint paths
Generated from the sweep indices. Verify each with `check_policy.py` before flying.

## A0 `n0.0_k1_d0_s42` — K=1, M=0
BASELINE (already flown 07-24)

```
/home/alouette/Bureau/CrazyPlayGround/logs/skrl/cartpole_direct/2026-07-27_17-16-00_ppo_torch/checkpoints/best_agent.pt
```

## A1 `n0.02_k8_d1_s42` — K=8, M=0
K=8, noise .02, disturb

```
/home/alouette/Bureau/CrazyPlayGround/logs/skrl/cartpole_direct/2026-07-28_01-40-32_ppo_torch/checkpoints/best_agent.pt
```

## A2 `m0_latoff_aroff` — K=8, M=0
K=8, NO latency in training

```
/home/alouette/Bureau/CrazyPlayGround/logs/skrl/cartpole_direct/2026-08-28_13-17-16_ppo_torch/checkpoints/best_agent.pt
```

## A3 `m0_laton_aroff` — K=8, M=0
K=8, latency-trained  <-- best in sim

```
/home/alouette/Bureau/CrazyPlayGround/logs/skrl/cartpole_direct/2026-08-28_14-05-19_ppo_torch/checkpoints/best_agent.pt
```

## A4 `m1_laton_aroff` — K=8, M=1
K=8 + 1 past action, latency-trained

```
/home/alouette/Bureau/CrazyPlayGround/logs/skrl/cartpole_direct/2026-08-28_15-37-52_ppo_torch/checkpoints/best_agent.pt
```
