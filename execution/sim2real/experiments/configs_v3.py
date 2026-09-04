"""Sweep 3: physical domain randomization, as groundwork for adaptive policies.

Sweeps 1-2 randomised only what acts *on* the drone (observation noise, external
wrench, loop latency). The drone itself was always the nominal 27 g Crazyflie, so
there was no model mismatch for a policy to adapt to -- which is the setting the
adaptive-control literature (RMA, Zhang et al. T-RO 2025) actually targets.

Sweep 3 randomises the body: mass, inertia, and actuation effectiveness. Crucially
the cascade PID keeps its nominal parameters from crazyflie.yaml, so the mismatch
is real and the policy has to absorb it.

    physical_dr    off / narrow / wide
    history_len    1 / 8            (can the policy infer the body from history?)
    action_hist_len 0 / 1           (does seeing its own command help inference?)

Latency (2-6 steps) and correlated noise are held ON throughout, since sweep 2
showed latency is the dominant transfer factor. Evaluation uses a WIDER held-out
DR range than training, following the papers' generalisation protocol.

12 runs x ~25 min ~= 5 h.
"""

import itertools

MAX_ITERATIONS = 600
NUM_ENVS = 4096
TASK = "Vel-Hovering-Robust"
WANDB_PROJECT = "crazyplayground-hover"
WANDB_GROUP = "hover-dr-sweep-v3"
SEED = 42

BASE = {
    "env.add_noise": "True", "env.noise_std": 0.02, "env.noise_corr": 0.9,
    "env.noise_bias_std": 0.01,
    "env.disturb": "True", "env.disturb_gust_prob": 0.01,
    "env.action_latency_steps": 2, "env.action_latency_max": 6,
    # expose ground truth to a critic; harmless for a symmetric PPO run and it makes
    # these checkpoints directly reusable for asymmetric actor-critic experiments
    "env.privileged_state": "True",
}

DR_LEVELS = {
    "off":    {"env.physical_dr": "False"},
    "narrow": {"env.physical_dr": "True",
               "env.rand_mass_scale_range": "[0.9,1.1]",
               "env.rand_inertia_scale_range": "[0.9,1.1]",
               "env.rand_thrust_scale_range": "[0.92,1.08]",
               "env.rand_torque_scale_range": "[0.92,1.08]"},
    "wide":   {"env.physical_dr": "True",
               "env.rand_mass_scale_range": "[0.7,1.4]",
               "env.rand_inertia_scale_range": "[0.7,1.4]",
               "env.rand_thrust_scale_range": "[0.8,1.2]",
               "env.rand_torque_scale_range": "[0.8,1.2]"},
}


def build_sweep():
    runs = []
    for dr, k, m in itertools.product(["off", "narrow", "wide"], [1, 8], [0, 1]):
        ov = dict(BASE); ov.update(DR_LEVELS[dr])
        ov["env.history_len"] = k
        ov["env.action_hist_len"] = m
        runs.append({"tag": f"dr{dr}_k{k}_m{m}", "seed": SEED, "overrides": ov,
                     "factors": {"physical_dr": dr, "history_len": k, "action_hist_len": m,
                                 "latency": "2-6", "noise_std": 0.02, "disturb": True,
                                 "seed": SEED}})
    return runs


SWEEP = build_sweep()

if __name__ == "__main__":
    for i, r in enumerate(SWEEP):
        print(f"{i:2d}  {r['tag']:16s}  {r['factors']}")
    print(f"\n{len(SWEEP)} runs x ~{MAX_ITERATIONS} iters")
