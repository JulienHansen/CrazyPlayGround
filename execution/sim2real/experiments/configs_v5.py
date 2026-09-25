"""Sweep 5: separate observation noise from airframe randomisation.

Sweep 3 concluded that airframe randomisation buys no generalisation, but it
varied the airframe while holding a single noise level, so "the airframe does not
matter" and "this noise level masks it" were not distinguishable. Sweep 1 has the
mirror problem: it varied noise with no airframe randomisation at all.

This is the missing factorial. Everything else is held at the best known setting
(K=8, M=0, delay 2-6 steps), and the two factors move independently:

    noise_std       sigma  0.0, 0.02, 0.05
    airframe        off, narrow, wide

Evaluation uses the same held-out range as sweep 3, wider than any used in
training, so the generalisation claim is tested rather than assumed.

9 runs x ~25 min ~= 3.8 h.
"""

import itertools

MAX_ITERATIONS = 600
NUM_ENVS = 4096
TASK = "Vel-Hovering-Robust"
WANDB_PROJECT = "crazyplayground-hover"
WANDB_GROUP = "hover-noise-dr-factorial-v5"
SEED = 42

BASE = {
    "env.history_len": 8, "env.action_hist_len": 0,
    "env.noise_corr": 0.9, "env.noise_bias_std": 0.01,
    "env.disturb": "True", "env.disturb_gust_prob": 0.01,
    "env.action_latency_steps": 2, "env.action_latency_max": 6,
    "env.action_rate_reward_scale": 0.0,
    "env.privileged_state": "True",
}

NOISE = [0.0, 0.02, 0.05]
DR = {
    "off": {"env.physical_dr": "False"},
    "narrow": {"env.physical_dr": "True",
               "env.rand_mass_scale_range": "[0.9,1.1]",
               "env.rand_inertia_scale_range": "[0.9,1.1]",
               "env.rand_thrust_scale_range": "[0.92,1.08]",
               "env.rand_torque_scale_range": "[0.92,1.08]"},
    "wide": {"env.physical_dr": "True",
             "env.rand_mass_scale_range": "[0.7,1.4]",
             "env.rand_inertia_scale_range": "[0.7,1.4]",
             "env.rand_thrust_scale_range": "[0.8,1.2]",
             "env.rand_torque_scale_range": "[0.8,1.2]"},
}


def build_sweep():
    runs = []
    for sigma, (dr_name, dr) in itertools.product(NOISE, DR.items()):
        tag = f"n{sigma}_dr{dr_name}"
        overrides = dict(BASE)
        overrides["env.add_noise"] = "True" if sigma > 0 else "False"
        overrides["env.noise_std"] = sigma
        overrides.update(dr)
        runs.append({"tag": tag, "seed": SEED, "overrides": overrides,
                     "factors": {"noise_std": sigma, "physical_dr": dr_name, "seed": SEED}})
    return runs
