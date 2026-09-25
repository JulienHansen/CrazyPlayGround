"""Sweep 6: does a multiscale distance reward reduce steady-state error?

The base reward is 20(1 - tanh(d/0.8)). Its gradient is largest at the goal in
absolute terms, but the term is already within 4% of its maximum at d = 5 cm, so
the difference between a 5 cm hover and a 1 cm hover is worth about 0.5 reward
units against a per-step return dominated by other terms. The policy has little
incentive to close the last few centimetres.

"multiscale" replaces it with a mean of exponentials over length scales
1.0, 0.3, 0.1 and 0.03 m. Both peak at the same value, so the trade against the
velocity terms is unchanged, but the reward difference between 5 cm and 1 cm is
4.9x larger, and there is still usable gradient at a centimetre.

Two shapes x two seeds, everything else at the best known setting.

4 runs x ~25 min ~= 1.7 h.
"""

import itertools

MAX_ITERATIONS = 600
NUM_ENVS = 4096
TASK = "Vel-Hovering-Robust"
WANDB_PROJECT = "crazyplayground-hover"
WANDB_GROUP = "hover-reward-shape-v6"

BASE = {
    "env.history_len": 8, "env.action_hist_len": 0,
    "env.add_noise": "True", "env.noise_std": 0.02, "env.noise_corr": 0.9,
    "env.noise_bias_std": 0.01,
    "env.disturb": "True", "env.disturb_gust_prob": 0.01,
    "env.action_latency_steps": 2, "env.action_latency_max": 6,
    "env.action_rate_reward_scale": 0.0,
}

SHAPES = ["tanh", "multiscale"]
SEEDS = [42, 123]


def build_sweep():
    runs = []
    for shape, seed in itertools.product(SHAPES, SEEDS):
        tag = f"{shape}_s{seed}"
        overrides = dict(BASE)
        overrides["env.reward_shape"] = shape
        runs.append({"tag": tag, "seed": seed, "overrides": overrides,
                     "factors": {"reward_shape": shape, "seed": seed}})
    return runs
