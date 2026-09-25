"""Sweep 4: how much history, of which kind?

Sweeps 1-2 varied the observation window K and the action history M, but never
together over a range: K was pinned at 8 whenever M moved, and only M=1 was ever
flown. The hardware result that motivates this is uncomfortable — A3 (K=8, M=0)
and A4 (K=8, M=1) differ only in M, and A4 was 2.4x worse in position error. That
is either a real cost of feeding the policy its own commands, or one seed.

This grid separates the two axes and puts several seeds behind the answer:

    history_len      K   4, 8, 16
    action_hist_len  M   0, 2, 4, 8

Delay (2-6 steps) and correlated noise are held on, since sweep 2 showed delay is
the dominant transfer factor and there is no point measuring history without it.

12 runs x ~25 min ~= 5 h.
"""

import itertools

MAX_ITERATIONS = 600
NUM_ENVS = 4096
TASK = "Vel-Hovering-Robust"
WANDB_PROJECT = "crazyplayground-hover"
WANDB_GROUP = "hover-history-sweep-v4"
SEED = 42

BASE = {
    "env.add_noise": "True", "env.noise_std": 0.02, "env.noise_corr": 0.9,
    "env.noise_bias_std": 0.01,
    "env.disturb": "True", "env.disturb_gust_prob": 0.01,
    "env.action_latency_steps": 2, "env.action_latency_max": 6,
    "env.action_rate_reward_scale": 0.0,
}

HISTORY = [4, 8, 16]
ACTIONS = [0, 2, 4, 8]


def build_sweep():
    runs = []
    for k, m in itertools.product(HISTORY, ACTIONS):
        tag = f"k{k}_m{m}"
        overrides = dict(BASE)
        overrides["env.history_len"] = k
        overrides["env.action_hist_len"] = m
        runs.append({"tag": tag, "seed": SEED, "overrides": overrides,
                     "factors": {"history_len": k, "action_hist_len": m, "seed": SEED}})
    return runs
