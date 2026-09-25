"""Second sweep: the factors the first one missed.

Sweep 1 varied observation noise x observation-window x disturbance, and found the
window cuts chatter ~2x. But it left out the two things most likely to actually
explain the hardware chatter:

  * the policy never saw its own previous command, so it could neither regulate its
    step-to-step change directly nor infer a disturbance (which needs the residual
    between what was commanded and what happened) -- and the action-rate penalty in
    sweep 1 was therefore scoring a quantity the policy could not perceive;
  * there was no loop latency at all, while the real loop runs at 85.7 Hz with radio
    and estimator lag on top. Delay plus high gain is the textbook cause of the
    2-4 Hz limit cycle seen on hardware.

Sweep 2 holds the best-known sweep-1 setting fixed (K=8, disturbances on) and varies:
    action_hist_len M       0, 1, 4       does seeing its own command help?
    latency                 none / 2-6    train with dead time or not
    action_rate penalty     0, -0.05      now that it is actually observable

Latency is randomised per episode over 2-6 steps (20-60 ms) rather than fixed: the
measured sweep of the deployed policy gives 0.0060 chatter at 0 steps, 0.0300 at 2,
0.0517 at 4, against 0.0690 on hardware -- so the true loop delay is at least 4 steps
but has never been measured directly, and baking in one guessed value is fragile.

Observation noise is also switched from white to AR(1) correlated (rho=0.9), because
white noise is unrealistically easy to filter by temporal averaging and probably
flattered the sweep-1 window result.

12 runs x ~26 min ~= 5.3 h.
"""

import itertools

MAX_ITERATIONS = 600
NUM_ENVS = 4096
TASK = "Vel-Hovering-Robust"
WANDB_PROJECT = "crazyplayground-hover"
WANDB_GROUP = "hover-robustness-sweep-v2"

# Held fixed: best-known configuration from sweep 1, with realistic (correlated) noise.
BASE = {
    "env.history_len": 8,
    "env.add_noise": "True",
    "env.noise_std": 0.02,
    "env.noise_corr": 0.9,
    "env.noise_bias_std": 0.01,
    "env.disturb": "True",
    "env.disturb_gust_prob": 0.01,
    "env.disturb_force_bias_range": "[-0.02,0.02]",
    "env.disturb_torque_bias_range": "[-0.0002,0.0002]",
    "env.disturb_gust_force_range": "[-0.06,0.06]",
    "env.disturb_gust_torque_range": "[-0.0006,0.0006]",
}

SEED = 42


def build_sweep():
    runs = []
    for m, lat, ar in itertools.product([0, 1, 4], [False, True], [0.0, -0.05]):
        ov = dict(BASE)
        ov["env.action_hist_len"] = m
        ov["env.action_latency_steps"] = 2 if lat else 0
        ov["env.action_latency_max"] = 6 if lat else 0
        ov["env.action_rate_reward_scale"] = ar
        tag = f"m{m}_lat{'on' if lat else 'off'}_ar{'on' if ar else 'off'}"
        runs.append({"tag": tag, "seed": SEED, "overrides": ov,
                     "factors": {"action_hist_len": m, "latency": "2-6" if lat else "0",
                                 "action_rate": ar, "history_len": 8,
                                 "noise_std": 0.02, "disturb": True, "seed": SEED}})
    return runs


SWEEP = build_sweep()

if __name__ == "__main__":
    for i, r in enumerate(SWEEP):
        print(f"{i:2d}  {r['tag']:16s}  {r['factors']}")
    print(f"\n{len(SWEEP)} runs x ~{MAX_ITERATIONS} iters")
