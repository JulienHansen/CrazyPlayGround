"""Training-sweep design for the hover robustness experiment.

Factors (all on the ``Vel-Hovering-Robust`` task):
  * observation noise  -- env.add_noise / env.noise_std
  * external disturbance -- env.disturb (+ gust probability)
  * observation window  -- env.history_len (frame stacking, K)
  * anti-chatter penalty -- env.action_rate_reward_scale

Rationale: the measured sim-to-real gap was ~10x worse command chatter on the real
drone with near-equal position accuracy, so the sweep is built to find which of
these factors reduces chatter and improves disturbance rejection without giving up
tracking performance.

Training length: the original 48k-timestep run over-trained (reward peaked around
timestep ~10k then declined ~35%). `MAX_ITERATIONS` keeps runs just past the peak.
"""

import itertools

# skrl: timesteps = max_iterations * rollouts(32). 600 -> 19.2k timesteps.
MAX_ITERATIONS = 600
NUM_ENVS = 4096
TASK = "Vel-Hovering-Robust"
WANDB_PROJECT = "crazyplayground-hover"
WANDB_GROUP = "hover-robustness-sweep"

# Disturbance preset used whenever disturbances are enabled.
DISTURB_ON = {
    "env.disturb": "True",
    "env.disturb_gust_prob": 0.01,          # ~1 gust/s at 100 Hz
    "env.disturb_force_bias_range": "[-0.02,0.02]",
    "env.disturb_torque_bias_range": "[-0.0002,0.0002]",
    "env.disturb_gust_force_range": "[-0.06,0.06]",
    "env.disturb_gust_torque_range": "[-0.0006,0.0006]",
}


def _run(tag, noise_std, history_len, disturb, seed, action_rate=0.0):
    ov = {
        "env.add_noise": "True" if noise_std > 0 else "False",
        "env.noise_std": noise_std,
        "env.history_len": history_len,
        "env.action_rate_reward_scale": action_rate,
    }
    if disturb:
        ov.update(DISTURB_ON)
    else:
        ov["env.disturb"] = "False"
    return {"tag": tag, "seed": seed, "overrides": ov,
            "factors": {"noise_std": noise_std, "history_len": history_len,
                        "disturb": disturb, "action_rate": action_rate, "seed": seed}}


def build_sweep():
    """~24 runs: a noise x history grid (2 seeds), then disturbance + merged configs."""
    runs = []

    # A. noise x history grid, no disturbance, 2 seeds  -> 18 runs
    for noise, k, seed in itertools.product([0.0, 0.02, 0.05], [1, 4, 8], [42, 123]):
        runs.append(_run(f"n{noise}_k{k}_d0_s{seed}", noise, k, False, seed))

    # B. disturbance x (noise, history), seed 42        -> 4 runs
    for noise, k in itertools.product([0.02, 0.05], [4, 8]):
        runs.append(_run(f"n{noise}_k{k}_d1_s42", noise, k, True, 42))

    # C. anchors                                        -> 2 runs
    #    memoryless with disturbance (does the window actually help?)
    runs.append(_run("n0.05_k1_d1_s42", 0.05, 1, True, 42))
    #    full robustness + explicit anti-chatter penalty (the deploy candidate)
    runs.append(_run("n0.05_k8_d1_ar_s42", 0.05, 8, True, 42, action_rate=-0.05))

    return runs


SWEEP = build_sweep()

if __name__ == "__main__":
    for i, r in enumerate(SWEEP):
        print(f"{i:2d}  {r['tag']:24s}  {r['factors']}")
    print(f"\n{len(SWEEP)} runs x ~{MAX_ITERATIONS} iters")
