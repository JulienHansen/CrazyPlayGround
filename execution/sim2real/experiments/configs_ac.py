"""Asymmetric vs symmetric critic under observation noise.

The question: the actor can only see a noisy, latency-delayed, stacked observation,
while a critic used only during training could be given the CLEAN state. Does that
asymmetry help?

`vel_hovering_robust` exposes a 24-dim privileged vector whose last 6 entries are the
**noise-free** observation, alongside the true disturbance wrench, latency and noise
sigma. The two agent configs are identical -- same layer sizes, optimiser and
preprocessors -- and differ in exactly one line:

    symmetric   value network input: OBSERVATIONS
    asymmetric  value network input: concatenate([STATES, OBSERVATIONS])

so any difference is attributable to the critic's access to privileged information.

Noise is deliberately high (0.05, AR(1) 0.9) because that is the regime where the
asymmetry should matter. Physical DR is OFF to isolate the perception asymmetry from
the dynamics-identification question that sweep 3 covered.

Note on the critic input: STATES alone would be the classic Pinto asymmetric critic,
but with a history-dependent actor (K=8 stacking counts) a state-only critic is biased,
so the actor's observation is concatenated -- the observation-state variant. A true
memory-state critic needs a recurrent actor's hidden state.

4 runs x ~13 min ~= 1 h.
"""

MAX_ITERATIONS = 600
NUM_ENVS = 4096
TASK = "Vel-Hovering-Robust"
WANDB_PROJECT = "crazyplayground-hover"
WANDB_GROUP = "critic-asymmetry"

BASE = {
    "env.privileged_state": "True",     # identical env for both variants
    "env.physical_dr": "False",
    "env.history_len": 8,
    "env.action_hist_len": 0,
    "env.add_noise": "True",
    "env.noise_std": 0.05,
    "env.noise_corr": 0.9,
    "env.noise_bias_std": 0.01,
    "env.action_latency_steps": 2,
    "env.action_latency_max": 6,
    "env.disturb": "True",
    "env.disturb_gust_prob": 0.01,
}

VARIANTS = {
    "sym":  "skrl_sym_cfg_entry_point",
    "asym": "skrl_asym_cfg_entry_point",
}


def build_sweep():
    runs = []
    for name, entry in VARIANTS.items():
        for seed in (42, 123):
            runs.append({"tag": f"{name}_s{seed}", "seed": seed, "agent": entry,
                         "overrides": dict(BASE),
                         "factors": {"critic": name, "seed": seed, "history_len": 8,
                                     "action_hist_len": 0, "noise_std": 0.05,
                                     "physical_dr": "off", "disturb": True}})
    return runs


SWEEP = build_sweep()

if __name__ == "__main__":
    for i, r in enumerate(SWEEP):
        print(f"{i}  {r['tag']:12s} agent={r['agent']}")
    print(f"\n{len(SWEEP)} runs")
