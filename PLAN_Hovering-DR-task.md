# Plan — `Feature/Hovering-DR-task` (WP1, issue #20)

Re-implements WP1 (`Vel-Hovering-DR` — domain randomisation; `Vel-Hovering-Robust` in Arthur's PR #19, commits `525696e` + `d5ae8d0`)
on a fresh branch off `develop`, with the review fixes folded into the commit that introduces each
feature. Untracked working note — not part of the branch.

## Ground rules

- Branch: `Feature/Hovering-DR-task` from `develop` (`52536d9`). `vel_hovering.py` is identical on
  `develop` and PR #19, so no rebase work.
- Files: `source/CrazyPlayGround/CrazyPlayGround/tasks/direct/hovering/vel_hovering_dr.py` (new),
  `hovering/__init__.py` (register), and in commit 4 one new agent yaml.
- Every commit leaves the task importable and, with default knobs, identical to `Vel-Hovering`.
- Out of scope: the W&B yaml edit from PR #19 (touches every hovering task), measured numbers in the
  docstring, sweeps/experiments (later WPs).
- Local only; push when asked.

## Commits

### 1. Task skeleton, registration, frame stacking, action history
- `DRQuadcopterEnvCfg(QuadcopterEnvCfg)` + `DRQuadcopterEnv(QuadcopterEnv)`; register
  `Vel-Hovering-DR` (same agent entry points as `Vel-Hovering`).
- Knobs: `obs_input_hist` (1), `action_input_hist` (0).
- Obs layout, oldest-first: `[K x (lin_vel_b(3), desired_pos_b(3)), M x action(3)]` → `6K + 3M`.
- **All** validation and `observation_space`/`state_space` computed in the constructor before
  `super().__init__` (sees Hydra overrides). No `__post_init__` duplicate.
- Buffers allocated once in `__init__` — no `_ensure_buffers()`: `DirectRLEnv.__init__` never calls
  `_reset_idx`/`_get_observations` (verified: only `reset()`/`step()` do).
- Observation window primed with the post-reset observation.

### 2. Loop latency + action-rate penalty
- Knobs: `action_latency_steps` (0), `action_latency_max_steps` (0; > steps → uniform per episode).
- One command buffer `_cmd_hist` (newest last) shared by the action history, the latency (`buf[L-1-d]`, per env) and the penalty (`[-1] - [-2]`); zeroed on reset.
- `action_rate_reward_scale` (0.0) penalises **commanded** `‖c_t − c_{t−1}‖²`, not the delayed action
  the base stores in `self._actions` (fixes review issue #1). Logged as `Episode_Reward/action_rate`.

### 3. AR(1) noise, per-episode bias, constant wind
- Knobs: `noise_corr` (0.0), `noise_bias_std` (0.0), reuse base `add_noise`/`noise_std`.
  AR(1): `n ← ρ n + √(1−ρ²) ε` (unit stationary variance), state drawn from N(0, 1) at reset so the
  noise is at full strength from the first step. Warning if correlation/bias is set while
  `add_noise=False`.
- Noise-free state stored in `true_state`; obs = `true_state` + noise.
- Wind: `wind_force_range` ((0, 0) = off). Per episode: uniform strength, uniform 3-D direction,
  constant world-frame force, added through Isaac Lab's `instantaneous_wrench_composer`
  (`is_global=True`). Replaces PR #19's per-axis bias, gusts and body torque.
- All attributes the DR env adds are public (no leading `_`); knob checks in `check_cfg`.

### 4. Actuation randomisation (thrust and torque effectiveness)
- Knobs: `thrust_scale_range`, `torque_scale_range` ((1, 1) = off), drawn per episode.
- The body gets `scale x controller output`: `(scale - 1) x T` is added to the instantaneous wrench
  composer on top of the base env's thrust `T` (permanent composer). The PID keeps its nominal
  values.

### Deferred (kept for later commits)
- **Multiscale distance reward.** `mean_s exp(-d/s)`, default scales `(1.0, 0.3, 0.1, 0.03)` m:
  keeps a slope at every distance (steep at the goal like a small `s`, reach like a large `s`).
  One more branch in `_distance_reward`.
- **Mass / inertia randomisation.** Design: Isaac Lab's `randomize_rigid_body_mass` pattern, i.e.
  always scale `data.default_mass` / `default_inertia` (no compounding), write through
  `root_physx_view.set_masses` / `set_inertias`, `inertia = default x mass_scale x inertia_scale`.
  Cost: PhysX mass properties are CPU tensors, so each write forces a GPU->CPU sync. With random
  episode lengths some envs reset almost every step. Measure on 4096 envs first; options are to
  randomise once at startup or every K steps in a batch.
- **Privileged critic.** `obs["critic"]` with the true state, the randomisation, wind, delay and the
  commands in flight; a separate `skrl_ppo_priv_cfg.yaml` (`separate: True`, value input
  `concatenate([OBSERVATIONS, STATES])`, explicit `observation_preprocessor` **and**
  `state_preprocessor`), registered as `skrl_ppo_priv_cfg_entry_point`. With the current yaml, skrl
  receives the critic state but the value network never reads it.

### 5. Distance reward shape: tanh or single exponential
- Knobs: `distance_reward_shape` ("tanh" default | "exp"), `distance_reward_length` (0.3 m, "exp" only).
- "exp" = `exp(-d / length)`: 1 at the goal like tanh (weight keeps its meaning), slope `1/length`
  at the goal vs 1.25 for tanh; flatter far away (reach trade-off: early in training the logged
  distance reward is much lower than with tanh, since untrained drones are far from the goal).
- Implemented through a `_distance_reward(d)` hook in `vel_hovering.py` (same expression for
  Vel-Hovering), so the base `_get_rewards` still weights and logs every term.

### 6. Merge into Vel-Hovering
- One task, `Vel-Hovering`: `vel_hovering.py` carries every knob above, all off by default; class
  names stay `QuadcopterEnv` / `QuadcopterEnvCfg`. `vel_hovering_dr.py` and the `Vel-Hovering-DR`
  registration are removed.
- Tests: `test/test_vel_hovering.py` (renamed), plus `test/vel_hovering_classic.py`, a frozen copy
  of the pre-WP1 env. Four knobs-off tests check that the merged env with defaults matches it
  exactly (bit-for-bit over a fixed 20-step rollout), logs included, and that the shared cfg
  defaults are unchanged.

## Validation (per commit)

Env: `conda activate DragonFly` (Isaac Sim 4.5 pip, Isaac Lab 0.54.4, skrl 2.1.0, RTX 4090).
`skrl_ppo_cfg.yaml` has `wandb: True,` → set `WANDB_MODE=disabled` for smoke runs.

```bash
cd CrazyPlayGround
python scripts/list_envs.py | grep Vel-Hovering
WANDB_MODE=disabled python scripts/skrl/train.py --task=Vel-Hovering \
    --num_envs=256 --max_iterations=5 --headless
# then with the commit's knobs on, e.g. commit 2:
WANDB_MODE=disabled python scripts/skrl/train.py --task=Vel-Hovering \
    --num_envs=256 --max_iterations=5 --headless \
    env.obs_input_hist=4 env.action_input_hist=3 env.action_latency_steps=2 env.action_latency_max_steps=6 \
    env.action_rate_reward_scale=-0.01
```

- Commit 1: also an invalid override (`env.action_input_hist=-1`) must fail at construction.
- Commit 4: run with `--agent skrl_priv_cfg_entry_point env.privileged_state=true` and confirm the value
  net input size is the critic dim.

## Open decisions

1. ~~Disturbance frame~~: decided, constant world-frame wind (commit 3).
2. Privileged critic: deferred (see "Deferred" above).
