# CLAUDE.md

Guidance for AI coding agents working in this repository; it doubles as a map
for humans. Keep it short and true: when a change makes something here wrong,
update this file in the same PR.

## What this repo is

Crazyflie 2.1 reinforcement-learning environments on Isaac Lab (Direct
workflow), a cascaded PID inner loop modelled on the Crazyflie firmware, and
scripts that fly trained policies on real drones. Several researchers share it,
and experiment results are centralised in one W&B team, `crazyplayground`.

## Where things are

| Path | What |
|---|---|
| `source/CrazyPlayGround/CrazyPlayGround/tasks/direct/<task>/` | Environments. `__init__.py` registers the gym IDs; `agents/*.yaml` holds per-trainer hyperparameters |
| `source/CrazyPlayGround/CrazyPlayGround/controllers/` | Cascaded PID (`cascade_pid.py`, gains in `crazyflie.yaml`) that turns policy setpoints into thrust and moments |
| `source/CrazyPlayGround/CrazyPlayGround/wandb_tb_patch.py` | Forwards skrl's TensorBoard scalars to W&B; imported by the package |
| `scripts/<trainer>/train.py`, `play.py` | Isaac Lab entry points; skrl is the trainer the team uses |
| `execution/` | Deployment on real Crazyflies over radio (cflib); see **Real hardware** |
| `experiments/config/sweep/<task>/*.json` | Experiment definitions; see **Running experiments** |
| `experiments/parser.py`, `wandb_check.py` | Expand a sweep file into runs; check which runs are already done on W&B |
| `tests/` | pytest. Hardware tests drive the real scripts against a fake Crazyflie |
| `docs/` | MkDocs: environments, controller, configuration, deployment |

Task IDs read `<Level>-<Task>`, where the level is the setpoint the policy sends
to the PID: `Pos`, `Vel`, `Att` or `Rate` (e.g. `Vel-Hovering`). List them with
`python scripts/list_envs.py`.

## Versions and where code can run

- **Isaac Lab 2.3.2** (Docker base image `nvcr.io/nvidia/isaac-lab:2.3.2`),
  Python 3.10/3.11. Isaac Sim needs Linux and an NVIDIA GPU, so training,
  playback and anything importing `isaaclab` run only on the training machine or
  in Docker. On a Mac, only the pure-Python tools in `experiments/` run, with
  their tests.
- **skrl >= 2.1.** Its API differs from skrl 1.x, which most examples online
  use. Check the installed source before writing skrl code.
- **Quaternions are WXYZ** (Isaac Lab 2.x). An upgrade to Isaac Lab 3 (XYZW,
  Python 3.12) is planned; follow `docs/isaaclab-3-quaternion-migration.md`.
- Install with `pip install -e "source/CrazyPlayGround[rl]"` for training,
  `[deploy]` for hardware, `[all]` for both.

## Commands

```bash
python scripts/list_envs.py
python scripts/skrl/train.py --task=Vel-Hovering --num_envs=4096 --headless
python scripts/skrl/play.py --task=Vel-Hovering --checkpoint logs/skrl/<dir>/<run>/checkpoints/best_agent.pt
# inside Docker, prefix Isaac Lab scripts with: /workspace/isaaclab/isaaclab.sh -p

python experiments/parser.py experiments/config/sweep/hovering/test_01.json        # add --runs or --json
python experiments/wandb_check.py experiments/config/sweep/hovering/test_01.json   # add --runs; needs `wandb login`

python -m pytest tests/test_sweep_parser.py tests/test_wandb_check.py   # needs only pytest
python -m pytest tests/                                                 # also torch, pyarrow, cflib
```

`train.py` takes Hydra overrides: `env.<field>=<value>` for the env config and
`agent.agent.<field>=<value>` for skrl (e.g. `agent.agent.learning_rate=3e-4`).
A plain override fails on a key the config does not have, which catches typos;
prefix `++` only for keys you mean to add. Each run writes
`logs/skrl/<directory>/<timestamp>_ppo_torch/` with `checkpoints/` and the
resolved `params/env.yaml` and `params/agent.yaml`.

## Running experiments

Any result that is compared, reported or used to pick a policy comes from a
sweep file. Ad-hoc `train.py` runs are for debugging only; run them with
`agent.agent.experiment.wandb=False` so they stay out of W&B.

### Sweep files

`experiments/config/sweep/<task>/<name>.json`. The full rules are in the
`experiments/parser.py` docstring; in short:

```json
{
  "version": 1,
  "wandb": { "project": "hovering" },
  "defaults": { "trainer": "skrl", "env_name": "Vel-Hovering", "lr": 5e-4,
                "env_kwargs": { "add_noise": true, "noise_std": 0.01 }, "seed": [42, 123, 7] },
  "experiments": [
    { "name": "noise-off", "env_kwargs": { "add_noise": false } },
    { "name": "lr-scan", "sweep": { "lr": [2.5e-4, 5e-4, 1e-3] } }
  ]
}
```

- Each experiment is merged into `defaults`: dicts key by key, other values
  replaced.
- `sweep` runs every combination of its values. Keys are dotted paths
  (`"env_kwargs.noise_std"`). Outside `sweep`, a list is a plain value (a range,
  layer sizes). `seed` is the exception: its list always means one run per seed.
- `trainer`, `env_name` and `seed` are required. `env_kwargs.<field>` must be a
  field of the task's env config class. Agent keys (`lr`, `lambda_coef`,
  `entropy_coef`, `rollout_length`, `num_envs`, `total_timesteps` = env steps
  summed over all envs) are translated by the trainer script.
- Values reach the trainer exactly as written. Write floats as floats (`0.0`,
  not `0`): Isaac Lab refuses an int for a float field.
- A run is identified by `run_key`, a hash of its resolved settings plus
  `version`. Changing a value in the file therefore creates new runs on its own.
  **Bump `version` when code changes what the same settings mean**: reward,
  observations, actions, dynamics or randomisation, the controller, or the
  trainer's key translation.

### Workflow

1. Write or extend a sweep file. Keep the baseline in the same file, vary one
   factor per experiment, and give every config you compare the same seeds.
2. `python experiments/parser.py <file>`: check the run count and the duplicates it drops.
3. `python experiments/wandb_check.py <file>`: only `failed` and `missing` runs get trained.
4. Launch with the trainer script (see **Status**). Launching is the user's
   decision: it spends shared GPU hours and writes to the team W&B. Ask first.
5. Compare on W&B by group. A group is one config, and W&B plots the mean and
   spread over its seeds. Only compare runs with the same `total_timesteps`.
   When reporting, give the sweep file, the commit and the W&B groups.

## W&B conventions

- Entity `crazyplayground` (the team). One project per task, named after the
  task's folder: `tasks/direct/<task>/` for the YAMLs,
  `experiments/config/sweep/<task>/` for sweep files (`hovering`, `track`,
  `drone_racing`, ...). A new task or sweep folder follows the same rule.
- A run is named `<label>/s<seed>`; the label is the experiment name plus its
  sweep values (`lr-scan/lr=0.0005`). Its group is `<label> [<config_key>]`,
  shared by all seeds of one config. The key keeps two sweep files that reuse a
  label in separate groups.
- Run config: `run_key`, `config_key` (same hash without the seed), and `exp`
  (the resolved sweep settings). Summary: `completed: true` once training ends.
- Find runs with
  `wandb.Api().runs("crazyplayground/<project>", filters={"config.run_key": key})`.
  Never delete or edit W&B runs.

## Gotchas

- **skrl 2.x API:** `PPO_CFG` dataclass, `inputs["observations"]`, and a policy
  `compute` that returns `(mean, {"log_std": ...})`.
- **Preprocessor key:** the YAMLs write `state_preprocessor:`, and skrl's Runner
  maps it to `observation_preprocessor`, the key stored in checkpoints.
  Deployment code must configure `observation_preprocessor`. With
  `state_preprocessor` it silently skips observation normalisation.
- **One simulator per process:** Isaac Lab cannot build a second simulator in
  one process, so each training or evaluation run is its own subprocess.
- **Env settings and W&B:** skrl sends only agent, trainer and model settings to
  W&B. Env settings reach W&B only if the launcher adds them.
- **Branch names in lowercase** (`feature/...`, never `Feature/...`): macOS is
  case-insensitive, so mixed case makes `git fetch` delete and recreate refs.

## Real hardware

Scripts in `execution/` fly real drones. Never run them. Don't change control
loops, rates, limits or safety checks unless the user asks. Flag any change to
frames, signs, units or quaternion order for bench verification before a flight.
Test against the fake Crazyflie (`tests/test_hover_scripts.py`).

## Git

- Branch from `develop` and open PRs into `develop`; `main` gets releases from
  `develop`.
- `logs/`, `wandb/` and checkpoints are gitignored; keep it that way.

## Status and open decisions (2026-10-19)

- **Not written yet:** `experiments/train_skrl.py`, the launcher. It will:
  - translate sweep keys into skrl/Hydra overrides;
  - re-check W&B just before each run;
  - add `run_key`, `config_key` and `exp` to the W&B config;
  - set `summary["completed"]`.

  Until it exists, don't hand-roll comparison runs; ask.
- **Planned:** `experiments/train_orcax.py` (Orcax, a JAX RL library), after the
  Isaac Lab 3 / Python 3.12 upgrade. `"orcax"` is not an accepted `trainer` yet.
- **Undecided by the team. Ask, don't assume:**
  - the reference hover env (`Vel-Hovering-Robust` and `Vel-Hovering-DR` are on
    unmerged branches);
  - the evaluation condition and score used to rank policies;
  - the minimum number of seeds (examples use 3).
