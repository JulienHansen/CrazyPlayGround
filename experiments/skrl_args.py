"""Turn the runs of a sweep file into ``scripts/skrl/train.py`` arguments.

run_skrl.sh calls this and launches one train.py per printed line. Sweep keys map
to train.py flags and Hydra overrides:

    env_name         --task
    seed             --seed
    num_envs         --num_envs
    algo             --algorithm
    lr               agent.agent.learning_rate
    lambda_coef      agent.agent.lambda
    entropy_coef     agent.agent.entropy_loss_scale
    rollout_length   agent.agent.rollouts
    total_timesteps  agent.trainer.timesteps = total_timesteps / num_envs, rounded up
    env_kwargs.<k>   env.<k>

Any other key is refused. Each run also gets its W&B name, group, project and
entity, plus ``run_key``, ``config_key`` and ``exp`` (the resolved settings) in
the W&B config, which is what wandb_check.py looks up.

Usage (from the repo root):
    python experiments/skrl_args.py experiments/config/sweep/hovering/test_01.json         # failed + missing runs
    python experiments/skrl_args.py experiments/config/sweep/hovering/test_01.json --all   # every run, no W&B query

Each line is ``<run_key>\\t<run name>\\t<shell-quoted train.py args>``.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys

from parser import Run, Sweep, SweepError, load_sweep

FLAGS = {"env_name": "--task", "seed": "--seed", "num_envs": "--num_envs", "algo": "--algorithm"}
AGENT_KEYS = {
    "lr": "agent.agent.learning_rate",
    "lambda_coef": "agent.agent.lambda",
    "entropy_coef": "agent.agent.entropy_loss_scale",
    "rollout_length": "agent.agent.rollouts",
}
OTHER_KEYS = ("trainer", "total_timesteps", "env_kwargs")
WANDB = "++agent.agent.experiment.wandb_kwargs"


def run_args(run: Run, sweep: Sweep) -> list[str]:
    """The train.py arguments that train ``run``.

    Raises:
        SweepError: a key train.py cannot take, or ``total_timesteps`` without
            ``num_envs``.
    """
    config = run.config
    unknown = [k for k in config if k not in FLAGS and k not in AGENT_KEYS and k not in OTHER_KEYS]
    if unknown:
        raise SweepError(f"{run.name}: skrl does not take key(s) {unknown}")
    if config["trainer"] != "skrl":
        raise SweepError(f"{run.name}: trainer is {config['trainer']!r}, not 'skrl'")

    args = [f"{flag}={config[key]}" for key, flag in FLAGS.items() if key in config]
    args += [f"{path}={_value(config[key])}" for key, path in AGENT_KEYS.items() if key in config]
    if "total_timesteps" in config:
        # skrl counts timesteps per env; the sweep counts them summed over all envs.
        # Rounded up, so a run trains at least total_timesteps.
        num_envs = config.get("num_envs")
        if num_envs is None:
            raise SweepError(f"{run.name}: total_timesteps needs num_envs")
        args.append(f"agent.trainer.timesteps={math.ceil(config['total_timesteps'] / num_envs)}")
    args += [f"env.{k}={_value(v)}" for k, v in config.get("env_kwargs", {}).items()]

    args += [
        "agent.agent.experiment.wandb=True",
        # two runs started in the same second would otherwise share a log dir
        f"agent.agent.experiment.experiment_name={run.run_key}",
        f"{WANDB}.project={_value(sweep.project)}",
        f"{WANDB}.entity={_value(sweep.entity)}",
        f"{WANDB}.name={_value(run.name)}",
        f"{WANDB}.group={_value(run.group)}",
        f"{WANDB}.config.run_key={_value(run.run_key)}",
        f"{WANDB}.config.config_key={_value(run.config_key)}",
    ]
    args += [f"{WANDB}.config.exp.{path}={_value(v)}" for path, v in _leaves(config)]
    return args


def _value(value) -> str:
    """A Hydra override value: JSON keeps ``0.0`` a float and quotes strings."""
    return json.dumps(value, separators=(",", ":"))


def _leaves(obj: dict, prefix: str = ""):
    for key, value in obj.items():
        if isinstance(value, dict):
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Print the train.py arguments of a sweep's runs.")
    cli.add_argument("sweep_file")
    cli.add_argument("--all", action="store_true", help="every run, without asking W&B which are done")
    args = cli.parse_args(argv)

    try:
        sweep = load_sweep(args.sweep_file)
        runs = sweep.runs
        if not args.all:
            from wandb_check import check_sweep

            try:
                runs = check_sweep(sweep).to_train()
            except Exception as e:  # login, network, permissions: report rather than guess
                print(f"[ERROR] W&B query failed: {e}", file=sys.stderr)
                return 1
        lines = [f"{run.run_key}\t{run.name}\t{shlex.join(run_args(run, sweep))}" for run in runs]
    except (SweepError, OSError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    print(f"[INFO] {len(lines)} of {len(sweep.runs)} runs to train", file=sys.stderr)
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
