"""Expand a sweep file into the training runs it describes.

A sweep file lives in ``config/sweep/<task>/`` and has four top-level keys:

    version      int; bump it when an env or a trainer changes meaning, so runs
                 logged under an older version no longer count as done
    wandb        {"project": ...}, plus "entity" (defaults to the team)
    defaults     settings shared by every experiment
    experiments  list of {"name": ..., <overrides>, "sweep": {...}}

Each experiment is merged into ``defaults`` (dicts key by key, any other value
replaced), expanded over every combination of its ``sweep`` values, then over
``seed``. Sweep keys are dotted paths into the merged settings, e.g.
``"env_kwargs.noise_std"``. Outside ``sweep`` a list is a plain value (a
randomisation range, layer sizes); ``seed`` is the only key whose list means
"one run per value".

The parser checks only the keys it relies on: ``trainer``, ``env_name`` and
``seed``. Every other key is passed on as written; each ``train_<trainer>.py``
maps them to its own settings and rejects the ones it does not know.

Each run gets two hashes of its resolved settings and ``version``:

    run_key     identifies the run; wandb_check.py looks it up on W&B
    config_key  the same without ``seed``, so every seed of one config shares
                it; it picks the W&B group, and W&B averages over a group

The hashes treat ``1.0`` and ``1`` (or ``100e6`` and ``100000000``) as the same
value, but the settings themselves are passed on unchanged: Isaac Lab refuses
an int for a float field ("Incorrect type under namespace").

When two experiments resolve to the same run (a scan that passes through the
defaults, say), the run is produced once and the summary says so.

Usage (from the repo root):
    python parser.py config/sweep/hovering/test_01.json          # per-experiment summary
    python parser.py config/sweep/hovering/test_01.json --runs   # one line per run
    python parser.py config/sweep/hovering/test_01.json --json   # resolved runs, for scripts
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
import re
import sys
from typing import Callable, NoReturn

DEFAULT_ENTITY = "crazyplayground"
TRAINERS = ("skrl",)
TOP_LEVEL_KEYS = ("version", "wandb", "defaults", "experiments")
WANDB_KEYS = ("project", "entity")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
KEY_LENGTH = 12


class SweepError(ValueError):
    """The sweep file is malformed; the message says where."""


@dataclasses.dataclass
class Run:
    name: str  # W&B run name: <label>/s<seed>
    group: str  # W&B group "<label> [<config_key>]", shared by every seed of one config
    experiment: str  # experiment that first produced this run
    also_in: list[str]  # later experiments that resolved to the same run
    seed: int
    sweep_values: dict  # sweep key -> value, for this run
    config: dict  # resolved settings, seed included
    run_key: str
    config_key: str


@dataclasses.dataclass
class Sweep:
    path: str
    version: int
    entity: str
    project: str
    experiments: list[str]  # experiment names, in file order
    runs: list[Run]


def load_sweep(path: str) -> Sweep:
    """Read a sweep file and expand it into runs."""
    try:
        with open(path) as f:
            data = json.load(
                f,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
    except json.JSONDecodeError as e:
        raise SweepError(f"{path}: invalid JSON: {e}") from None
    except SweepError as e:
        raise SweepError(f"{path}: {e}") from None
    return expand(data, path)


def expand(data: dict, path: str = "<sweep>") -> Sweep:
    """Expand already-loaded sweep data into runs. ``path`` only labels errors."""

    def fail(where: str, message: str) -> NoReturn:
        raise SweepError(f"{path}: {where}: {message}")

    if not isinstance(data, dict):
        fail("top level", "must be an object")
    _check_keys(data, TOP_LEVEL_KEYS, "top level", fail)

    version = data.get("version")
    if not _is_int(version) or version < 1:
        fail("version", f"must be an int >= 1, got {version!r}")

    wandb = data.get("wandb")
    if not isinstance(wandb, dict):
        fail("wandb", 'must be an object like {"project": "hovering"}')
    _check_keys(wandb, WANDB_KEYS, "wandb", fail)
    project = wandb.get("project")
    entity = wandb.get("entity", DEFAULT_ENTITY)
    for key, value in (("project", project), ("entity", entity)):
        if not isinstance(value, str) or not value:
            fail(f"wandb.{key}", f"must be a non-empty string, got {value!r}")

    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        fail("defaults", "must be an object")
    for key in ("name", "sweep"):
        if key in defaults:
            fail("defaults", f"{key!r} belongs in an experiment, not in defaults")

    experiments = data.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        fail("experiments", "must be a non-empty list")

    names: list[str] = []
    runs: dict[str, Run] = {}  # run_key -> run
    labels: dict[str, str] = {}  # config_key -> label; the first experiment names it
    for i, experiment in enumerate(experiments):
        where = f"experiments[{i}]"
        if not isinstance(experiment, dict):
            fail(where, "must be an object")
        name = experiment.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            fail(where, f"'name' must match {NAME_RE.pattern}, got {name!r}")
        where = f"{where} ({name})"
        if name in names:
            fail(where, "another experiment already has this name")
        names.append(name)

        overrides = {k: v for k, v in experiment.items() if k not in ("name", "sweep")}
        base = _merge(defaults, overrides)
        axes = _check_sweep(
            experiment.get("sweep", {}), overrides, base, f"{where} sweep", fail
        )

        for combo in itertools.product(*(values for _, values in axes)):
            config = copy.deepcopy(base)
            sweep_values = {}
            for (key, _), value in zip(axes, combo):
                _set_path(config, key, copy.deepcopy(value))
                sweep_values[key] = value
            seeds = _check_required(config, where, fail)

            settings = {k: v for k, v in config.items() if k != "seed"}
            config_key = _hash({"version": version, "config": settings})
            label = labels.setdefault(config_key, _label(name, sweep_values))
            # labels can repeat across sweep files sharing a W&B project; the key keeps
            # one group per config
            group = f"{label} [{config_key}]"
            for seed in seeds:
                run_config = {**copy.deepcopy(settings), "seed": seed}
                run_key = _hash({"version": version, "config": run_config})
                if run_key in runs:
                    first = runs[run_key]
                    if name != first.experiment and name not in first.also_in:
                        first.also_in.append(name)
                    continue
                runs[run_key] = Run(
                    name=f"{label}/s{seed}",
                    group=group,
                    experiment=name,
                    also_in=[],
                    seed=seed,
                    sweep_values=copy.deepcopy(sweep_values),
                    config=run_config,
                    run_key=run_key,
                    config_key=config_key,
                )

    return Sweep(
        path=path,
        version=version,
        entity=entity,
        project=project,
        experiments=names,
        runs=list(runs.values()),
    )


def _check_keys(
    obj: dict,
    allowed: tuple[str, ...],
    where: str,
    fail: Callable[[str, str], NoReturn],
) -> None:
    unknown = [k for k in obj if k not in allowed]
    if unknown:
        fail(where, f"unknown key(s) {unknown}; allowed: {list(allowed)}")


def _check_sweep(
    sweep: dict,
    overrides: dict,
    base: dict,
    where: str,
    fail: Callable[[str, str], NoReturn],
) -> list[tuple[str, list]]:
    """Validate an experiment's ``sweep`` block; return its (key, values) axes in file order."""
    if not isinstance(sweep, dict):
        fail(where, 'must be an object like {"lr": [1e-4, 3e-4]}')
    axes: list[tuple[str, list]] = []
    for key, values in sweep.items():
        parts = key.split(".")
        if not all(parts):
            fail(where, f"bad key {key!r}")
        if parts[0] == "seed":
            fail(
                where,
                "'seed' is always expanded; give the experiment a 'seed' list instead",
            )
        if not isinstance(values, list) or not values:
            fail(
                where,
                f"{key!r} must be a non-empty list of values to try, got {values!r}",
            )
        canonical = [_canonical(v) for v in values]
        if len(set(canonical)) != len(canonical):
            fail(where, f"{key!r} lists the same value twice")
        # a setting is set in one place: either the experiment fixes it or the sweep varies it
        for fixed in _leaf_paths(overrides):
            if _overlap(parts, fixed):
                fail(
                    where,
                    f"{key!r} is also set by the experiment ({'.'.join(fixed)!r}); set it in one place",
                )
        for other, _ in axes:
            if _overlap(parts, other.split(".")):
                fail(where, f"{key!r} and {other!r} set the same setting")
        node = base
        for depth, part in enumerate(parts[:-1]):
            node = node.get(part)
            if node is None:
                break  # _set_path creates the missing levels
            if not isinstance(node, dict):
                fail(
                    where, f"{key!r}: {'.'.join(parts[: depth + 1])!r} is not an object"
                )
        axes.append((key, values))
    return axes


def _check_required(
    config: dict, where: str, fail: Callable[[str, str], NoReturn]
) -> list[int]:
    """Check the keys the parser itself relies on; return the seeds."""
    trainer = config.get("trainer")
    if trainer not in TRAINERS:
        fail(where, f"'trainer' must be one of {list(TRAINERS)}, got {trainer!r}")
    env_name = config.get("env_name")
    if not isinstance(env_name, str) or not env_name:
        fail(where, f"'env_name' must be a non-empty string, got {env_name!r}")
    seeds = config.get("seed")
    if _is_int(seeds):
        seeds = [seeds]
    if not isinstance(seeds, list) or not seeds or not all(_is_int(s) for s in seeds):
        fail(where, f"'seed' must be an int or a non-empty list of ints, got {seeds!r}")
    if len(set(seeds)) != len(seeds):
        fail(where, f"'seed' lists the same seed twice: {seeds}")
    return seeds


def _merge(base: dict, override: dict) -> dict:
    """Merge ``override`` into a copy of ``base``: dicts key by key, any other value replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_path(config: dict, key: str, value) -> None:
    """Set a dotted sweep key, with the same merge rule as ``_merge``."""
    *parents, last = key.split(".")
    node = config
    for part in parents:
        node = node.setdefault(part, {})
    if isinstance(node.get(last), dict) and isinstance(value, dict):
        node[last] = _merge(node[last], value)
    else:
        node[last] = value


def _leaf_paths(obj: dict, prefix: tuple[str, ...] = ()):
    for key, value in obj.items():
        if isinstance(value, dict):
            yield from _leaf_paths(value, prefix + (key,))
        else:
            yield prefix + (key,)


def _overlap(a, b) -> bool:
    """True when one path is a prefix of the other, i.e. both set the same setting."""
    n = min(len(a), len(b))
    return tuple(a[:n]) == tuple(b[:n])


def _canonical(value) -> str:
    """JSON text that is equal for equal settings: sorted keys, whole floats as ints."""
    return json.dumps(
        _whole_floats_to_int(value), sort_keys=True, separators=(",", ":")
    )


def _whole_floats_to_int(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _whole_floats_to_int(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_whole_floats_to_int(v) for v in value]
    return value


def _hash(value) -> str:
    return hashlib.sha1(_canonical(value).encode()).hexdigest()[:KEY_LENGTH]


def _label(experiment: str, sweep_values: dict) -> str:
    """``lr-scan/lr=0.0005``, or just the experiment name when it has no sweep."""
    if not sweep_values:
        return experiment
    values = ",".join(
        f"{k}={v if isinstance(v, str) else json.dumps(v, separators=(',', ':'))}"
        for k, v in sweep_values.items()
    )
    return f"{experiment}/{values}"


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    # json.load would silently keep the last of two equal keys
    out = {}
    for key, value in pairs:
        if key in out:
            raise SweepError(f"key {key!r} appears twice in the same object")
        out[key] = value
    return out


def _reject_constant(name: str) -> NoReturn:
    raise SweepError(f"{name} is not a valid setting")


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        description="Expand a sweep file into the runs it describes."
    )
    cli.add_argument("sweep_file")
    show = cli.add_mutually_exclusive_group()
    show.add_argument("--runs", action="store_true", help="one line per run")
    show.add_argument(
        "--json", action="store_true", help="print the expanded sweep as JSON"
    )
    args = cli.parse_args(argv)

    try:
        sweep = load_sweep(args.sweep_file)
    except (SweepError, OSError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(dataclasses.asdict(sweep), indent=2))
        return 0

    print(f"{sweep.path} -> {sweep.entity}/{sweep.project} (version {sweep.version})")
    if args.runs:
        for run in sweep.runs:
            also = f"  (also in {', '.join(run.also_in)})" if run.also_in else ""
            print(f"  {run.run_key}  {run.config['trainer']:<6}  {run.name}{also}")
    else:
        width = max(len(name) for name in sweep.experiments)
        for name in sweep.experiments:
            owned = sum(run.experiment == name for run in sweep.runs)
            shared = sorted(
                {run.experiment for run in sweep.runs if name in run.also_in}
            )
            line = f"  {name:<{width}}  {owned:>4} runs"
            if shared:
                count = sum(name in run.also_in for run in sweep.runs)
                line += f"  (+{count} already in {', '.join(shared)})"
            print(line)
    duplicates = sum(len(run.also_in) for run in sweep.runs)
    print(
        f"  total: {len(sweep.runs)} runs"
        + (f", {duplicates} duplicates dropped" if duplicates else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
