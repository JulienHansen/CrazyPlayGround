#!/usr/bin/env bash
# Train the pending runs of a sweep file with scripts/skrl/train.py, N at a time.
#
# Usage (from anywhere):
#   experiments/run_skrl.sh <sweep.json> [-j N] [--all] [--dry-run] [-- <train.py flags>]
#
#   -j N       train.py processes running at once (default 2); tune it on nvidia-smi
#   --all      train every run of the sweep, without asking W&B which are done
#   --dry-run  print the commands, launch nothing
#   -- ...     flags passed to every train.py as is, e.g. -- --device cuda:1 --video.
#              Not Hydra overrides: those would change a run without changing its run_key.
#
# Inside Docker: PYTHON="/workspace/isaaclab/isaaclab.sh -p" experiments/run_skrl.sh ...
# Each run logs to logs/sweeps/<sweep name>/<run_key>.log; skrl output goes to logs/skrl/ as usual.
set -euo pipefail

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

sweep="" jobs=2 dry_run=0 select=() extra=()
while (($#)); do
  case "$1" in
    -j) jobs="$2"; shift 2 ;;
    --all) select=(--all); shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage ;;
    --) shift; extra=("$@"); break ;;
    -*) echo "unknown option $1" >&2; usage 1 ;;
    *) sweep=$(realpath "$1"); shift ;;
  esac
done
[[ -n $sweep ]] || usage 1
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo "-j takes a positive integer, got $jobs" >&2; exit 1; }

cd "$(dirname "$0")/.."
read -ra python <<< "${PYTHON:-python}"
log_dir="logs/sweeps/$(basename "$sweep" .json)"

# expand the sweep before launching anything, so a bad file or W&B error stops here
lines=$("${python[@]}" experiments/skrl_args.py "$sweep" "${select[@]}")

run() {
  local key=$1 name=$2; shift 2
  if "$@" > "$log_dir/$key.log" 2>&1; then
    echo "[done]   $name"
  else
    echo "[FAILED] $name: exit $?, see $log_dir/$key.log"
    return 1
  fi
}

# Ctrl-C reaches the whole process group already; also stop the children on TERM
trap 'trap - TERM; kill -TERM 0' TERM
reap() {
  if wait -n; then streak=0; else failed=$((failed + 1)) streak=$((streak + 1)); fi
  running=$((running - 1))
}

((dry_run)) || mkdir -p "$log_dir"
running=0 failed=0 streak=0
while IFS=$'\t' read -r key name argv; do
  [[ -n $key ]] || continue
  eval "args=($argv)"  # argv is shlex-quoted by skrl_args.py
  cmd=("${python[@]}" scripts/skrl/train.py "${args[@]}" --headless "${extra[@]}")
  if ((dry_run)); then printf '%q ' "${cmd[@]}"; echo; continue; fi
  ((running < jobs)) || reap
  # a broken setup (W&B access, bad override) would otherwise fail every run in turn
  ((streak < 3)) || { echo "3 runs failed in a row: launching no more" >&2; break; }
  echo "[start]  $name -> $log_dir/$key.log"
  run "$key" "$name" "${cmd[@]}" &
  running=$((running + 1))
done <<< "$lines"
while ((running)); do reap; done

((failed == 0)) || { echo "$failed run(s) failed" >&2; exit 1; }
