# IsaacLab 3.0 quaternion convention — migration watchlist

## Situation

[IsaacLab issue #5186](https://github.com/isaac-sim/IsaacLab/issues/5186) confirms that IsaacLab 3.0
(verified against the `v3.0.0-beta2.patch1` tag, March 2026) switched every quaternion in the
framework from **WXYZ** (scalar-first) to **XYZW** (scalar-last), to align with Warp, PhysX, and
Newton conventions.

CrazyPlayGround currently pins **IsaacLab 2.3.2** (`docker/Dockerfile`, `README.md`)and A full repo audit by ClaudeCode confirmed every quaternion site in this codebase consistently and
correctly uses **WXYZ** today — there is no bug and nothing needs to change while pinned to 2.3.2.
This document exists so that whoever upgrades past IsaacLab 3.0 knows exactly what to check.

## Checklist for upgrading past IsaacLab 3.0

1. Run IsaacLab's own quaternion-finder migration tool against this repo first: [`scripts/tools/find_quaternions.py`](https://github.com/isaac-sim/IsaacLab/blob/main/scripts/tools/find_quaternions.py) in the IsaacLab repo (e.g. `python tools/find_quaternions.py --path <this-repo> --check-identity`). See the [migration guide](https://isaac-sim.github.io/IsaacLab/develop/source/migration/migrating_to_isaaclab_3-0.html) for full usage.
2. Re-verify every site listed above by hand — the finder tool targets IsaacLab API calls, but this
   repo also has several hand-rolled `quat_apply`/`quat_conjugate`/`quat_inv` copies that use raw
   index arithmetic (`quat[..., 0]` as the scalar component) and won't be caught by a tool scoped to
   IsaacLab's own API surface.
3. Pay special attention to the **real-drone telemetry boundary**: the Crazyflie firmware exposes
   named log fields `stateEstimate.qw/qx/qy/qz` via `cflib`. These are read by name (not by index),
   so they are unaffected by IsaacLab's convention change — but the in-memory tensor built from them
   (`current_quat` in `common/utils.py`) is currently WXYZ to match the simulation side. If the
   simulation side moves to XYZW while this callback keeps building `[qw, qx, qy, qz]`, an explicit
   conversion boundary must be added between real-drone telemetry and any code shared with
   simulation (e.g. `common/utils.py`'s `quat_apply`, `flight_logger.py`, `flight_recorder.py`).
4. Update this document once the migration is complete, or delete it if WXYZ/XYZW tracking moves
   somewhere else.
