# Policy Evaluation Tools

Offline evaluation of trained policies: metrics and figures computed from recorded flights or from
simulated rollouts. Nothing in this directory flies a drone. The layout mirrors
`execution/single_drone_exec/`, so each tool sits next to the flight script it evaluates.

## Single drone, hover (`single_drone_exec/hover/`)
- **`analyze_vel.py`**: Hover metrics of flights recorded by `execution/single_drone_exec/hover/exec_vel.py`
  (position error, jitter, drift, settling time, command smoothness, motor PWM, crash/success).
  Needs numpy and pyarrow; matplotlib only for `--plot`.

```bash
# Run from CrazyPlayGround/
python sim_eval/single_drone_exec/hover/analyze_vel.py flight.parquet
python sim_eval/single_drone_exec/hover/analyze_vel.py logs/*.parquet --skip-s 3 --json-out report.json
```

System identification (step excitation, delay and noise) lives in [`../sim2real/`](../sim2real/README.md).
