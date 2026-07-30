#!/usr/bin/env bash
set -euo pipefail

# Full matched seed1 control/treatment training. Launch only after the Stage-138
# identity result has been reported; stop at training health so causal
# discovery is a separately reported stage.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage138_frozen_clean_rct_smoke_master
upstream_control=exp_local/cqn_flow_high_utd/stage138_frozen_clean_rct_smoke_controller
control=exp_local/cqn_flow_high_utd/stage139_frozen_rct_seed1_controller

clean=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724
source="$clean/snapshots/5000_snapshot.pkl"
control_run=exp_local/cqn_flow_high_utd/stage139_frozen_clean_control_utd4_seed1_20260724
treatment_run=exp_local/cqn_flow_high_utd/stage139_frozen_clean_rct_utd4_seed1_20260724

mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$upstream_control/reported"
test "$(jq -r .gate "$upstream_control/summary.json")" = pass
test -f "$source"

.venv/bin/python - \
  "$upstream_control" \
  exp_local/cqn_flow_high_utd/stage85_direct_q_replay_utd4_seed3_20260724/train.csv \
  "$control/preregistration.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

smoke_control = Path(sys.argv[1])
reference_csv = Path(sys.argv[2])
output = Path(sys.argv[3])

def final_row(path):
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append(
                    {
                        "iteration": float(row["iteration"]),
                        "total_time": float(row["total_time"]),
                        "backend_update_time_sec": float(
                            row["backend_update_time_sec"]
                        ),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit(f"no throughput row: {path}")
    return rows[-1]

smoke = {
    label: final_row(
        smoke_control.parent
        / f"stage138_frozen_clean_rct_{label}_20260724"
        / "train.csv"
    )
    for label in ("seed1", "seed2")
}
reference = final_row(reference_csv)
backend_ratio = max(
    row["backend_update_time_sec"] for row in smoke.values()
) / max(reference["backend_update_time_sec"], 1e-12)
estimate = reference["total_time"] * max(1.0, backend_ratio)
payload = {
    "status": "running",
    "stage": 139,
    "research_route": "A",
    "training_seed": 1,
    "endpoint_step": 10000,
    "checkpoint_selection": "none; endpoint preregistered",
    "matched_difference": "causal_rct_weight 0.0 versus 0.1 only",
    "reference_direct_q_10k_seconds": reference["total_time"],
    "observed_frozen_to_reference_backend_ratio": backend_ratio,
    "estimated_parallel_wall_seconds": estimate,
    "next_gate": "fresh paired H1 round-robin causal discovery",
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

train_one() {
  local launch=$1
  local gpu=$2
  local destination=$3
  local log=$4
  if ! test -f "$destination/snapshots/10000_snapshot.pkl"; then
    MUJOCO_GL=egl .venv/bin/python train.py \
      "launch=$launch" env=bigym/move_plate \
      seed=1 "gpu_id=$gpu" num_train_frames=10500 \
      "method.frozen_policy_snapshot=$source" \
      wandb.use=false save_csv=true log_eval_video=false \
      "hydra.run.dir=$destination" \
      > "$log" 2>&1
  fi
}

train_one \
  cqn_direct_q_h1_rct_frozen_clean_control_high_utd4_gate \
  1 "$control_run" "$control/control.log" &
control_pid=$!
train_one \
  cqn_direct_q_h1_rct_frozen_clean_high_utd4_gate \
  5 "$treatment_run" "$control/treatment.log" &
treatment_pid=$!
printf "%s\n" "$control_pid" > "$control/control.pid"
printf "%s\n" "$treatment_pid" > "$control/treatment.pid"

status=0
if wait "$control_pid"; then
  touch "$control/control_training_complete"
else
  status=1
fi
if wait "$treatment_pid"; then
  touch "$control/treatment_training_complete"
else
  status=1
fi
test "$status" -eq 0

check_one() {
  local run=$1
  local weight=$2
  local output=$3
  .venv/bin/python scripts/check_cqn_direct_q_rct_training_gate.py \
    --run-dir "$run" \
    --output "$output" \
    --expected-causal-rct-weight "$weight" \
    --expected-exploration-prob 0.2 \
    --expected-level 1 \
    --required-snapshot-step 10000 \
    --min-log-rows 10 \
    --min-online-starts 200 \
    --min-starts-per-dimension 5 \
    --expected-frozen-policy-snapshot "$source"
}

check_one "$control_run" 0 "$control/control_health.json"
check_one "$treatment_run" 0.1 "$control/treatment_health.json"

jq -n \
  --slurpfile preregistration "$control/preregistration.json" \
  --slurpfile control_health "$control/control_health.json" \
  --slurpfile treatment_health "$control/treatment_health.json" \
  '{
    status: "ok",
    stage: 139,
    preregistration: $preregistration[0],
    control: $control_health[0],
    treatment: $treatment_health[0],
    gate: (
      if (
        $control_health[0].gate == "pass"
        and $treatment_health[0].gate == "pass"
      ) then "pass" else "fail" end
    ),
    next_gate_if_pass:
      "report, then run paired seed1 H1 round-robin causal discovery",
    next_gate_if_fail:
      "report and repair training or frozen-policy identity before causal probing"
  }' > "$control/summary.json"

if test "$(jq -r .gate "$control/summary.json")" = pass; then
  touch "$control/causal_discovery_ready"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
