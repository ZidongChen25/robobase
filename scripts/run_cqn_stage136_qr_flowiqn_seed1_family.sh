#!/usr/bin/env bash
set -euo pipefail

# Full seed1 objective-family training.  This script is intentionally not part
# of the automatic launcher: Stage-135 must first be measured, reported, and
# acknowledged by a `reported` marker.  Stage-136 stops after training-health
# artifacts; family task selection is the subsequent reported stage.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_smoke_master
stage135_control=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_smoke_controller
control=exp_local/cqn_flow_high_utd/stage136_qr_flowiqn_seed1_family_controller

anchor_smoke=exp_local/cqn_flow_high_utd/stage135_flowiqn_anchor_only_smoke_seed1_20260724
equal_smoke=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_equal_smoke_seed2_20260724
ratio_smoke=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_dbc_ratio_smoke_seed3_20260724

anchor_run=exp_local/cqn_flow_high_utd/stage136_flowiqn_anchor_only_utd4_seed1_20260724
equal_run=exp_local/cqn_flow_high_utd/stage136_qr_flowiqn_equal_utd4_seed1_20260724
ratio_run=exp_local/cqn_flow_high_utd/stage136_qr_flowiqn_dbc_ratio_utd4_seed1_20260724

mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$stage135_control/reported"
test "$(jq -r .gate "$stage135_control/summary.json")" = pass

eta_json=$(
  .venv/bin/python - \
    "$anchor_smoke/train.csv" \
    "$equal_smoke/train.csv" \
    "$ratio_smoke/train.csv" <<'PY'
import csv
import json
import sys

estimates = []
for path in sys.argv[1:]:
    rows = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append(
                    (float(row["iteration"]), float(row["total_time"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not rows or rows[-1][0] <= 0:
        raise SystemExit(f"no throughput row in {path}")
    iteration, total_time = rows[-1]
    estimates.append(total_time * 10500.0 / iteration)
payload = {
    "per_arm_seconds": {
        "anchor_only": estimates[0],
        "joint_equal": estimates[1],
        "dbc_ratio": estimates[2],
    },
    "estimated_wall_seconds": max(estimates[0], estimates[1]) + estimates[2],
    "schedule": "anchor/equal parallel; ratio starts on first released GPU",
}
print(json.dumps(payload))
PY
)
jq -n \
  --argjson eta "$eta_json" \
  '{
    status: "running",
    stage: 136,
    selection_use_forbidden: true,
    training_seed: 1,
    num_train_frames: 10500,
    eta: $eta,
    next_stage:
      "fresh 210000/211000/212000 family selection after result report"
  }' > "$control/preregistration.json"

train_one() {
  local launch=$1
  local gpu=$2
  local destination=$3
  local log=$4
  if ! test -f "$destination/snapshots/10000_snapshot.pkl"; then
    MUJOCO_GL=egl .venv/bin/python train.py \
      "launch=$launch" env=bigym/move_plate \
      seed=1 "gpu_id=$gpu" num_train_frames=10500 \
      wandb.use=false save_csv=true log_eval_video=false \
      "hydra.run.dir=$destination" \
      > "$log" 2>&1
  fi
}

check_one() {
  local arm=$1
  local destination=$2
  local output=$3
  .venv/bin/python scripts/check_cqn_qr_flowiqn_training_gate.py \
    --run-dir "$destination" \
    --output "$output" \
    --expected-arm "$arm" \
    --required-snapshot-step 10000 \
    --min-log-rows 10
}

train_one \
  cqn_flowiqn_bc_target_two_tower_high_utd4_gate \
  1 "$anchor_run" "$control/anchor_only.log" &
anchor_pid=$!
train_one \
  cqn_qr_flowiqn_equal_bc_target_two_tower_high_utd4_gate \
  5 "$equal_run" "$control/joint_equal.log" &
equal_pid=$!
printf "%s\n" "$anchor_pid" > "$control/anchor_only.pid"
printf "%s\n" "$equal_pid" > "$control/joint_equal.pid"

finished_pid=
if ! wait -n -p finished_pid "$anchor_pid" "$equal_pid"; then
  touch "$control/first_parallel_arm_failed"
  exit 1
fi
if test "$finished_pid" = "$anchor_pid"; then
  ratio_gpu=1
  remaining_pid=$equal_pid
  touch "$control/anchor_only_training_complete"
else
  ratio_gpu=5
  remaining_pid=$anchor_pid
  touch "$control/joint_equal_training_complete"
fi

train_one \
  cqn_qr_flowiqn_dbc_ratio_bc_target_two_tower_high_utd4_gate \
  "$ratio_gpu" "$ratio_run" "$control/dbc_ratio.log" &
ratio_pid=$!
printf "%s\n" "$ratio_pid" > "$control/dbc_ratio.pid"
printf "%s\n" "$ratio_gpu" > "$control/dbc_ratio.gpu"

wait "$remaining_pid"
if test "$remaining_pid" = "$anchor_pid"; then
  touch "$control/anchor_only_training_complete"
else
  touch "$control/joint_equal_training_complete"
fi
wait "$ratio_pid"
touch "$control/dbc_ratio_training_complete"

gate_status=0
if ! check_one \
  anchor_only "$anchor_run" "$control/anchor_only_gate.json"; then
  gate_status=1
fi
if ! check_one \
  joint_equal "$equal_run" "$control/joint_equal_gate.json"; then
  gate_status=1
fi
if ! check_one \
  dbc_ratio "$ratio_run" "$control/dbc_ratio_gate.json"; then
  gate_status=1
fi

jq -n \
  --slurpfile preregistration "$control/preregistration.json" \
  --slurpfile anchor "$control/anchor_only_gate.json" \
  --slurpfile equal "$control/joint_equal_gate.json" \
  --slurpfile ratio "$control/dbc_ratio_gate.json" \
  '{
    status: "ok",
    stage: 136,
    preregistration: $preregistration[0],
    selection_use_forbidden: true,
    arms: {
      anchor_only: $anchor[0],
      joint_equal: $equal[0],
      dbc_ratio: $ratio[0]
    },
    gate: (
      if (
        $anchor[0].gate == "pass"
        and $equal[0].gate == "pass"
        and $ratio[0].gate == "pass"
      ) then "pass" else "fail" end
    ),
    next_gate_if_pass:
      "report health, then launch common-split family evaluator",
    next_gate_if_fail:
      "report and diagnose the failed arm before any task evaluation"
  }' > "$control/summary.json"

if test "$gate_status" -eq 0; then
  touch "$control/family_eval_ready"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
