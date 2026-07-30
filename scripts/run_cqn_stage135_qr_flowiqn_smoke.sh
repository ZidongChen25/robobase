#!/usr/bin/env bash
set -euo pipefail

# Route-B conditional continuation after the strict Stage-134 summary.  This
# stage only checks implementation/training health for a preregistered
# three-arm objective family.  It intentionally stops after writing the smoke
# result so that the measured result is reported before any full training gate.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage134_unbiased_dimension_confirmation_master
strict_summary=exp_local/cqn_flow_high_utd/stage134_strict_autoresearch_summary_20260724.json
control=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_smoke_controller

anchor_run=exp_local/cqn_flow_high_utd/stage135_flowiqn_anchor_only_smoke_seed1_20260724
equal_run=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_equal_smoke_seed2_20260724
ratio_run=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_dbc_ratio_smoke_seed3_20260724

mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$strict_summary"
test "$(jq -r .status "$strict_summary")" = ok

if test "$(jq -r .route_b.overall_gate "$strict_summary")" = pass; then
  jq -n \
    --arg upstream "$strict_summary" \
    '{
      status: "ok",
      gate: "skip",
      reason: "route_b_already_passed",
      upstream_summary: $upstream
    }' > "$control/summary.json"
  touch "$control/skipped_route_b_already_passed"
  touch "$control/complete"
  exit 0
fi

train_one() {
  local launch=$1
  local seed=$2
  local gpu=$3
  local destination=$4
  local log=$5
  if ! test -f "$destination/snapshots/1000_snapshot.pkl"; then
    MUJOCO_GL=egl .venv/bin/python train.py \
      "launch=$launch" env=bigym/move_plate \
      "seed=$seed" "gpu_id=$gpu" num_train_frames=1500 \
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
    --required-snapshot-step 1000 \
    --min-log-rows 2
}

train_one \
  cqn_flowiqn_bc_target_two_tower_high_utd4_gate \
  1 1 "$anchor_run" "$control/anchor_only.log" &
anchor_pid=$!
train_one \
  cqn_qr_flowiqn_equal_bc_target_two_tower_high_utd4_gate \
  2 5 "$equal_run" "$control/joint_equal.log" &
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
  3 "$ratio_gpu" "$ratio_run" "$control/dbc_ratio.log" &
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
  --arg upstream "$strict_summary" \
  --slurpfile anchor "$control/anchor_only_gate.json" \
  --slurpfile equal "$control/joint_equal_gate.json" \
  --slurpfile ratio "$control/dbc_ratio_gate.json" \
  '{
    status: "ok",
    stage: 135,
    research_question:
      "Does all-pairs endpoint quantile regression repair anchor-only FlowIQN?",
    upstream_summary: $upstream,
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
      "fresh seed1 family screen before any multiseed promotion",
    next_gate_if_fail:
      "diagnose the failing objective arm without task-quality claims"
  }' > "$control/summary.json"

if test "$gate_status" -eq 0; then
  touch "$control/promotion_ready"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
