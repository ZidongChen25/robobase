#!/usr/bin/env bash
set -euo pipefail

# Common-split seed1 family gate.  Launch only after Stage-136 health has been
# reported.  The evaluator selects every arm under the same budget, freezes
# one treatment on validation, and opens sealed confirmation exactly once.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage136_qr_flowiqn_seed1_family_master
stage136_control=exp_local/cqn_flow_high_utd/stage136_qr_flowiqn_seed1_family_controller
control=exp_local/cqn_flow_high_utd/stage137_qr_flowiqn_family_gate_controller

anchor_run=exp_local/cqn_flow_high_utd/stage136_flowiqn_anchor_only_utd4_seed1_20260724
equal_run=exp_local/cqn_flow_high_utd/stage136_qr_flowiqn_equal_utd4_seed1_20260724
ratio_run=exp_local/cqn_flow_high_utd/stage136_qr_flowiqn_dbc_ratio_utd4_seed1_20260724
clean_run=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724
clean_snapshot="$clean_run/snapshots/5000_snapshot.pkl"

output=exp_local/cqn_flow_high_utd/stage137_qr_flowiqn_family_gate_seed210000_20260724
work=exp_local/cqn_flow_high_utd/stage137_qr_flowiqn_family_work_seed210000_20260724

mkdir -p "$control" "$output" "$work"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$stage136_control/reported"
test "$(jq -r .gate "$stage136_control/summary.json")" = pass

# Stage-67 used the same two-GPU process-per-checkpoint evaluator and took
# 2977.43 s for 42 jobs. This family gate has 51 jobs, giving 3615.45 s.
jq -n \
  '{
    status: "running",
    stage: 137,
    prior_elapsed_seconds: 2977.432200908661,
    prior_eval_jobs: 42,
    current_eval_jobs: 51,
    estimated_wall_seconds: (
      2977.432200908661 * 51 / 42
    ),
    screen: {seed_start: 210000, episodes: 10},
    validation: {seed_start: 211000, episodes: 50},
    confirmation: {seed_start: 212000, episodes: 200},
    selection_use_forbidden_after_confirmation: true
  }' > "$control/preregistration.json"

MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_qr_flowiqn_family_gate.py \
  --anchor "anchor_only=$anchor_run" \
  --treatment "joint_equal=$equal_run" \
  --treatment "dbc_ratio=$ratio_run" \
  --clean-run-dir "$clean_run" \
  --clean-snapshot "$clean_snapshot" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$output" \
  --work-root "$work" \
  --checkpoint-step 1000 2000 3000 4000 5000 \
    6000 7000 8000 9000 10000 \
  --screen-top-k 2 \
  --beta 0.3 1 3 \
  --screen-beta 1 \
  --num-flow-steps 8 \
  --num-action-flow-samples 8 \
  --screen-episodes 10 \
  --screen-seed-start 210000 \
  --validation-episodes 50 \
  --validation-seed-start 211000 \
  --confirmation-episodes 200 \
  --confirmation-seed-start 212000 \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 212200 \
  --clean-min-ci-lower -0.05 \
  > "$control/gate.log" 2>&1

test "$(jq -r .status "$output/summary.json")" = ok
if test "$(jq -r .gate "$output/summary.json")" = pass; then
  touch "$control/multiseed_training_ready"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
