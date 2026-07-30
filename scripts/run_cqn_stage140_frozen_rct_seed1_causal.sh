#!/usr/bin/env bash
set -euo pipefail

# Seed1 causal mechanism discovery on fresh simulator branches. This is a
# promotion screen only; final claims require three training seeds, a new
# sealed split, strict crossed CIs, and the separate native calibration gate.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage139_frozen_rct_seed1_master
upstream_control=exp_local/cqn_flow_high_utd/stage139_frozen_rct_seed1_controller
control=exp_local/cqn_flow_high_utd/stage140_frozen_rct_seed1_causal_controller

control_run=exp_local/cqn_flow_high_utd/stage139_frozen_clean_control_utd4_seed1_20260724
treatment_run=exp_local/cqn_flow_high_utd/stage139_frozen_clean_rct_utd4_seed1_20260724
output=exp_local/cqn_flow_high_utd/stage140_frozen_rct_seed1_causal_seed214000_20260724
control_probe="$output/control/probe.json"
treatment_probe="$output/treatment/probe.json"

mkdir -p "$control" "$output/control" "$output/treatment"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$upstream_control/reported"
test "$(jq -r .gate "$upstream_control/summary.json")" = pass

jq -n \
  '{
    status: "running",
    stage: 140,
    claim_scope: "seed1_discovery_only",
    endpoint_step: 10000,
    checkpoint_selection: "none",
    eval_seed_start: 214000,
    num_eval_seeds: 12,
    anchor_steps: [30, 75, 120],
    intervention_horizon: 1,
    dimension_selection: "round_robin",
    continuation_policy: "frozen clean CQN-AS, value beta null",
    estimated_parallel_wall_seconds: 360,
    promotion_gate:
      "exact matched outcomes, treatment causal direction positive, treatment pairwise improves control, and treatment beats all imitation proxies"
  }' > "$control/preregistration.json"

seeds=$(seq -s, 214000 214011)
probe_one() {
  local gpu=$1
  local run=$2
  local output_json=$3
  local log=$4
  MUJOCO_GL=egl .venv/bin/python \
    scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "$run" \
    --snapshot "$run/snapshots/10000_snapshot.pkl" \
    --output "$output_json" \
    --gpu-id "$gpu" \
    --eval-seeds "$seeds" \
    --anchor-steps 30,75,120 \
    --force-level 1 \
    --dimension-selection round_robin \
    --intervention-mode sibling_horizon \
    --intervention-horizon 1 \
    --max-continuation-steps 300 \
    --flow-readout auto \
    --policy-value-beta bc \
    --bootstrap-replicates 2000 \
    --probe-seed 214100 \
    > "$log" 2>&1
}

probe_one 1 "$control_run" "$control_probe" "$control/control_probe.log" &
control_pid=$!
probe_one 5 "$treatment_run" "$treatment_probe" \
  "$control/treatment_probe.log" &
treatment_pid=$!
printf "%s\n" "$control_pid" > "$control/control_probe.pid"
printf "%s\n" "$treatment_pid" > "$control/treatment_probe.pid"

status=0
if wait "$control_pid"; then
  touch "$control/control_probe_complete"
else
  status=1
fi
if wait "$treatment_pid"; then
  touch "$control/treatment_probe_complete"
else
  status=1
fi
test "$status" -eq 0

gate_status=0
if ! .venv/bin/python scripts/summarize_cqn_paired_causal_arms.py \
  --pair "seed1=$control_probe,$treatment_probe" \
  --output "$output/summary.json" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 214200 \
  --min-training-seeds 1 \
  --min-eval-seeds 12 \
  --min-informative-states 12 \
  --required-positive-training-seeds 1; then
  gate_status=1
fi

jq -n \
  --slurpfile preregistration "$control/preregistration.json" \
  --slurpfile result "$output/summary.json" \
  '{
    status: "ok",
    stage: 140,
    preregistration: $preregistration[0],
    result: $result[0],
    gate: $result[0].gate,
    next_gate_if_pass:
      "report, then train matched frozen-policy control/RCT seeds2/3",
    next_gate_if_fail:
      "report and diagnose value target/objective before multiseed replication"
  }' > "$control/summary.json"

if test "$gate_status" -eq 0; then
  touch "$control/multiseed_replication_ready"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
