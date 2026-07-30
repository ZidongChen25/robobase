#!/usr/bin/env bash
set -euo pipefail

# Route-A corrective smoke after the current two-GPU chain releases both GPUs.
# The original CQN-AS target critic and image encoder are the behavior policy,
# protected bitwise. Only the independent direct-Q/value tower is optimized.
# This stage stops after health and paired closed-loop equivalence so its result
# is reported before any three-seed causal promotion.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_smoke_master
control=exp_local/cqn_flow_high_utd/stage138_frozen_clean_rct_smoke_controller

clean1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724
clean2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724
source1="$clean1/snapshots/5000_snapshot.pkl"
source2="$clean2/snapshots/5000_snapshot.pkl"

run1=exp_local/cqn_flow_high_utd/stage138_frozen_clean_rct_seed1_20260724
run2=exp_local/cqn_flow_high_utd/stage138_frozen_clean_rct_seed2_20260724

mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$source1"
test -f "$source2"

# Current direct-Q seed3 measured 152.49 s through step 1k. Scaling to 1.5k,
# adding two eight-episode paired evaluations and checkpoint audits gives a
# conservative 8-minute wall estimate once both GPUs are released.
jq -n \
  --arg upstream "$upstream_master" \
  '{
    status: "running",
    stage: 138,
    research_route: "A",
    hypothesis:
      "A direct causal value tower can learn without weakening CQN-AS behavior when the validation-best legacy-C51 policy and image encoder are imported and frozen exactly.",
    upstream_gpu_release_event: $upstream,
    training_seeds: [1, 2],
    num_train_frames: 1500,
    paired_eval: {
      episodes_per_seed: 8,
      seed_starts: [161000, 161100],
      exact_tolerance: 0
    },
    estimated_wall_seconds_after_gpu_release: 480,
    selection_use_forbidden: true,
    promotion_gate:
      "both health gates pass, policy/encoder remain bitwise identical with zero gradients, and both paired closed-loop equivalence gates pass exactly"
  }' > "$control/preregistration.json"

train_one() {
  local seed=$1
  local gpu=$2
  local source=$3
  local destination=$4
  local log=$5
  if ! test -f "$destination/snapshots/1000_snapshot.pkl"; then
    MUJOCO_GL=egl .venv/bin/python train.py \
      launch=cqn_direct_q_h1_rct_frozen_clean_high_utd4_gate \
      env=bigym/move_plate \
      "seed=$seed" "gpu_id=$gpu" num_train_frames=1500 \
      "method.frozen_policy_snapshot=$source" \
      wandb.use=false save_csv=true log_eval_video=false \
      "hydra.run.dir=$destination" \
      > "$log" 2>&1
  fi
}

train_one 1 1 "$source1" "$run1" "$control/seed1_train.log" &
pid1=$!
train_one 2 5 "$source2" "$run2" "$control/seed2_train.log" &
pid2=$!
printf "%s\n" "$pid1" > "$control/seed1_train.pid"
printf "%s\n" "$pid2" > "$control/seed2_train.pid"

status=0
if wait "$pid1"; then
  touch "$control/seed1_training_complete"
else
  status=1
fi
if wait "$pid2"; then
  touch "$control/seed2_training_complete"
else
  status=1
fi
test "$status" -eq 0

check_one() {
  local destination=$1
  local source=$2
  local output=$3
  .venv/bin/python scripts/check_cqn_direct_q_rct_training_gate.py \
    --run-dir "$destination" \
    --output "$output" \
    --expected-causal-rct-weight 0.1 \
    --expected-exploration-prob 0.2 \
    --expected-level 1 \
    --required-snapshot-step 1000 \
    --min-log-rows 2 \
    --min-online-starts 30 \
    --min-starts-per-dimension 1 \
    --expected-frozen-policy-snapshot "$source"
}

check_one "$run1" "$source1" "$control/seed1_health.json"
check_one "$run2" "$source2" "$control/seed2_health.json"
test "$(jq -r .gate "$control/seed1_health.json")" = pass
test "$(jq -r .gate "$control/seed2_health.json")" = pass
touch "$control/health_gate_complete"

eval_pair() {
  local gpu=$1
  local seed_start=$2
  local clean=$3
  local source=$4
  local candidate=$5
  local label=$6
  local clean_json="$control/${label}_clean.json"
  local candidate_json="$control/${label}_candidate.json"

  MUJOCO_GL=egl .venv/bin/python \
    scripts/eval_cqn_flow_policy_value.py \
    --run-dir "$clean" \
    --snapshot "$source" \
    --output "$clean_json" \
    --work-dir "$control/${label}_clean_work" \
    --gpu-id "$gpu" \
    --num-eval-episodes 8 \
    --eval-seed-start "$seed_start" \
    --policy-value-beta bc \
    > "$control/${label}_clean_eval.log" 2>&1

  MUJOCO_GL=egl .venv/bin/python \
    scripts/eval_cqn_flow_policy_value.py \
    --run-dir "$candidate" \
    --snapshot "$candidate/snapshots/1000_snapshot.pkl" \
    --output "$candidate_json" \
    --work-dir "$control/${label}_candidate_work" \
    --gpu-id "$gpu" \
    --num-eval-episodes 8 \
    --eval-seed-start "$seed_start" \
    --policy-value-beta bc \
    > "$control/${label}_candidate_eval.log" 2>&1

  .venv/bin/python scripts/summarize_cqn_behavior_equivalence.py \
    --reference "$clean_json" \
    --candidate "$candidate_json" \
    --output "$control/${label}_equivalence.json" \
    --atol 0
}

eval_pair 1 161000 "$clean1" "$source1" "$run1" seed1 &
eval1_pid=$!
eval_pair 5 161100 "$clean2" "$source2" "$run2" seed2 &
eval2_pid=$!
printf "%s\n" "$eval1_pid" > "$control/seed1_eval.pid"
printf "%s\n" "$eval2_pid" > "$control/seed2_eval.pid"

status=0
if wait "$eval1_pid"; then
  touch "$control/seed1_equivalence_complete"
else
  status=1
fi
if wait "$eval2_pid"; then
  touch "$control/seed2_equivalence_complete"
else
  status=1
fi
test "$status" -eq 0

jq -n \
  --slurpfile preregistration "$control/preregistration.json" \
  --slurpfile health1 "$control/seed1_health.json" \
  --slurpfile health2 "$control/seed2_health.json" \
  --slurpfile equivalence1 "$control/seed1_equivalence.json" \
  --slurpfile equivalence2 "$control/seed2_equivalence.json" \
  '{
    status: "ok",
    stage: 138,
    preregistration: $preregistration[0],
    health: {
      seed1: $health1[0],
      seed2: $health2[0]
    },
    closed_loop_equivalence: {
      seed1: $equivalence1[0],
      seed2: $equivalence2[0]
    },
    gate: (
      if (
        $health1[0].gate == "pass"
        and $health2[0].gate == "pass"
        and $equivalence1[0].gate == "pass"
        and $equivalence2[0].gate == "pass"
      ) then "pass" else "fail" end
    ),
    next_gate_if_pass:
      "report, then run seed3 plus three-seed H=1 round-robin causal value comparison against frozen no-RCT controls",
    next_gate_if_fail:
      "report and repair the exact import/freeze/action-parity mechanism before any causal claim"
  }' > "$control/summary.json"

if test "$(jq -r .gate "$control/summary.json")" = pass; then
  touch "$control/three_seed_causal_ready"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
