#!/usr/bin/env bash
set -euo pipefail

# Run the preregistered all-pairs quantile + sorted-CFM smoke on GPU1 while the
# upstream chain occupies GPU5. The main Stage-135 controller reuses the
# resulting snapshot and still owns the final three-arm stage summary.

cd /home/zc1525/robobase_jaxflat

control=exp_local/cqn_flow_high_utd/stage135_joint_equal_prefill_controller
destination=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_equal_smoke_seed2_20260724
health="$control/joint_equal_health.json"

mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

if ! test -f "$destination/snapshots/1000_snapshot.pkl"; then
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_qr_flowiqn_equal_bc_target_two_tower_high_utd4_gate \
    env=bigym/move_plate seed=2 gpu_id=1 num_train_frames=1500 \
    wandb.use=false save_csv=true log_eval_video=false \
    "hydra.run.dir=$destination" \
    > "$control/train.log" 2>&1
fi

.venv/bin/python scripts/check_cqn_qr_flowiqn_training_gate.py \
  --run-dir "$destination" \
  --output "$health" \
  --expected-arm joint_equal \
  --required-snapshot-step 1000 \
  --min-log-rows 2 \
  > "$control/check.log" 2>&1

test "$(jq -r .gate "$health")" = pass
touch "$control/complete"
