#!/usr/bin/env bash
set -euo pipefail

# Final preregistered Stage-135 smoke: DBC-like small sorted-CFM anchor plus
# unit-weight all-pairs quantile endpoint loss. The main Stage-135 controller
# reuses this artifact when it reaches the three-arm health summary.

cd /home/zc1525/robobase_jaxflat

control=exp_local/cqn_flow_high_utd/stage135_dbc_ratio_prefill_controller
destination=exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_dbc_ratio_smoke_seed3_20260724
health="$control/dbc_ratio_health.json"

mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

if ! test -f "$destination/snapshots/1000_snapshot.pkl"; then
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_qr_flowiqn_dbc_ratio_bc_target_two_tower_high_utd4_gate \
    env=bigym/move_plate seed=3 gpu_id=1 num_train_frames=1500 \
    wandb.use=false save_csv=true log_eval_video=false \
    "hydra.run.dir=$destination" \
    > "$control/train.log" 2>&1
fi

.venv/bin/python scripts/check_cqn_qr_flowiqn_training_gate.py \
  --run-dir "$destination" \
  --output "$health" \
  --expected-arm dbc_ratio \
  --required-snapshot-step 1000 \
  --min-log-rows 2 \
  > "$control/check.log" 2>&1

test "$(jq -r .gate "$health")" = pass
touch "$control/complete"
