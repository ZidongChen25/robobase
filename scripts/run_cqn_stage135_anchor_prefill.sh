#!/usr/bin/env bash
set -euo pipefail

# Opportunistically fill the otherwise idle GPU1 with the already-preregistered
# Stage-135 anchor smoke. The main Stage-135 controller reuses its snapshot and
# remains responsible for the complete three-arm health result.

cd /home/zc1525/robobase_jaxflat

control=exp_local/cqn_flow_high_utd/stage135_anchor_prefill_controller
destination=exp_local/cqn_flow_high_utd/stage135_flowiqn_anchor_only_smoke_seed1_20260724
health="$control/anchor_only_health.json"

mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

if ! test -f "$destination/snapshots/1000_snapshot.pkl"; then
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_flowiqn_bc_target_two_tower_high_utd4_gate \
    env=bigym/move_plate seed=1 gpu_id=1 num_train_frames=1500 \
    wandb.use=false save_csv=true log_eval_video=false \
    "hydra.run.dir=$destination" \
    > "$control/train.log" 2>&1
fi

.venv/bin/python scripts/check_cqn_qr_flowiqn_training_gate.py \
  --run-dir "$destination" \
  --output "$health" \
  --expected-arm anchor_only \
  --required-snapshot-step 1000 \
  --min-log-rows 2 \
  > "$control/check.log" 2>&1

test "$(jq -r .gate "$health")" = pass
touch "$control/complete"
