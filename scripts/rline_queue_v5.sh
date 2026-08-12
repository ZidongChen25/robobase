#!/bin/bash
# Queue v5: remaining Wave-5/6 arms. Fixes v4's two failure modes:
# no eval (quoting preserved via arrays), fresh stamps (no dir reuse).
set -u
cd /home/zc1525/robobase_jaxflat

declare -A UUID=(
  [0]="GPU-79eb6469-87b1-9dd3-ca2d-da93a916e919"
  [1]="GPU-ce804993-c33e-3d10-5676-5bae093a7d96"
  [2]="GPU-80b9cc0d-df5c-be12-e848-042d37578544"
  [3]="GPU-03f1431f-36c0-b258-6ca1-05007175e3eb"
  [4]="GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08"
  [5]="GPU-2f044e6a-9150-0e30-7d97-009bdd425b11"
)
declare -A EGL=([0]=4 [1]=5 [2]=2 [3]=3 [4]=0 [5]=1)

ZSCHED="method.bc_lambda_schedule='step_linear(1.0,0.0125,50000,0.0,20000)'"
ASCHED="method.bc_lambda_schedule='linear(1.0,0.0125,50000)'"
V1=exp_local/cqn_rline/counterfactual_episodes_v1

launch_job() {  # $1 job-index  $2 gpu-index
  local g=$2 u=${UUID[$2]} e=${EGL[$2]}
  case $1 in
    0)  # zero70 s2
      local run=exp_local/cqn_trunc_arms/cfaug_move_plate/seed2_20260812zero70b
      [ -e "$run" ] && { echo "SKIP job0: $run exists"; return 1; }
      mkdir -p "$run/replay"
      ls $V1/*.npz | head -120 | xargs -I{} cp {} "$run/replay/"
      nohup bash scripts/run_cqn_trunc_arm.sh cfaug "$u" "$e" 2 20260812zero70b \
        move_plate 0 "$ZSCHED" \
        > exp_local/cqn_rline/q4_20260812zero70b_s2.log 2>&1 &
      ;;
    1)  # anneal s2
      local run=exp_local/cqn_trunc_arms/cfaug_move_plate/seed2_20260812annealb
      [ -e "$run" ] && { echo "SKIP job1: $run exists"; return 1; }
      mkdir -p "$run/replay"
      ls $V1/*.npz | head -120 | xargs -I{} cp {} "$run/replay/"
      nohup bash scripts/run_cqn_trunc_arm.sh cfaug "$u" "$e" 2 20260812annealb \
        move_plate 0 "$ASCHED" \
        > exp_local/cqn_rline/q4_20260812annealb_s2.log 2>&1 &
      ;;
    2)  # combined s2, fresh stamp
      local run=exp_local/cqn_trunc_arms/combined_flip_cup/seed2_20260812combfc2
      [ -e "$run" ] && { echo "SKIP job2: $run exists"; return 1; }
      nohup bash scripts/run_cqn_trunc_arm.sh combined "$u" "$e" 2 20260812combfc2 \
        flip_cup 560 env.append_floating_base_to_low_dim=false \
        > exp_local/cqn_rline/q4_20260812combfc2_s2.log 2>&1 &
      ;;
  esac
  echo "[v5] job $1 -> GPU$g ($(date +%H:%M:%S))"
  return 0
}

next=0
while [ $next -lt 3 ]; do
  for g in 0 1 2 3 4 5; do
    [ $next -ge 3 ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 4000 ]; then
      if launch_job $next $g; then
        next=$((next+1))
        sleep 300  # let the new run claim VRAM before rescanning
      else
        next=$((next+1))  # skip-guard hit; move on
      fi
    fi
  done
  sleep 300
done
echo "[v5] all jobs dispatched ($(date +%H:%M:%S))"
