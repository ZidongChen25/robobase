#!/bin/bash
# Queue v7: anneal-A s3/s4 (extend to 4-seed claim gate after s2's 85.0).
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
ASCHED="method.bc_lambda_schedule='linear(1.0,0.0125,50000)'"
V1=exp_local/cqn_rline/counterfactual_episodes_v1
next=0
SEEDS=(3 4)
while [ $next -lt 2 ]; do
  for g in 0 1 2 3 4 5; do
    [ $next -ge 2 ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 4000 ]; then
      S=${SEEDS[$next]}
      RUN=exp_local/cqn_trunc_arms/cfaug_move_plate/seed${S}_20260813annealc
      [ -e "$RUN" ] && { echo "[v7] SKIP s$S: exists"; next=$((next+1)); continue; }
      mkdir -p "$RUN/replay"
      ls $V1/*.npz | head -120 | xargs -I{} cp {} "$RUN/replay/"
      nohup bash scripts/run_cqn_trunc_arm.sh cfaug "${UUID[$g]}" "${EGL[$g]}" \
        "$S" 20260813annealc move_plate 0 "$ASCHED" \
        > "exp_local/cqn_rline/q7_annealc_s${S}.log" 2>&1 &
      echo "[v7] anneal s$S -> GPU$g ($(date +%H:%M:%S))"
      next=$((next+1)); sleep 300
    fi
  done
  sleep 300
done
echo "[v7] all dispatched"
