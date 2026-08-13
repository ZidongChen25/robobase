#!/bin/bash
# Queue v6: Wave-7 follow-on — fence-v2 s3/s4, recovery-teach s1/s2.
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

FV2=exp_local/cqn_rline/inject_fence_v2_moveplate/replay
RT=exp_local/cqn_rline/inject_recovteach_v2_moveplate/replay

# job spec: seed stamp source-dir
JOBS=(
  "3 20260813fv2 $FV2"
  "4 20260813fv2 $FV2"
  "1 20260813rt $RT"
  "2 20260813rt $RT"
)

next=0
while [ $next -lt ${#JOBS[@]} ]; do
  for g in 0 1 2 3 4 5; do
    [ $next -ge ${#JOBS[@]} ] && break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$mem" -lt 4000 ]; then
      set -- ${JOBS[$next]}
      SEED=$1; STAMP=$2; SRC=$3
      RUN=exp_local/cqn_trunc_arms/cfaug_move_plate/seed${SEED}_${STAMP}
      if [ -e "$RUN" ]; then
        echo "[v6] SKIP seed$SEED $STAMP: dir exists"
        next=$((next+1)); continue
      fi
      mkdir -p "$RUN/replay"
      cp "$SRC"/*.npz "$RUN/replay/"
      nohup bash scripts/run_cqn_trunc_arm.sh cfaug "${UUID[$g]}" "${EGL[$g]}" \
        "$SEED" "$STAMP" move_plate 0 \
        > "exp_local/cqn_rline/q6_${STAMP}_s${SEED}.log" 2>&1 &
      echo "[v6] seed$SEED $STAMP -> GPU$g ($(date +%H:%M:%S))"
      next=$((next+1))
      sleep 300
    fi
  done
  sleep 300
done
echo "[v6] all dispatched ($(date +%H:%M:%S))"
