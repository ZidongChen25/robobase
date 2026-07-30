#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[162r-retry] probing EGL every 300s"
while true; do
  if MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
    .venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>')
mujoco.Renderer(m, 84, 84)" > /dev/null 2>&1; then
    echo "[162r-retry] EGL recovered, resuming"
    break
  fi
  sleep 300
done
exec bash scripts/resume_cqn_stage162_double.sh
