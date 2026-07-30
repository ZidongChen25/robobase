#!/usr/bin/env bash
# Auto-retry launcher: waits until a fresh EGL context can be created,
# then starts the stage163c chain.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[163c-retry] probing EGL every 300s"
while true; do
  if MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \
    .venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>')
mujoco.Renderer(m, 84, 84)" > /dev/null 2>&1; then
    echo "[163c-retry] EGL recovered, launching"
    break
  fi
  sleep 300
done
exec bash scripts/run_cqn_stage163c_official_qc8.sh
