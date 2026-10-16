#!/bin/bash
# Build an overlay directory with the gym-era D4RL stack (gym 0.23.1,
# mujoco-py 2.1.2.14, d4rl) for the RoboBase venv, without touching the venv,
# and write <overlay>/env.sh to source before running gym-era D4RL tasks.
#
#   bash scripts/setup_legacy_d4rl_overlay.sh ~/tmp/rlpd_d4rl/pyoverlay
#   source ~/tmp/rlpd_d4rl/pyoverlay/env.sh
#   python train_rlpd.py launch=rlpd_d4rl_antmaze seed=0
#
# Needs MuJoCo 2.1 in ~/.mujoco/mujoco210, EGL headers (/usr/include/EGL) and
# an NVIDIA driver (mujoco-py's EGL builder).  No root access is required.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OVERLAY=$(realpath -m "$1")
PY="$ROOT/.venv/bin/python"
mkdir -p "$OVERLAY"
uv pip install --python "$PY" --target "$OVERLAY" --no-deps \
  "gym==0.23.1" gym-notices "mujoco-py==2.1.2.14" "cython<3" "fasteners==0.15" \
  monotonic patchelf "git+https://github.com/Farama-Foundation/D4RL@master"

# Hosts with the GLEW runtime but no GLEW dev headers: mujoco-py's EGL shim
# only needs these declarations; MuJoCo 2.1 ships the implementation
# (libglewegl.so).
mkdir -p "$OVERLAY/mujoco_py/vendor/egl/GL"
if [ ! -f /usr/include/GL/glew.h ]; then
cat > "$OVERLAY/mujoco_py/vendor/egl/GL/glew.h" <<'EOF'
/* Minimal GLEW declarations for mujoco_py/gl/eglshim.c (implementation:
 * MuJoCo 2.1's libglewegl.so). Only the symbols eglshim.c uses. */
#ifndef __glew_h__
#define __glew_h__
#include <GL/gl.h>
#include <GL/glext.h>
#define GLEW_OK 0
#define GLEWAPI extern
GLEWAPI GLenum glewInit(void);
GLEWAPI const GLubyte *glewGetErrorString(GLenum error);
GLEWAPI PFNGLBINDBUFFERPROC __glewBindBuffer;
GLEWAPI PFNGLBINDFRAMEBUFFERPROC __glewBindFramebuffer;
GLEWAPI PFNGLBLITFRAMEBUFFERPROC __glewBlitFramebuffer;
GLEWAPI PFNGLBUFFERDATAPROC __glewBufferData;
GLEWAPI PFNGLDELETEBUFFERSPROC __glewDeleteBuffers;
GLEWAPI PFNGLGENBUFFERSPROC __glewGenBuffers;
GLEWAPI PFNGLMAPBUFFERARBPROC __glewMapBufferARB;
GLEWAPI PFNGLUNMAPBUFFERPROC __glewUnmapBuffer;
#define glBindBuffer __glewBindBuffer
#define glBindFramebuffer __glewBindFramebuffer
#define glBlitFramebuffer __glewBlitFramebuffer
#define glBufferData __glewBufferData
#define glDeleteBuffers __glewDeleteBuffers
#define glGenBuffers __glewGenBuffers
#define glMapBufferARB __glewMapBufferARB
#define glUnmapBuffer __glewUnmapBuffer
#endif
EOF
fi

cat > "$OVERLAY/env.sh" <<EOF
export PYTHONPATH=$OVERLAY\${PYTHONPATH:+:\$PYTHONPATH}
export PATH=$OVERLAY/bin:\$PATH
export MUJOCO_PY_MUJOCO_PATH=\$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=\$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF

# First import compiles mujoco-py's Cython extension into the overlay.
source "$OVERLAY/env.sh"
"$PY" -c "import mujoco_py, gym, d4rl; gym.make('antmaze-umaze-v2'); print('legacy D4RL overlay ready')"
