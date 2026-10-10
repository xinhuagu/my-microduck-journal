#!/usr/bin/env bash
# Drive a policy in the CPU MuJoCo viewer under WSLg, rendering on the NVIDIA GPU.
#
#   scripts/run_viewer.sh <walk.onnx> [extra infer_policy.py args...]
#   e.g. scripts/run_viewer.sh exports/first_walk_10000.onnx
#        scripts/run_viewer.sh walk.onnx --roulade roulade.onnx
#
# Run from the microduck_rl checkout, in a real terminal (Windows Terminal works; some embedded
# terminals swallow the arrow keys). Keys are read from THIS terminal, not the MuJoCo window.
# If the window is blank and its title starts with "[WARN:COPY MODE]", run `wsl --shutdown` from
# PowerShell and try again. Set RENDER=cpu to force llvmpipe software rendering.
set -u
POLICY="${1:?path to walking policy .onnx}"; shift
if [ "${RENDER:-gpu}" = "gpu" ]; then
  export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12
fi
exec uv run scripts/infer_policy.py --walking "$POLICY" --new-cmd-obs "$@"
