#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/turn_sweep.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Walk sweep over checkpoints of a VelStand run: export each, then measure straight-walk yaw drift at
# 0.30 m/s and in-place turning at 0.6 rad/s (the two numbers that differ most between checkpoints).
#   RUNDIR=logs/rsl_rl/velstand/<run> PREFIX=velstand_v2 TASK=Mjlab-VelStand-Flat-MicroDuck turn_sweep.sh 5000 5250 5500 5750
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/microduck_rl" || exit 1
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNDIR="${RUNDIR:?run dir}"; PREFIX="${PREFIX:-velstand}"; TASK="${TASK:-Mjlab-VelStand-Flat-MicroDuck}"
EXPERT="${EXPERT:-logs/rsl_rl/microduck_stand/2026-10-11_08-37-10_microduck_stand/model_7498.pt}"
ANCHOR="${ANCHOR:-logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt}"
CASES="${CASES:-walk0.30 turn0.6}"
for it in "$@"; do
  f="exports/${PREFIX}_${it}.onnx"
  if [ ! -s "$f" ]; then
    MICRODUCK_STAND_EXPERT="$EXPERT" MICRODUCK_WALK_ANCHOR="$ANCHOR" \
    uv run scripts/export.py "$TASK" --checkpoint-file "$RUNDIR/model_${it}.pt" --onnx-file "$f" > "/tmp/export_sweep_${it}.log" 2>&1 \
      && echo "exported $it" || { echo "export FAILED for $it"; tail -2 "/tmp/export_sweep_${it}.log"; continue; }
  fi
  echo "--- checkpoint $it"
  CASES="$CASES" bash "$S/turn_probe.sh" "$f" --walking 12 | grep -E "^(turn|walk) "
done
