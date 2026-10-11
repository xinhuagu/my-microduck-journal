#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/train_standup_teacher_v2_straight.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Stand-up teacher v2: warm start from model_5999 with the stand-straight tweak, 1500 iterations max.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/microduck_rl" || exit 1
MAXIT="${1:-1500}"
LOG=logs/standup_straight_train.log
{
  echo "=== $(date -Is) StandUp v2 (stand straight): warm start from microduck_stand/model_5999.pt, max_iterations=$MAXIT"
  MICRODUCK_WARM_START=1 MICRODUCK_STANDUP_STRAIGHT=1 \
  uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.resume True --agent.load-run 2026-10-10_23-57-42_microduck_stand --agent.load-checkpoint model_5999.pt \
    --agent.max-iterations "$MAXIT"
  echo "=== $(date -Is) training exited with $?"
} 2>&1 | tee -a "$LOG"
echo "Training finished. Press Enter to close."; read -r _
