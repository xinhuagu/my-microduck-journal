#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/train_velstand_v1.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# VelStand full training: warm start from first-walk model_10000, local stand-up expert + walk anchor.
#   train_velstand_v1.sh <standup_iter> [max_iterations]
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/microduck_rl" || exit 1
IT="${1:?stand-up checkpoint iteration}"; MAXIT="${2:-3000}"
STAND_RUN=$(ls -dt logs/rsl_rl/microduck_stand/*/ | head -1)
EXPERT="$STAND_RUN/model_${IT}.pt"; [ -f "$EXPERT" ] || { echo "missing $EXPERT"; exit 1; }
ANCHOR=logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt
LOG=logs/velstand_train.log
{
  echo "=== $(date -Is) VelStand training: expert=$EXPERT anchor=$ANCHOR max_iterations=$MAXIT"
  MICRODUCK_WARM_START=1 MICRODUCK_STAND_EXPERT="$EXPERT" MICRODUCK_WALK_ANCHOR="$ANCHOR" \
  uv run train Mjlab-VelStand-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.resume True --agent.load-run first-walk-src --agent.load-checkpoint model_10000.pt \
    --agent.max-iterations "$MAXIT"
  echo "=== $(date -Is) training exited with $?"
} 2>&1 | tee -a "$LOG"
echo "Training finished. Press Enter to close."; read -r _
