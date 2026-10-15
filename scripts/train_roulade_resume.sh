#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/train_roulade_resume.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Resume the forward-roll training after the shutdown of 2026-10-15 21:48 (stopped at iteration 301, checkpoint model_250).
# Plain resume from the newest checkpoint of the newest microduck_roulade run (curricula continue); a resume creates a NEW
# timestamped run directory, so the newest-dir lookup is what keeps this correct across several stops.
#   train_roulade_resume.sh [target_total_iterations=10000]
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/microduck_rl" || exit 1
TOTAL="${1:-10000}"
RUN=$(ls -dt logs/rsl_rl/microduck_roulade/*/ | head -1); RUN=${RUN%/}
IT=$(ls "$RUN" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
REMAIN=$((TOTAL - IT)); [ "$REMAIN" -gt 0 ] || { echo "already at $IT >= $TOTAL"; exit 0; }
LOG=logs/roulade_train.log
{
  echo "=== $(date -Is) roulade RESUME from $(basename "$RUN")/model_${IT}.pt for $REMAIN more iterations (target $TOTAL)"
  uv run train Mjlab-Roulade-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.resume True --agent.load-run "$(basename "$RUN")" --agent.load-checkpoint "model_${IT}.pt" \
    --agent.max-iterations "$REMAIN"
  echo "=== $(date -Is) training exited with $?"
} 2>&1 | tee -a "$LOG"
echo "Training finished. Press Enter to close."; read -r _
