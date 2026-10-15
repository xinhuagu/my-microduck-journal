#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/train_roulade.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Forward roll (roulade): smoke test, then full training from scratch on Mjlab-Roulade-Flat-MicroDuck.
#   train_roulade.sh [max_iterations=10000]
# Standalone episodic task: no teachers, no warm start. Checkpoints every 250 iterations; stop any time with Ctrl-C.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/microduck_rl" || exit 1
MAXIT="${1:-10000}"
LOG=logs/roulade_train.log
{
  echo "=== $(date -Is) roulade smoke test (3 iters, 64 envs)"
  uv run train Mjlab-Roulade-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 3 > logs/roulade_smoke.log 2>&1
  RC=$?
  grep -aE "Learning iteration|Traceback|Error" logs/roulade_smoke.log | grep -av DeprecationWarning | tail -3
  if [ $RC -ne 0 ]; then echo "SMOKE FAILED (exit $RC)"; tail -15 logs/roulade_smoke.log; exit $RC; fi
  echo "=== $(date -Is) smoke OK - training Mjlab-Roulade-Flat-MicroDuck, 4096 envs, max_iterations=$MAXIT"
  uv run train Mjlab-Roulade-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations "$MAXIT"
  echo "=== $(date -Is) training exited with $?"
} 2>&1 | tee -a "$LOG"
echo "Training finished. Press Enter to close."; read -r _
