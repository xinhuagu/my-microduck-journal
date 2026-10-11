#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/train_standup_teacher.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# StandUp teacher: smoke test (5 iters / 64 envs), then full training (4096 envs, 6000 iters max).
# Output is mirrored to logs/standup_train.log inside microduck_rl.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/microduck_rl" || exit 1
mkdir -p logs
LOG=logs/standup_train.log
{
  echo "=== $(date -Is) StandUp smoke test (5 iters, 64 envs)"
  uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5 > logs/standup_smoke.log 2>&1
  RC=$?
  grep -E "Learning iteration|Steps per second|Traceback|Error" logs/standup_smoke.log | tail -6
  if [ $RC -ne 0 ]; then echo "SMOKE FAILED (exit $RC) - not starting full training"; tail -15 logs/standup_smoke.log; exit $RC; fi
  echo "=== $(date -Is) smoke OK - starting full training"
  if uv run train --help 2>&1 | grep -q -- '--agent.run-name'; then RUNNAME="--agent.run-name standup-teacher"; else RUNNAME=""; fi
  echo "cmd: uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 6000 $RUNNAME"
  uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 6000 $RUNNAME
  echo "=== $(date -Is) training exited with $?"
} 2>&1 | tee -a "$LOG"
echo "Training finished. Press Enter to close."; read -r _
