#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/eval_battery.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Run the batched evaluation battery (eval_battery.py, copied next to this script) inside microduck_rl.
#   eval_battery.sh <task> <checkpoint.pt> [num_envs=256] [only=getup,push,tracking,body] [out.json]
# Teachers for VelStand tasks come from EXPERT / ANCHOR (defaults below); set MICRODUCK_BODY_WIDE / MICRODUCK_TURN
# in the environment when the checkpoint was trained with them.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
TASK="${1:?task}"; CKPT="${2:?checkpoint}"; N="${3:-256}"; ONLY="${4:-getup,push,tracking,body}"; OUT="${5:-}"
EXPERT="${EXPERT:-logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt}"
ANCHOR="${ANCHOR:-logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt}"
mkdir -p scripts_local && cp "$S/eval_battery.py" scripts_local/eval_battery.py
OUTARG=(); [ -n "$OUT" ] && OUTARG=(--out "$OUT")
MICRODUCK_STAND_EXPERT="$EXPERT" MICRODUCK_WALK_ANCHOR="$ANCHOR" \
uv run python scripts_local/eval_battery.py --task "$TASK" --checkpoint "$CKPT" --num-envs "$N" --only "$ONLY" "${OUTARG[@]}" 2>&1 \
  | grep -vE "DeprecationWarning|warnings.warn|UserWarning|tyro/_parsers"
