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
SEEDS="${6:-${SEEDS:-0}}"; NAME="${7:-${NAME:-}}"   # 6th arg or env SEEDS="0,1,2"; 7th arg or env NAME = leaderboard label
# Checkpoint shorthand so run directories are never typed by hand (resumes create a NEW timestamped dir):
#   latest:<experiment>[:<iteration>]   -> newest run dir under logs/rsl_rl/<experiment>/ (default: its last checkpoint)
#   e.g. latest:velstand:2500  or  latest:microduck_stand
if [[ "$CKPT" == latest:* ]]; then
  exp=$(echo "$CKPT" | cut -d: -f2); it=$(echo "$CKPT" | cut -d: -f3)
  run=$(ls -dt logs/rsl_rl/"$exp"/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); run=${run%/}
  [ -z "$it" ] && it=$(ls "$run" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
  CKPT="$run/model_${it}.pt"; echo "[battery] resolved $2 -> $CKPT"
fi
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
EXPERT="${EXPERT:-logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt}"
ANCHOR="${ANCHOR:-logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt}"
mkdir -p scripts_local && cp "$S/eval_battery.py" scripts_local/eval_battery.py
OUTARG=(); [ -n "$OUT" ] && OUTARG=(--out "$OUT")
[ -n "$NAME" ] && OUTARG+=(--name "$NAME")
MICRODUCK_STAND_EXPERT="$EXPERT" MICRODUCK_WALK_ANCHOR="$ANCHOR" \
uv run python scripts_local/eval_battery.py --task "$TASK" --checkpoint "$CKPT" --num-envs "$N" --only "$ONLY" --seeds "$SEEDS" "${OUTARG[@]}" 2>&1 \
  | grep -vE "DeprecationWarning|warnings.warn|UserWarning|tyro/_parsers"
