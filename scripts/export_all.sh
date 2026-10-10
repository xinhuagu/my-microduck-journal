#!/usr/bin/env bash
# Export every checkpoint of a microduck_rl run to ONNX.
#
#   scripts/export_all.sh <run_dir> <task_id> [out_dir] [name_prefix]
#   e.g. scripts/export_all.sh logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk \
#          Mjlab-Velocity-Flat-MicroDuck exports first_walk
#
# Run from the microduck_rl checkout. ~13 s per checkpoint on an RTX 5090.
set -u
RUN="${1:?run dir}"
TASK="${2:?task id}"
OUT="${3:-exports}"
PREFIX="${4:-policy}"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p "$OUT"
ok=0; fail=0
for it in $(ls "$RUN"/model_*.pt | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n); do
  out="$OUT/${PREFIX}_${it}.onnx"
  if [ -s "$out" ]; then echo "skip $out (exists)"; ok=$((ok+1)); continue; fi
  t0=$(date +%s)
  if uv run scripts/export.py "$TASK" --checkpoint-file "$RUN/model_${it}.pt" --onnx-file "$out" > "$OUT/export_${it}.log" 2>&1; then
    echo "ok   model_${it}.pt -> $out ($(( $(date +%s) - t0 )) s)"; ok=$((ok+1))
  else
    echo "FAIL model_${it}.pt (see $OUT/export_${it}.log)"; fail=$((fail+1))
  fi
done
echo "done: $ok ok, $fail failed"
