#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/benchmark_all.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# The benchmark: every policy through the same battery, 256 envs, seeds 0,1,2, one JSON per policy in
# logs/eval/, then the markdown table. Skips policies whose JSON already exists (delete to re-run).
#   benchmark_all.sh            # everything
#   ONLY_NAMES="ours_velstand_v4" benchmark_all.sh   # a subset
export PATH="$HOME/.local/bin:$PATH"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
mkdir -p logs/eval
SEEDS="${SEEDS:-0,1,2}"; N="${N:-256}"
VS=Mjlab-VelStand-Flat-MicroDuck; VEL=Mjlab-Velocity-Flat-MicroDuck; SU=Mjlab-StandUp-Flat-MicroDuck
ALL=getup,push,tracking,body
# name | task | file | trials
ENTRIES=(
  "official_velstand_v7|$VS|pretrained/official/velstand.onnx|$ALL"
  "official_alpha_walking|$VEL|pretrained/official/alpha_walking.onnx|$ALL"
  "official_alpha_stand|$SU|pretrained/official/alpha_stand.onnx|getup,body"
  "ours_velstand_v4|$VS|exports/velstand_v4_2999.onnx|$ALL"
  "ours_velstand_v5_turning|$VS|exports/velstand_v5_2500.onnx|$ALL"
  "ours_velstand_v2|$VS|exports/velstand_v2_5750.onnx|$ALL"
  "ours_walk_teacher_v1|$VEL|exports/first_walk_10000.onnx|$ALL"
  "ours_walk_teacher_v2_turn|$VEL|exports/walk_v2_1750.onnx|$ALL"
  "ours_standup_teacher_v3|$SU|exports/standup_v3_2999.onnx|getup,body"
  "community_running_HannesVonEssen|$VEL|pretrained/community/microduck-running/policy.onnx|$ALL"
)
t0=$(date +%s)
for e in "${ENTRIES[@]}"; do
  IFS='|' read -r name task file trials <<< "$e"
  [ -n "${ONLY_NAMES:-}" ] && ! grep -qw "$name" <<< "$ONLY_NAMES" && continue
  out="logs/eval/${name}.json"
  if [ -s "$out" ]; then echo "=== $name: exists, skipped"; continue; fi
  if [ ! -s "$file" ]; then echo "=== $name: file missing ($file), skipped"; continue; fi
  echo "=== $(date -Is) $name ($task, $file, seeds $SEEDS)"
  NAME="$name" SEEDS="$SEEDS" bash "$S/eval_battery.sh" "$task" "$file" "$N" "$trials" "$out" "$SEEDS" "$name" 2>&1 | grep -E "^\[battery\] (headline|done|wrote)|^  [a-z_.0-9-]+ +[-+]|Traceback|Error" | grep -vE "DeprecationWarning"
done
echo "=== $(date -Is) all done in $(( ($(date +%s) - t0) / 60 )) min"
cp "$S/benchmark_table.py" scripts_local/benchmark_table.py
uv run python scripts_local/benchmark_table.py logs/eval --out logs/eval/benchmark.md \
  --order "official_velstand_v7,ours_velstand_v4,ours_velstand_v5_turning,ours_velstand_v2,official_alpha_walking,ours_walk_teacher_v1,ours_walk_teacher_v2_turn,community_running_HannesVonEssen,official_alpha_stand,ours_standup_teacher_v3"
echo "BENCHMARK DONE"
