#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round7b_promote.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline 8b — promote the two leaders of the search (S3 anchor ×0.5 + stand-up imitation ×5, S2 imitation ×5).
# The search judged only each run's last checkpoint on one seed, and its control (S0) showed that C's settings
# erode the recovery again with more iterations — so the last checkpoint is not necessarily the best. Here:
# battery sweep (tracking + push, 1 seed) over every saved checkpoint of each run, pick by release score,
# full 3-seed battery on the pick, export, release gate. Log: logs/pipeline8.log (appended).
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline8.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
TEACHER=logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt
WALK3=logs/rsl_rl/velocity/2026-10-15_22-09-39_velocity/model_2000.pt
TASK=Mjlab-VelStand-Rough-Backlash-MicroDuck
export MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK3"
export EXPERT="$TEACHER" ANCHOR="$WALK3"

until grep -qE "PIPELINE DONE \(8\)|PIPELINE STOP" "$PL"; do sleep 60; done
log "PIPELINE 8b START — promote S3 and S2 (sweep every checkpoint, then 3-seed battery + gate)"
GATE_FILES=""
for V in S3_anchor_half_expert_x5 S2_expert_x5; do
  RUN=$(grep -E "S: $V evaluated" "$PL" | tail -1 | sed -E 's/.*evaluated \((.*)\/model_[0-9]+\.pt\).*/\1/')
  SW=logs/search_v03_sweep_$V; mkdir -p "$SW"
  log "8b: $V — sweep $RUN"
  for f in $(ls "$RUN"/model_*.pt | sort -V); do
    it=$(basename "$f" | sed -E 's/model_([0-9]+)\.pt/\1/'); [ "$it" -lt 2500 ] && continue
    [ -s "$SW/${V}_${it}.json" ] && continue
    bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$f" 256 tracking,push "$SW/${V}_${it}.json" 0 "${V}_${it}" > /dev/null 2>&1
  done
  python3 "$S/pick_from_battery.py" "$SW" release | tee -a "$PL"
  PICK=$(python3 "$S/pick_from_battery.py" "$SW" release | grep '^PICK' | awk '{print $2}')
  log "8b: $V pick = $RUN/model_${PICK}.pt; full battery (3 seeds, incl. goal)"
  bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$RUN/model_${PICK}.pt" 256 getup,push,tracking,body,goal "logs/eval/search_${V}.json" 0,1,2 "search_${V}" 2>&1 \
    | grep -E "^\[battery\] (done|wrote)|Traceback" | tee -a "$PL"
  uv run scripts/export.py "$TASK" --checkpoint-file "$RUN/model_${PICK}.pt" --onnx-file "exports/search_${V}_${PICK}.onnx" > /tmp/export_8b.log 2>&1 \
    && log "8b: exported exports/search_${V}_${PICK}.onnx" || log "8b: export failed (/tmp/export_8b.log)"
  GATE_FILES="$GATE_FILES logs/eval/search_${V}.json"
done
log "RELEASE GATE (8b)"
python3 "$S/release_gate.py" $GATE_FILES logs/eval/cand_C_v02.json 2>&1 | tee -a "$PL"
log "PIPELINE DONE (8b)"
echo "Pipeline 8b finished. Press Enter to close."; read -r _
