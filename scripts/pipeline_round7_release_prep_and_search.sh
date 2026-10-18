#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round7_release_prep_and_search.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline 8 — what happens after candidate C, fully automatic. Log: logs/pipeline8.log.
#   W. wait for pipeline 7 (candidate C) to finish
#   R. release prep for v0.2 (rule in release_plan.py): default + fast variant ONNX, walk / stand teachers, eval JSONs,
#      notes generated from the battery JSONs, GIF of the fast policy (push -> get-up -> goal -> stop), SHA256SUMS
#      -> folder release-v0.2 next to this script. Publishing to GitHub is NOT done here (draft + review outside).
#   S. background search for "fast AND robust": resume from C's pick for 1000 iterations per variant, one change each
#      (S0 control = no change), battery (tracking + push + get-up, 1 seed) -> leaderboard; the best one, if its preview
#      passes, gets the full 3-seed battery + release gate.
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

# ---------- W ----------
log "PIPELINE 8 START — waiting for pipeline 7 (candidate C)"
until awk '/PIPELINE 7 RESUME/{s=NR} {l[NR]=$0} END{for(i=s;i<=NR;i++) print l[i]}' logs/pipeline7.log | grep -qE "PIPELINE DONE \(7\)|PIPELINE STOP"; do sleep 60; done
CPICK=$(awk '/PIPELINE 7 RESUME/{s=NR} {l[NR]=$0} END{for(i=s;i<=NR;i++) print l[i]}' logs/pipeline7.log | grep -oE "C: pick = \S+" | tail -1 | sed 's/C: pick = //; s/;$//')
log "W: pipeline 7 done; C pick = $CPICK"

# ---------- R ----------
REL="$S/release-v0.2"; rm -rf "$REL"; mkdir -p "$REL"
log "R: release plan"
python3 "$S/release_plan.py" logs/eval "$REL" 2>&1 | tee -a "$PL" | tail -12
python3 - "$REL/plan.json" <<'EOF' > /tmp/plan.env
import json, sys
p = json.load(open(sys.argv[1]))
print(f'DEFAULT={p["default"]}'); print(f'FAST={p["fast"] or ""}')
for n, f in p["files"].items(): print(f'FILE_{n}={f}')
EOF
. /tmp/plan.env
DFILE=$(eval echo "\$FILE_${DEFAULT}"); FFILE=""; [ -n "$FAST" ] && FFILE=$(eval echo "\$FILE_${FAST}")
log "R: default = $DEFAULT ($DFILE), fast = ${FAST:-none} (${FFILE:-—})"
env -u MICRODUCK_BODY_WIDE -u MICRODUCK_STAND_EXPERT -u MICRODUCK_WALK_ANCHOR uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --checkpoint-file "$WALK3" --onnx-file exports/walk_teacher_v3_2000.onnx > /tmp/export_walk3.log 2>&1 || log "R: walk teacher export failed"
cp "$DFILE" "$REL/velstand_default.onnx"
[ -n "$FFILE" ] && cp "$FFILE" "$REL/velstand_fast.onnx"
cp exports/walk_teacher_v3_2000.onnx "$REL/walk_teacher_v3.onnx"
cp exports/standup_v3_2999.onnx "$REL/standup_teacher_v3.onnx"
for n in "$DEFAULT" "$FAST" official_velstand_v7 ours_velstand_v4; do [ -n "$n" ] && [ -f "logs/eval/$n.json" ] && cp "logs/eval/$n.json" "$REL/eval_$n.json"; done
GIFSRC="${FFILE:-$DFILE}"
log "R: recording the GIF with $GIFSRC (1 duck, goal 1.5 m, push at 2 s)"
rm -rf logs/video_v02
BATTERY_GOAL_PUSH_ONLY=1 VIDEO=logs/video_v02 VIDEO_STEPS=1200 bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$GIFSRC" 1 goal /tmp/goal_video_v02.json 0 video_v02 > /tmp/video_v02.log 2>&1
MP4=$(ls logs/video_v02/*.mp4 2>/dev/null | head -1)
if [ -n "$MP4" ]; then
  uv run python "$S/make_goal_gif.py" "$MP4" "$REL/fall-getup-goal.gif" 0 12 8 400 | tee -a "$PL"
else
  log "R: no video produced (see /tmp/video_v02.log)"
fi
(cd "$REL" && sha256sum *.onnx eval_*.json *.gif > SHA256SUMS.txt 2>/dev/null; { echo; echo '### Checksums'; echo; echo '```'; grep -E '\.onnx$' SHA256SUMS.txt; echo '```'; } >> RELEASE_NOTES.md)
log "RELEASE V0.2 PREPARED in $REL (default=$DEFAULT, fast=${FAST:-none})"

# ---------- S ----------
log "S: background search from $CPICK, 1000 iterations per variant"
BASE_RUN=$(basename "$(dirname "$CPICK")"); BASE_IT=$(basename "$CPICK" | sed -E 's/model_([0-9]+)\.pt/\1/')
SD=logs/search_v03; mkdir -p "$SD"
run_variant() {  # <name> <env assignments...>
  local name="$1"; shift
  [ -s "$SD/$name.json" ] && { log "S: $name already evaluated, skipped"; return; }
  log "S: $name ($*) — resume $BASE_RUN/model_${BASE_IT}.pt +1000"
  env MICRODUCK_SPEED_FINE_W=0 MICRODUCK_EXPERT_BC_COEF=3.0 MICRODUCK_GROUND_MIX=1 "$@" \
    uv run train "$TASK" --env.scene.num-envs 4096 --agent.resume True --agent.load-run "$BASE_RUN" --agent.load-checkpoint "model_${BASE_IT}.pt" \
    --agent.max-iterations 1000 > "logs/search_${name}_train.log" 2>&1
  local run; run=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); run=${run%/}
  local last; last=$(ls "$run" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
  env MICRODUCK_SPEED_FINE_W=0 MICRODUCK_EXPERT_BC_COEF=3.0 MICRODUCK_GROUND_MIX=1 "$@" \
    bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$run/model_${last}.pt" 256 tracking,push,getup "$SD/$name.json" 0 "$name" > /dev/null 2>&1
  log "S: $name evaluated ($run/model_${last}.pt)"
  python3 "$S/search_board.py" "$SD" | tee -a "$PL" > /dev/null
}
run_variant S0_control
run_variant S1_anchor_half MICRODUCK_ANCHOR_COEF=0.5
run_variant S2_expert_x5 MICRODUCK_EXPERT_BC_COEF=5.0
run_variant S3_anchor_half_expert_x5 MICRODUCK_ANCHOR_COEF=0.5 MICRODUCK_EXPERT_BC_COEF=5.0
run_variant S4_speed_term_half MICRODUCK_SPEED_FINE_W=0.5
log "S: leaderboard"
python3 "$S/search_board.py" "$SD" | tee -a "$PL"
BESTLINE=$(python3 "$S/search_board.py" "$SD" | grep '^BEST')
BESTNAME=$(echo "$BESTLINE" | awk '{print $2}'); BESTCK=$(echo "$BESTLINE" | awk '{print $4}')
if echo "$BESTLINE" | grep -q " PASS "; then
  log "S: $BESTNAME passes the single-seed preview — full battery (3 seeds, incl. body + goal) and release gate"
  bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$BESTCK" 256 getup,push,tracking,body,goal "logs/eval/search_best_v03.json" 0,1,2 "search_best_${BESTNAME}" 2>&1 | grep -E "^\[battery\] (done|wrote)" | tee -a "$PL"
  uv run scripts/export.py "$TASK" --checkpoint-file "$BESTCK" --onnx-file "exports/search_best_v03_${BESTNAME}.onnx" > /tmp/export_search.log 2>&1 \
    && log "S: exported exports/search_best_v03_${BESTNAME}.onnx" || log "S: export failed (/tmp/export_search.log)"
  python3 "$S/release_gate.py" logs/eval/search_best_v03.json logs/eval/ours_velstand_v4.json 2>&1 | tee -a "$PL"
else
  log "S: best variant $BESTNAME does not pass the preview — no full battery"
fi
log "PIPELINE DONE (8)"
echo "Pipeline 8 finished. Press Enter to close."; read -r _
