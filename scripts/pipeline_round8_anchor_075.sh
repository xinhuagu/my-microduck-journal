#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round8_anchor_075.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline 9 — S5: the point between the two search leaders. S3 (anchor x0.5 + stand-up imitation x5) recovers push
# falls like v4 but turns 0.51 / crouches -26.6; S2 (imitation x5, anchor x1.0) turns and crouches but recovers
# 92.5-95 %. S5 = anchor x0.75 + imitation x5, resumed from C model_2250 for 1000 iterations like the others;
# battery sweep over 2500..3249, pick by release score, 3-seed battery, export, release gate. Log: logs/pipeline9.log.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline9.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
TEACHER=logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt
WALK3=logs/rsl_rl/velocity/2026-10-15_22-09-39_velocity/model_2000.pt
TASK=Mjlab-VelStand-Rough-Backlash-MicroDuck
BASE=logs/rsl_rl/velstand/2026-10-18_12-34-58_velstand/model_2250.pt
export MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_SPEED_FINE_W=0 MICRODUCK_GROUND_MIX=1
export MICRODUCK_EXPERT_BC_COEF=5.0 MICRODUCK_ANCHOR_COEF=0.75
export MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK3" EXPERT="$TEACHER" ANCHOR="$WALK3"
V=S5_anchor_075_expert_x5

log "PIPELINE 9 START: $V — resume $BASE +1000 (ANCHOR_COEF=0.75, EXPERT_BC_COEF=5.0)"
uv run train "$TASK" --env.scene.num-envs 4096 --agent.resume True --agent.load-run "$(basename "$(dirname "$BASE")")" \
  --agent.load-checkpoint "$(basename "$BASE")" --agent.max-iterations 1000 > "logs/search_${V}_train.log" 2>&1
RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); RUN=${RUN%/}
log "$V: training exited ($(grep -E 'Learning iteration' "logs/search_${V}_train.log" | tail -1 | tr -s ' ')), run $RUN"
SW=logs/search_v03_sweep_$V; mkdir -p "$SW"
for f in $(ls "$RUN"/model_*.pt | sort -V); do
  it=$(basename "$f" | sed -E 's/model_([0-9]+)\.pt/\1/'); [ "$it" -lt 2500 ] && continue
  [ -s "$SW/${V}_${it}.json" ] && continue
  bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$f" 256 tracking,push "$SW/${V}_${it}.json" 0 "${V}_${it}" > /dev/null 2>&1
done
python3 "$S/pick_from_battery.py" "$SW" release | tee -a "$PL"
PICK=$(python3 "$S/pick_from_battery.py" "$SW" release | grep '^PICK' | awk '{print $2}')
log "$V: pick = $RUN/model_${PICK}.pt; full battery (3 seeds, incl. goal)"
bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$RUN/model_${PICK}.pt" 256 getup,push,tracking,body,goal "logs/eval/search_${V}.json" 0,1,2 "search_${V}" 2>&1 \
  | grep -E "^\[battery\] (done|wrote)|Traceback" | tee -a "$PL"
uv run scripts/export.py "$TASK" --checkpoint-file "$RUN/model_${PICK}.pt" --onnx-file "exports/search_${V}_${PICK}.onnx" > /tmp/export_9.log 2>&1 \
  && log "$V: exported exports/search_${V}_${PICK}.onnx" || log "$V: export failed (/tmp/export_9.log)"
log "RELEASE GATE (9)"
python3 "$S/release_gate.py" "logs/eval/search_${V}.json" logs/eval/search_S3_anchor_half_expert_x5.json logs/eval/search_S2_expert_x5.json 2>&1 | tee -a "$PL"
log "PIPELINE DONE (9)"
echo "Pipeline 9 finished. Press Enter to close."; read -r _
