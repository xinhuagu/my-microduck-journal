#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round6b_resume_candidate_C.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline 7, resumed after the stop of 2026-10-18 01:11: candidate C was at iteration 2089 (checkpoints up to
# model_2000 in logs/rsl_rl/velstand/2026-10-17_22-21-33_velstand). Plain resume from model_2000 for the remaining
# 1000 iterations (curricula continue; the resume writes a NEW run dir holding 2000..2999), then the battery sweep over
# 2000..2999 (taken from whichever dir holds them), pick, 3-seed battery, export, release gate. Log: logs/pipeline7.log.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline7.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
TEACHER=logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt
WALK3=logs/rsl_rl/velocity/2026-10-15_22-09-39_velocity/model_2000.pt
C1=logs/rsl_rl/velstand/2026-10-17_22-21-33_velstand
TASK=Mjlab-VelStand-Rough-Backlash-MicroDuck
export MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_SPEED_FINE_W=0 MICRODUCK_EXPERT_BC_COEF=3.0 MICRODUCK_GROUND_MIX=1
export MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK3" EXPERT="$TEACHER" ANCHOR="$WALK3"
FROM=$(ls "$C1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1)
REMAIN=$((3000 - FROM))
log "PIPELINE 7 RESUME: candidate C from $C1/model_${FROM}.pt for $REMAIN iterations; GPU power limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"
uv run train "$TASK" --env.scene.num-envs 4096 --agent.resume True --agent.load-run "$(basename "$C1")" --agent.load-checkpoint "model_${FROM}.pt" \
  --agent.max-iterations "$REMAIN" > logs/cand_C_resume.log 2>&1
log "C: resume exited with $? ($(grep -E 'Learning iteration' logs/cand_C_resume.log | tail -1 | tr -s ' '))"
C2=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); C2=${C2%/}
log "C: resumed run $C2"

SW=logs/eval_sweep_cand_C_v02; mkdir -p "$SW"
log "C: battery sweep (tracking + push) on checkpoints 2000..2999"
for it in 2000 2250 2500 2750 2999; do
  f="$C2/model_${it}.pt"; [ -f "$f" ] || f="$C1/model_${it}.pt"; [ -f "$f" ] || continue
  [ -s "$SW/cand_C_v02_${it}.json" ] && continue
  bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$f" 256 tracking,push "$SW/cand_C_v02_${it}.json" 0 "cand_C_v02_${it}" > /dev/null 2>&1
done
python3 "$S/pick_from_battery.py" "$SW" release | tee -a "$PL"
PICK=$(python3 "$S/pick_from_battery.py" "$SW" release | grep '^PICK' | awk '{print $2}')
PF="$C2/model_${PICK}.pt"; [ -f "$PF" ] || PF="$C1/model_${PICK}.pt"
log "C: pick = $PF; full battery (3 seeds, incl. goal)"
bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$PF" 256 getup,push,tracking,body,goal logs/eval/cand_C_v02.json 0,1,2 cand_C_v02 2>&1 \
  | grep -E "^\[battery\] (done|wrote)|Traceback" | tee -a "$PL"
uv run scripts/export.py "$TASK" --checkpoint-file "$PF" --onnx-file "exports/cand_C_v02_${PICK}.onnx" > /tmp/export_C.log 2>&1 \
  && log "C: exported exports/cand_C_v02_${PICK}.onnx" || log "C: export failed (/tmp/export_C.log)"
log "RELEASE GATE"
python3 "$S/release_gate.py" logs/eval/cand_C_v02.json logs/eval/cand_A_v02.json logs/eval/cand_B_v02.json logs/eval/ours_velstand_v4.json 2>&1 | tee -a "$PL"
log "PIPELINE DONE (7)"
echo "Pipeline 7 finished. Press Enter to close."; read -r _
