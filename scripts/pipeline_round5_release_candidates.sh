#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round5_release_candidates.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline 6 — two v0.2 release candidates, then the release gate. Logged to logs/pipeline6.log.
#   A. warm start v4 model_2999, anchor = walk v3 model_2000, TURN + SPEED + BODY_WIDE, MICRODUCK_SPEED_FINE_W=0
#      (speed only through the anchor) — 3000 iterations on Rough-Backlash
#   B. resume v6 model_2250 (curricula continue), MICRODUCK_SPEED_FINE_W=1.0, MICRODUCK_EXPERT_BC_COEF=2.0 — 2000 iterations
#   For each: battery sweep (tracking + push, 256 envs, seed 0) over the last 1000 iterations -> pick ("release" score)
#   -> full battery 3 seeds incl. goal -> logs/eval/cand_{A,B}_v02.json -> export -> release_gate.py on both.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline6.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
ckpts_between() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | awk -v a="$2" -v b="$3" '$1 >= a && $1 <= b'; }
last_ckpt() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1; }
TEACHER=logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt
WALK3=logs/rsl_rl/velocity/2026-10-15_22-09-39_velocity/model_2000.pt
V4RUN=2026-10-11_19-13-04_velstand; V6RUN=2026-10-17_09-56-19_velstand
TASK=Mjlab-VelStand-Rough-Backlash-MicroDuck
export MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK3"
export EXPERT="$TEACHER" ANCHOR="$WALK3"
for f in "$TEACHER" "$WALK3" "logs/rsl_rl/velstand/$V4RUN/model_2999.pt" "logs/rsl_rl/velstand/$V6RUN/model_2250.pt"; do
  [ -f "$f" ] || { log "PIPELINE STOP: missing $f"; exit 1; }
done
log "PIPELINE 6 START (v0.2 candidates A and B); GPU power limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"

evaluate() {   # <label> <run dir> <first it> <last it> <fine w> <bc coef>
  local label="$1" run="$2" a="$3" b="$4" w="$5" c="$6"
  local SW="logs/eval_sweep_$label"; mkdir -p "$SW"
  log "$label: battery sweep (tracking + push) on $run checkpoints $a..$b"
  for it in $(ckpts_between "$run" "$a" "$b"); do
    [ -s "$SW/${label}_${it}.json" ] && continue
    MICRODUCK_SPEED_FINE_W="$w" MICRODUCK_EXPERT_BC_COEF="$c" bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$run/model_${it}.pt" 256 tracking,push "$SW/${label}_${it}.json" 0 "${label}_${it}" > /dev/null 2>&1
  done
  python3 "$S/pick_from_battery.py" "$SW" release | tee -a "$PL"
  local pick; pick=$(python3 "$S/pick_from_battery.py" "$SW" release | grep '^PICK' | awk '{print $2}')
  log "$label: pick = $run/model_${pick}.pt; full battery (3 seeds, incl. goal)"
  MICRODUCK_SPEED_FINE_W="$w" MICRODUCK_EXPERT_BC_COEF="$c" bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$run/model_${pick}.pt" 256 getup,push,tracking,body,goal "logs/eval/${label}.json" 0,1,2 "$label" 2>&1 \
    | grep -E "^\[battery\] (done|wrote)|Traceback" | tee -a "$PL"
  MICRODUCK_SPEED_FINE_W="$w" MICRODUCK_EXPERT_BC_COEF="$c" uv run scripts/export.py "$TASK" --checkpoint-file "$run/model_${pick}.pt" --onnx-file "exports/${label}_${pick}.onnx" > "/tmp/export_${label}.log" 2>&1 \
    && log "$label: exported exports/${label}_${pick}.onnx" || log "$label: export failed (/tmp/export_${label}.log)"
}

# ---------- A ----------
log "A: warm start $V4RUN/model_2999.pt, anchor $WALK3, SPEED_FINE_W=0 (speed only via the anchor), 3000 iterations"
MICRODUCK_WARM_START=1 MICRODUCK_SPEED_FINE_W=0 \
uv run train "$TASK" --env.scene.num-envs 4096 --agent.resume True --agent.load-run "$V4RUN" --agent.load-checkpoint model_2999.pt \
  --agent.max-iterations 3000 > logs/cand_A_train.log 2>&1
log "A: exited with $? ($(grep -E 'Learning iteration' logs/cand_A_train.log | tail -1 | tr -s ' '))"
ARUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); ARUN=${ARUN%/}
log "A: run $ARUN (last checkpoint $(last_ckpt "$ARUN"))"
evaluate cand_A_v02 "$ARUN" 2000 2999 0 1.0

# ---------- B ----------
log "B: resume $V6RUN/model_2250.pt, SPEED_FINE_W=1.0, EXPERT_BC_COEF=2.0, 2000 iterations"
MICRODUCK_SPEED_FINE_W=1.0 MICRODUCK_EXPERT_BC_COEF=2.0 \
uv run train "$TASK" --env.scene.num-envs 4096 --agent.resume True --agent.load-run "$V6RUN" --agent.load-checkpoint model_2250.pt \
  --agent.max-iterations 2000 > logs/cand_B_train.log 2>&1
log "B: exited with $? ($(grep -E 'Learning iteration' logs/cand_B_train.log | tail -1 | tr -s ' '))"
BRUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); BRUN=${BRUN%/}
BLAST=$(last_ckpt "$BRUN"); log "B: run $BRUN (last checkpoint $BLAST)"
evaluate cand_B_v02 "$BRUN" $((BLAST - 1000)) "$BLAST" 1.0 2.0

# ---------- gate ----------
log "RELEASE GATE"
python3 "$S/release_gate.py" logs/eval/cand_A_v02.json logs/eval/cand_B_v02.json logs/eval/ours_velstand_v4.json logs/eval/ours_velstand_v6.json 2>&1 | tee -a "$PL"
log "PIPELINE DONE (6)"
echo "Pipeline 6 finished. Press Enter to close."; read -r _
