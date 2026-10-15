#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round4_speed.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline 5 — the speed round, logged to logs/pipeline5.log.
#   R0. smoke: VelStand Rough-Backlash with TURN + SPEED + BODY_WIDE (fail fast before the long runs)
#   R1. walk teacher v3: Velocity, warm start from first-walk model_10000, MICRODUCK_TURN=1 + MICRODUCK_SPEED=1, 3000 iters
#   R2. battery (tracking, 256 envs) on its checkpoints 1500..2999 -> pick; GATE: speed@0.30 >= 0.20 and turn@0.6 >= 0.50
#   R3. VelStand v6: warm start from v4 model_2999, expert = teacher v3, anchor = walk v3 pick, TURN + SPEED + BODY_WIDE,
#       Rough-Backlash, 3000 iters
#   R4. battery (tracking + push) on v6 checkpoints 2000..2999 -> pick
#   R5. full battery, 3 seeds, incl. goal, on the v6 pick -> logs/eval/ours_velstand_v6.json
# The teacher is judged BEFORE any distillation (lesson from v5).
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline5.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
ckpts_between() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | awk -v a="$2" -v b="$3" '$1 >= a && $1 <= b'; }
WALK1_RUN=2026-09-07_20-38-41_first-walk
TEACHER=logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt
V4RUN=2026-10-11_19-13-04_velstand; V4IT=2999
TASK_VS=Mjlab-VelStand-Rough-Backlash-MicroDuck
for f in "logs/rsl_rl/velocity/$WALK1_RUN/model_10000.pt" "$TEACHER" "logs/rsl_rl/velstand/$V4RUN/model_${V4IT}.pt"; do
  [ -f "$f" ] || { log "PIPELINE STOP: missing $f"; exit 1; }
done
log "PIPELINE 5 START (speed round); GPU power limit: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"

# ---------- R0 ----------
log "R0: smoke VelStand ($TASK_VS) with TURN + SPEED + BODY_WIDE"
MICRODUCK_WARM_START=1 MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 \
MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="logs/rsl_rl/velocity/$WALK1_RUN/model_10000.pt" \
uv run train "$TASK_VS" --env.scene.num-envs 64 --agent.max-iterations 3 \
  --agent.resume True --agent.load-run "$V4RUN" --agent.load-checkpoint "model_${V4IT}.pt" > logs/pipeline5_smoke.log 2>&1
RC=$?
if [ $RC -ne 0 ] || grep -q Traceback logs/pipeline5_smoke.log; then log "PIPELINE STOP: R0 smoke failed (exit $RC)"; tail -5 logs/pipeline5_smoke.log | tee -a "$PL"; exit 1; fi
log "R0: smoke OK ($(grep -c 'track_lin_vel_x_fine' logs/pipeline5_smoke.log) lines with the speed term)"
# (the smoke leaves a 3-iteration run dir under logs/rsl_rl/velstand/; R3's run is created later, so 'newest' still finds R3)

# ---------- R1 ----------
log "R1: walk teacher v3 (TURN + SPEED), warm start from $WALK1_RUN/model_10000.pt, 3000 iterations"
MICRODUCK_WARM_START=1 MICRODUCK_TURN=1 MICRODUCK_SPEED=1 \
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$WALK1_RUN" --agent.load-checkpoint model_10000.pt \
  --agent.max-iterations 3000 > logs/walk_v3_train.log 2>&1
log "R1: exited with $? ($(grep -E 'Learning iteration' logs/walk_v3_train.log | tail -1 | tr -s ' '))"
W3RUN=$(ls -dt logs/rsl_rl/velocity/*/ | head -1); W3RUN=${W3RUN%/}
log "R1: walk v3 run $W3RUN"

# ---------- R2 ----------
log "R2: battery (tracking, 256 envs, seed 0) on walk v3 checkpoints 1500..2999"
SW=logs/eval_sweep_walk3; mkdir -p "$SW"
for it in $(ckpts_between "$W3RUN" 1500 2999); do
  MICRODUCK_TURN=1 MICRODUCK_SPEED=1 bash "$S/eval_battery.sh" Mjlab-Velocity-Flat-MicroDuck "$W3RUN/model_${it}.pt" 256 tracking "$SW/walk3_${it}.json" 0 "walk3_${it}" > /dev/null 2>&1
done
python3 "$S/pick_from_battery.py" "$SW" walk | tee -a "$PL"
PICKLINE=$(python3 "$S/pick_from_battery.py" "$SW" walk | grep '^PICK')
W3PICK=$(echo "$PICKLINE" | awk '{print $2}')
if ! echo "$PICKLINE" | grep -q PASS; then log "PIPELINE STOP (gate): walk teacher v3 does not meet the speed+turn gate — no distillation. $PICKLINE"; exit 0; fi
export WALK3="$W3RUN/model_${W3PICK}.pt"; log "R2: walk v3 pick = $WALK3"

# ---------- R3 ----------
log "R3: VelStand v6 ($TASK_VS, TURN + SPEED + BODY_WIDE), warm start from $V4RUN/model_${V4IT}.pt, expert=$TEACHER, anchor=$WALK3, 3000 iterations"
MICRODUCK_WARM_START=1 MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 \
MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK3" \
uv run train "$TASK_VS" --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$V4RUN" --agent.load-checkpoint "model_${V4IT}.pt" \
  --agent.max-iterations 3000 > logs/velstand_v6_train.log 2>&1
log "R3: exited with $? ($(grep -E 'Learning iteration' logs/velstand_v6_train.log | tail -1 | tr -s ' '))"
V6RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); V6RUN=${V6RUN%/}
log "R3: v6 run $V6RUN"

# ---------- R4 ----------
log "R4: battery (tracking + push, 256 envs, seed 0) on v6 checkpoints 2000..2999"
SV=logs/eval_sweep_v6; mkdir -p "$SV"
export EXPERT="$TEACHER" ANCHOR="$WALK3"
for it in $(ckpts_between "$V6RUN" 2000 2999); do
  MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$V6RUN/model_${it}.pt" 256 tracking,push "$SV/v6_${it}.json" 0 "v6_${it}" > /dev/null 2>&1
done
python3 "$S/pick_from_battery.py" "$SV" velstand | tee -a "$PL"
V6PICK=$(python3 "$S/pick_from_battery.py" "$SV" velstand | grep '^PICK' | awk '{print $2}')
log "R4: v6 pick = $V6RUN/model_${V6PICK}.pt"

# ---------- R5 ----------
log "R5: full battery (3 seeds, getup/push/tracking/body/goal) on the v6 pick"
MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$V6RUN/model_${V6PICK}.pt" 256 getup,push,tracking,body,goal logs/eval/ours_velstand_v6.json 0,1,2 ours_velstand_v6 2>&1 \
  | grep -E "^\[battery\] (headline|done|wrote)|^  [a-z_.0-9-]+ +[-+]|Traceback" | tee -a "$PL"
MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK3" \
uv run scripts/export.py "$TASK_VS" --checkpoint-file "$V6RUN/model_${V6PICK}.pt" --onnx-file "exports/velstand_v6_${V6PICK}.onnx" > /tmp/export_v6.log 2>&1 \
  && log "R5: exported exports/velstand_v6_${V6PICK}.onnx" || log "R5: export failed (see /tmp/export_v6.log)"
log "PIPELINE DONE (5)"
echo "Pipeline 5 finished. Press Enter to close."; read -r _
