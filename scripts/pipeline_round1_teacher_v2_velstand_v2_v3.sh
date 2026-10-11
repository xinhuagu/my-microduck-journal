#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round1_teacher_v2_velstand_v2_v3.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Overnight pipeline (all sequential, logged to logs/pipeline.log):
#   A. wait for the running StandUp v2 fine-tune (1500) to finish
#   B. posture probe on its last checkpoint -> STOP if the duck does not end upright (mean |pitch| > 12 deg)
#   C. continue StandUp v2 to 6000 iterations (resume, same straight rewards)
#   D. probe + push test on the final teacher
#   E. VelStand v2 (Flat), 6000 iterations: warm start first-walk/model_10000, expert = teacher v2, anchor = walk
#   F. push tests (standing + walking) + probe on VelStand v2
#   G. VelStand v3 (Rough-Backlash), 3000 iterations: warm start from VelStand v2
#   H. push tests on VelStand v3
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WALK=logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
latest_ckpt() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1; }

# probe helper: prints the posture table and a verdict line "PROBE mean_abs_pitch_last4=<deg> min_z=<m>"
probe() {  # <onnx> <flag>
  bash "$S/posture_probe.sh" "$1" "$2" 3 > /tmp/probe_out.txt 2>&1
  cat /tmp/probe_out.txt | tee -a "$PL"
  python3 - <<'EOF' | tee -a "$PL"
import re
rows=[]
for line in open("/tmp/probe_out.txt"):
    m=re.match(r"\s*(\d+)\s+\S+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
    if m: rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
rows=[r for r in rows if r[0]>=50]
last=rows[-4:] if len(rows)>=4 else rows
mp=sum(abs(r[2]) for r in last)/max(len(last),1)
mz=min(r[1] for r in rows) if rows else float('nan')
print(f"PROBE mean_abs_pitch_last4={mp:.1f} min_z={mz:.3f} rows={len(rows)}")
EOF
}

log "PIPELINE START"
# ---------- A ----------
log "A: waiting for StandUp v2 (1500) to finish"
until grep -qE "training exited" logs/standup_straight_train.log 2>/dev/null; do sleep 60; done
V2RUN=$(ls -dt logs/rsl_rl/microduck_stand/*/ | head -1); V2RUN=${V2RUN%/}
V2IT=$(latest_ckpt "$V2RUN"); log "A: v2 run $V2RUN last checkpoint model_${V2IT}.pt"

# ---------- B ----------
log "B: export + posture probe of teacher v2 @ $V2IT"
uv run scripts/export.py Mjlab-StandUp-Flat-MicroDuck --checkpoint-file "$V2RUN/model_${V2IT}.pt" --onnx-file "exports/standup_v2_${V2IT}.onnx" > /tmp/export_v2.log 2>&1 || { log "B: EXPORT FAILED"; tail -3 /tmp/export_v2.log | tee -a "$PL"; }
probe "exports/standup_v2_${V2IT}.onnx" --standing
MP=$(grep -oE "PROBE mean_abs_pitch_last4=[0-9.]+" "$PL" | tail -1 | sed 's/.*=//')
if [ "$(echo "$MP > 12" | bc)" = "1" ]; then log "PIPELINE STOP: teacher v2 still leans (mean |pitch| after pushes = ${MP} deg)"; exit 1; fi
log "B: teacher v2 ends upright (mean |pitch| = ${MP} deg) -> continuing"

# ---------- C ----------
log "C: continue StandUp v2 to 6000 iterations (resume from model_${V2IT}.pt)"
MICRODUCK_STANDUP_STRAIGHT=1 uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$(basename "$V2RUN")" --agent.load-checkpoint "model_${V2IT}.pt" \
  --agent.max-iterations 6000 > logs/standup_v2_cont.log 2>&1
log "C: exited with $? ($(grep -E 'Learning iteration' logs/standup_v2_cont.log | tail -1 | tr -s ' '))"
V2CRUN=$(ls -dt logs/rsl_rl/microduck_stand/*/ | head -1); V2CRUN=${V2CRUN%/}
TIT=$(latest_ckpt "$V2CRUN"); export TEACHER="$V2CRUN/model_${TIT}.pt"; log "C: teacher = $TEACHER"

# ---------- D ----------
log "D: probe + push test of the final teacher"
uv run scripts/export.py Mjlab-StandUp-Flat-MicroDuck --checkpoint-file "$TEACHER" --onnx-file "exports/standup_v2_${TIT}.onnx" > /tmp/export_v2c.log 2>&1
probe "exports/standup_v2_${TIT}.onnx" --standing
TASK=Mjlab-StandUp-Flat-MicroDuck RUNDIR="$V2CRUN" FLAG=--standing PREFIX=standup_v2 PREKEYS="" bash "$S/pushtest.sh" "$TIT" 2>&1 | tee -a "$PL"

# ---------- E ----------
log "E: VelStand v2 (Flat), 6000 iterations, expert=$TEACHER anchor=$WALK"
MICRODUCK_WARM_START=1 MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK" \
uv run train Mjlab-VelStand-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run first-walk-src --agent.load-checkpoint model_10000.pt \
  --agent.max-iterations 6000 > logs/velstand_v2_train.log 2>&1
log "E: exited with $? ($(grep -E 'Learning iteration' logs/velstand_v2_train.log | tail -1 | tr -s ' '))"
VS2RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -v first-walk-src | head -1); VS2RUN=${VS2RUN%/}
VS2IT=$(latest_ckpt "$VS2RUN"); log "E: velstand v2 run $VS2RUN last checkpoint model_${VS2IT}.pt"

# ---------- F ----------
log "F: VelStand v2 push tests + probe"
export EXPERT="$TEACHER" ANCHOR="$WALK"
TASK=Mjlab-VelStand-Flat-MicroDuck RUNDIR="$VS2RUN" FLAG=--walking PREFIX=velstand_v2 PREKEYS="" bash "$S/pushtest.sh" "$VS2IT" 2>&1 | tee -a "$PL"
TASK=Mjlab-VelStand-Flat-MicroDuck RUNDIR="$VS2RUN" FLAG=--walking PREFIX=velstand_v2 PREKEYS="Up Up" bash "$S/pushtest.sh" "$VS2IT" 2>&1 | tee -a "$PL"
probe "exports/velstand_v2_${VS2IT}.onnx" --walking

# ---------- G ----------
RB=$(uv run list-envs 2>/dev/null | grep -oE "Mjlab-VelStand-Rough-Backlash-MicroDuck" | head -1)
[ -z "$RB" ] && RB=$(uv run list-envs 2>/dev/null | grep -oE "Mjlab-VelStand-Flat-Backlash-MicroDuck" | head -1)
log "G: VelStand v3 ($RB), 3000 iterations, warm start from $VS2RUN/model_${VS2IT}.pt"
MICRODUCK_WARM_START=1 MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK" \
uv run train "$RB" --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$(basename "$VS2RUN")" --agent.load-checkpoint "model_${VS2IT}.pt" \
  --agent.max-iterations 3000 > logs/velstand_v3_train.log 2>&1
log "G: exited with $? ($(grep -E 'Learning iteration' logs/velstand_v3_train.log | tail -1 | tr -s ' '))"
VS3RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -v first-walk-src | head -1); VS3RUN=${VS3RUN%/}
VS3IT=$(latest_ckpt "$VS3RUN"); log "G: velstand v3 run $VS3RUN last checkpoint model_${VS3IT}.pt"

# ---------- H ----------
log "H: VelStand v3 push tests"
TASK="$RB" RUNDIR="$VS3RUN" FLAG=--walking PREFIX=velstand_v3 PREKEYS="" bash "$S/pushtest.sh" "$VS3IT" 2>&1 | tee -a "$PL"
TASK="$RB" RUNDIR="$VS3RUN" FLAG=--walking PREFIX=velstand_v3 PREKEYS="Up Up" bash "$S/pushtest.sh" "$VS3IT" 2>&1 | tee -a "$PL"
probe "exports/velstand_v3_${VS3IT}.onnx" --walking
log "PIPELINE DONE"
echo "Pipeline finished. Press Enter to close."; read -r _
