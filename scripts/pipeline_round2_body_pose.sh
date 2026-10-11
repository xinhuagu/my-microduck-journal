#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round2_body_pose.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline, part 3 (body-pose round), logged to logs/pipeline3.log. Waits for part 2 (logs/pipeline.log DONE/STOP), then:
#   I. StandUp teacher v3: warm start from model_7498 with MICRODUCK_STANDUP_STRAIGHT=1 + MICRODUCK_BODY_WIDE=1, 3000 iters
#   J. export + posture probe (must still end upright) + body-pose probe (must crouch / tilt)
#   K. VelStand v4: same task as v3, warm start from the newest velstand run, expert = teacher v3, BODY_WIDE, 3000 iters
#   L. export + push tests (standing + walking) + posture probe + body-pose probe
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WALK=logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt
T2RUN=2026-10-11_08-37-10_microduck_stand
T2IT=7498
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline3.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
latest_ckpt() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1; }

probe() {  # <onnx> <flag>  -> posture probe table + "PROBE mean_abs_pitch_last4=.. min_z=.."
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
bodyprobe() {  # <onnx> <flag>
  log "body-pose probe: $1 $2"
  bash "$S/bodypose_probe.sh" "$1" "$2" 2>&1 | tee -a "$PL"
}

log "PIPELINE PART 3 START (waiting for part 2)"
until grep -qE "PIPELINE DONE|PIPELINE STOP" logs/pipeline.log 2>/dev/null; do sleep 60; done
log "part 2 finished: $(grep -E 'PIPELINE DONE|PIPELINE STOP' logs/pipeline.log | tail -1)"

# ---------- I ----------
log "I: StandUp teacher v3 (STRAIGHT + BODY_WIDE), warm start from $T2RUN/model_${T2IT}.pt, 3000 iterations"
MICRODUCK_WARM_START=1 MICRODUCK_STANDUP_STRAIGHT=1 MICRODUCK_BODY_WIDE=1 \
uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$T2RUN" --agent.load-checkpoint "model_${T2IT}.pt" \
  --agent.max-iterations 3000 > logs/standup_v3_train.log 2>&1
log "I: exited with $? ($(grep -E 'Learning iteration' logs/standup_v3_train.log | tail -1 | tr -s ' '))"
T3RUN=$(ls -dt logs/rsl_rl/microduck_stand/*/ | head -1); T3RUN=${T3RUN%/}
T3IT=$(latest_ckpt "$T3RUN"); export TEACHER="$T3RUN/model_${T3IT}.pt"; log "I: teacher v3 = $TEACHER"

# ---------- J ----------
log "J: export + probes of teacher v3"
MICRODUCK_STANDUP_STRAIGHT=1 MICRODUCK_BODY_WIDE=1 uv run scripts/export.py Mjlab-StandUp-Flat-MicroDuck \
  --checkpoint-file "$TEACHER" --onnx-file "exports/standup_v3_${T3IT}.onnx" > /tmp/export_v3.log 2>&1 || { log "J: EXPORT FAILED"; tail -3 /tmp/export_v3.log | tee -a "$PL"; }
probe "exports/standup_v3_${T3IT}.onnx" --standing
MP=$(grep -oE "PROBE mean_abs_pitch_last4=[0-9.]+" "$PL" | tail -1 | sed 's/.*=//')
if [ "$(echo "$MP > 12" | bc)" = "1" ]; then log "PIPELINE STOP (part 3): teacher v3 leans again (mean |pitch| = ${MP} deg)"; exit 1; fi
log "J: teacher v3 ends upright (mean |pitch| = ${MP} deg)"
bodyprobe "exports/standup_v3_${T3IT}.onnx" --standing

# ---------- K ----------
TASK=$(grep -oE "G: VelStand v3 \([A-Za-z0-9-]+\)" logs/pipeline.log | tail -1 | sed -E 's/.*\((.*)\)/\1/')
[ -z "$TASK" ] && TASK=Mjlab-VelStand-Flat-MicroDuck
VSRC=$(ls -dt logs/rsl_rl/velstand/*/ | grep -v first-walk-src | head -1); VSRC=${VSRC%/}
VSIT=$(latest_ckpt "$VSRC")
log "K: VelStand v4 ($TASK, BODY_WIDE), warm start from $VSRC/model_${VSIT}.pt, expert = $TEACHER, 3000 iterations"
MICRODUCK_WARM_START=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK" \
uv run train "$TASK" --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$(basename "$VSRC")" --agent.load-checkpoint "model_${VSIT}.pt" \
  --agent.max-iterations 3000 > logs/velstand_v4_train.log 2>&1
log "K: exited with $? ($(grep -E 'Learning iteration' logs/velstand_v4_train.log | tail -1 | tr -s ' '))"
VS4RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -v first-walk-src | head -1); VS4RUN=${VS4RUN%/}
VS4IT=$(latest_ckpt "$VS4RUN"); log "K: velstand v4 run $VS4RUN last checkpoint model_${VS4IT}.pt"

# ---------- L ----------
log "L: VelStand v4 push tests + probes"
export EXPERT="$TEACHER" ANCHOR="$WALK" MICRODUCK_BODY_WIDE=1
TASK="$TASK" RUNDIR="$VS4RUN" FLAG=--walking PREFIX=velstand_v4 PREKEYS="" bash "$S/pushtest.sh" "$VS4IT" 2>&1 | tee -a "$PL"
TASK="$TASK" RUNDIR="$VS4RUN" FLAG=--walking PREFIX=velstand_v4 PREKEYS="Up Up" bash "$S/pushtest.sh" "$VS4IT" 2>&1 | tee -a "$PL"
probe "exports/velstand_v4_${VS4IT}.onnx" --walking
bodyprobe "exports/velstand_v4_${VS4IT}.onnx" --walking
log "PIPELINE DONE (part 3)"
echo "Pipeline part 3 finished. Press Enter to close."; read -r _
