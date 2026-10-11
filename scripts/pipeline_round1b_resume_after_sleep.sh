#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round1b_resume_after_sleep.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline, part 2 (the PC went to sleep at 12:34 during VelStand v2, iteration 2784; last checkpoint model_2750):
#   E2. resume VelStand v2 (Flat) from model_2750 to 6000 iterations (plain resume: curricula continue)
#   F.  push tests (standing + walking) + probe on VelStand v2
#   G.  VelStand v3 (Rough-Backlash), 3000 iterations: warm start from VelStand v2
#   H.  push tests on VelStand v3
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WALK=logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt
export TEACHER=logs/rsl_rl/microduck_stand/2026-10-11_08-37-10_microduck_stand/model_7498.pt
VS2SRC=2026-10-11_11-04-40_velstand
VS2FROM=2750
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
latest_ckpt() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1; }

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

log "PIPELINE PART 2 START (PC slept at 12:34 during E at iteration 2784; resuming from $VS2SRC/model_${VS2FROM}.pt)"
[ -f "$TEACHER" ] || { log "teacher missing: $TEACHER"; exit 1; }
[ -f "logs/rsl_rl/velstand/$VS2SRC/model_${VS2FROM}.pt" ] || { log "velstand v2 checkpoint missing"; exit 1; }

# ---------- E2 ----------
REMAIN=$((6000 - VS2FROM))
log "E2: resume VelStand v2 (Flat) from model_${VS2FROM}.pt for $REMAIN more iterations (-> 6000), expert=$TEACHER anchor=$WALK"
MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK" \
uv run train Mjlab-VelStand-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$VS2SRC" --agent.load-checkpoint "model_${VS2FROM}.pt" \
  --agent.max-iterations "$REMAIN" > logs/velstand_v2_resume.log 2>&1
log "E2: exited with $? ($(grep -E 'Learning iteration' logs/velstand_v2_resume.log | tail -1 | tr -s ' '))"
VS2RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -v first-walk-src | head -1); VS2RUN=${VS2RUN%/}
VS2IT=$(latest_ckpt "$VS2RUN"); log "E2: velstand v2 run $VS2RUN last checkpoint model_${VS2IT}.pt"

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
