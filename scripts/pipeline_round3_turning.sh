#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round3_turning.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline, part 4 (turning round), logged to logs/pipeline4.log. Waits for part 3 (logs/pipeline3.log), then:
#   M. walk sweep over VelStand v4 checkpoints (yaw drift at 0.30 m/s, turn 0.6) -> pick the cleanest (V4PICK)
#   N. walk teacher v2: Velocity task, warm start from first-walk model_10000 with MICRODUCK_TURN=1, 3000 iters
#   O. sweep the walk teacher's checkpoints 1500..3000 (drift, turn 0.3 / 0.6 / 1.0) -> pick (WALKPICK)
#   P. VelStand v5: same task as v3/v4, warm start from V4PICK, expert = teacher v3, anchor = WALKPICK,
#      MICRODUCK_TURN=1 + MICRODUCK_BODY_WIDE=1, 3000 iters
#   Q. sweep v5 -> pick; push tests, posture probe, body probe, full turn probe on the pick
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline4.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
latest_ckpt() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1; }
ckpts_from() {  # <rundir> <min_iter>: every checkpoint >= min_iter, ascending
  ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | awk -v m="$2" '$1 >= m'
}
# sweep + pick: prints the chosen iteration. Score = turn0.3 + turn0.6 + 0.5*turn1.0 - 2*|drift| (only the cases run count)
sweep_pick() {  # <rundir> <prefix> <task> <cases> <iters...>
  local rundir="$1" prefix="$2" task="$3" cases="$4"; shift 4
  RUNDIR="$rundir" PREFIX="$prefix" TASK="$task" CASES="$cases" EXPERT="${TEACHER:-}" ANCHOR="${WALK:-}" \
    bash "$S/turn_sweep.sh" "$@" > "/tmp/sweep_${prefix}.txt" 2>&1
  cat "/tmp/sweep_${prefix}.txt" | tee -a "$PL" >/dev/null
  python3 - "/tmp/sweep_${prefix}.txt" <<'EOF'
import re, sys
best, cur, rows = None, None, {}
for line in open(sys.argv[1], errors="ignore"):
    m = re.match(r"--- checkpoint (\d+)", line)
    if m: cur = int(m.group(1)); rows[cur] = {}; continue
    m = re.match(r"(turn [+-]?[0-9.]+|walk [0-9.]+)\s+yaw_rate=([-+0-9.]+)", line)
    if m and cur is not None: rows[cur][m.group(1)] = float(m.group(2))
def score(r):
    s = 0.0
    s += r.get("turn +0.3", 0.0) + r.get("turn +0.6", 0.0) + 0.5 * r.get("turn +1.0", 0.0)
    s -= 2.0 * abs(r.get("walk 0.30", 0.0))
    return s
for it, r in rows.items():
    if r and (best is None or score(r) > score(rows[best])): best = it
print(best if best is not None else "")
EOF
}
probe() {  # <onnx> <flag>
  bash "$S/posture_probe.sh" "$1" "$2" 3 > /tmp/probe_out.txt 2>&1
  cat /tmp/probe_out.txt | tee -a "$PL"
}

log "PIPELINE PART 4 START (waiting for part 3)"
until grep -qE "PIPELINE DONE \(part 3\)|PIPELINE STOP" logs/pipeline3.log 2>/dev/null; do sleep 60; done
log "part 3 finished: $(grep -E 'PIPELINE DONE|PIPELINE STOP' logs/pipeline3.log | tail -1)"
export WALK=logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt
TEACHER=$(grep -oE "I: teacher v3 = \S+" logs/pipeline3.log | tail -1 | sed 's/.*= //')
[ -f "$TEACHER" ] || { log "teacher v3 not found ($TEACHER)"; exit 1; }
export TEACHER
TASK=$(grep -oE "G: VelStand v3 \([A-Za-z0-9-]+\)" logs/pipeline.log | tail -1 | sed -E 's/.*\((.*)\)/\1/')
[ -z "$TASK" ] && TASK=Mjlab-VelStand-Flat-MicroDuck
V4RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -v first-walk-src | head -1); V4RUN=${V4RUN%/}
log "inputs: teacher=$TEACHER task=$TASK v4 run=$V4RUN"

# ---------- M ----------
log "M: walk sweep over VelStand v4 checkpoints (>= 1500)"
export MICRODUCK_BODY_WIDE=1
V4PICK=$(sweep_pick "$V4RUN" velstand_v4 "$TASK" "walk0.30 turn0.6" $(ckpts_from "$V4RUN" 1500))
[ -z "$V4PICK" ] && V4PICK=$(latest_ckpt "$V4RUN")
log "M: v4 pick = model_${V4PICK}.pt"

# ---------- N ----------
log "N: walk teacher v2 (Velocity, MICRODUCK_TURN=1), warm start from first-walk model_10000, 3000 iterations"
MICRODUCK_WARM_START=1 MICRODUCK_TURN=1 \
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run 2026-09-07_20-38-41_first-walk --agent.load-checkpoint model_10000.pt \
  --agent.max-iterations 3000 > logs/walk_v2_train.log 2>&1
log "N: exited with $? ($(grep -E 'Learning iteration' logs/walk_v2_train.log | tail -1 | tr -s ' '))"
W2RUN=$(ls -dt logs/rsl_rl/velocity/*/ | head -1); W2RUN=${W2RUN%/}
log "N: walk v2 run $W2RUN last checkpoint model_$(latest_ckpt "$W2RUN").pt"

# ---------- O ----------
log "O: sweep walk v2 checkpoints (>= 1500): drift + turn 0.3 / 0.6 / 1.0"
WALKPICK=$(sweep_pick "$W2RUN" walk_v2 Mjlab-Velocity-Flat-MicroDuck "walk0.30 turn0.3 turn0.6 turn1.0" $(ckpts_from "$W2RUN" 1500))
[ -z "$WALKPICK" ] && WALKPICK=$(latest_ckpt "$W2RUN")
export WALK="$W2RUN/model_${WALKPICK}.pt"; log "O: walk v2 pick = $WALK"
ln -sfn "../velocity/$(basename "$W2RUN")" logs/rsl_rl/velstand/walk-v2-src

# ---------- P ----------
log "P: VelStand v5 ($TASK, TURN + BODY_WIDE), warm start from $V4RUN/model_${V4PICK}.pt, expert = $TEACHER, anchor = $WALK, 3000 iterations"
MICRODUCK_WARM_START=1 MICRODUCK_TURN=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK" \
uv run train "$TASK" --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$(basename "$V4RUN")" --agent.load-checkpoint "model_${V4PICK}.pt" \
  --agent.max-iterations 3000 > logs/velstand_v5_train.log 2>&1
log "P: exited with $? ($(grep -E 'Learning iteration' logs/velstand_v5_train.log | tail -1 | tr -s ' '))"
V5RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); V5RUN=${V5RUN%/}

# ---------- Q ----------
log "Q: sweep v5 (>= 1500) + tests on the pick"
export MICRODUCK_TURN=1
V5PICK=$(sweep_pick "$V5RUN" velstand_v5 "$TASK" "walk0.30 turn0.3 turn0.6" $(ckpts_from "$V5RUN" 1500))
[ -z "$V5PICK" ] && V5PICK=$(latest_ckpt "$V5RUN")
log "Q: v5 pick = $V5RUN/model_${V5PICK}.pt"
export EXPERT="$TEACHER" ANCHOR="$WALK"
TASK="$TASK" RUNDIR="$V5RUN" FLAG=--walking PREFIX=velstand_v5 PREKEYS="" bash "$S/pushtest.sh" "$V5PICK" 2>&1 | tee -a "$PL"
TASK="$TASK" RUNDIR="$V5RUN" FLAG=--walking PREFIX=velstand_v5 PREKEYS="Up Up" bash "$S/pushtest.sh" "$V5PICK" 2>&1 | tee -a "$PL"
probe "exports/velstand_v5_${V5PICK}.onnx" --walking
log "body-pose probe v5"; bash "$S/bodypose_probe.sh" "exports/velstand_v5_${V5PICK}.onnx" --walking 2>&1 | tee -a "$PL"
log "turn probe v5 (all cases)"; bash "$S/turn_probe.sh" "exports/velstand_v5_${V5PICK}.onnx" --walking 12 2>&1 | grep -E "^(turn|walk) " | tee -a "$PL"
log "PIPELINE DONE (part 4)"
echo "Pipeline part 4 finished. Press Enter to close."; read -r _
