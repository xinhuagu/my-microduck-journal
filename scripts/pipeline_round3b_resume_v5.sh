#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round3b_resume_v5.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline part 4, resumed after the shutdown of 2026-10-12 00:31 (VelStand v5 stopped at iteration 253, checkpoint model_250).
#   P'. resume VelStand v5 (plain resume: curricula continue) for the remaining 2750 iterations
#   Q.  sweep v5 (>= 1500) -> pick; push tests, posture probe, body-pose probe (single-launch), turn probe
# Logs to logs/pipeline4.log, same markers as before.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline4.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
latest_ckpt() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | tail -1; }
ckpts_from() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | awk -v m="$2" '$1 >= m'; }
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
    return r.get("turn +0.3", 0.0) + r.get("turn +0.6", 0.0) + 0.5 * r.get("turn +1.0", 0.0) - 2.0 * abs(r.get("walk 0.30", 0.0))
for it, r in rows.items():
    if r and (best is None or score(r) > score(rows[best])): best = it
print(best if best is not None else "")
EOF
}

export TEACHER=logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt
export WALK=logs/rsl_rl/velocity/2026-10-11_22-59-41_velocity/model_1750.pt
TASK=Mjlab-VelStand-Rough-Backlash-MicroDuck
V5SRC=2026-10-12_00-14-51_velstand
V5FROM=$(latest_ckpt "logs/rsl_rl/velstand/$V5SRC")
REMAIN=$((3000 - V5FROM))
[ -f "$TEACHER" ] && [ -f "$WALK" ] && [ -f "logs/rsl_rl/velstand/$V5SRC/model_${V5FROM}.pt" ] || { echo "inputs missing"; exit 1; }

log "PART 4 RESUME: VelStand v5 from $V5SRC/model_${V5FROM}.pt for $REMAIN more iterations (TURN + BODY_WIDE, expert=$TEACHER, anchor=$WALK)"
MICRODUCK_TURN=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK" \
uv run train "$TASK" --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run "$V5SRC" --agent.load-checkpoint "model_${V5FROM}.pt" \
  --agent.max-iterations "$REMAIN" > logs/velstand_v5_resume.log 2>&1
log "P': exited with $? ($(grep -E 'Learning iteration' logs/velstand_v5_resume.log | tail -1 | tr -s ' '))"
V5RUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); V5RUN=${V5RUN%/}

log "Q: sweep v5 (>= 1500) + tests on the pick"
export MICRODUCK_TURN=1 MICRODUCK_BODY_WIDE=1
V5PICK=$(sweep_pick "$V5RUN" velstand_v5 "$TASK" "walk0.30 turn0.3 turn0.6" $(ckpts_from "$V5RUN" 1500))
[ -z "$V5PICK" ] && V5PICK=$(latest_ckpt "$V5RUN")
log "Q: v5 pick = $V5RUN/model_${V5PICK}.pt"
export EXPERT="$TEACHER" ANCHOR="$WALK"
TASK="$TASK" RUNDIR="$V5RUN" FLAG=--walking PREFIX=velstand_v5 PREKEYS="" bash "$S/pushtest.sh" "$V5PICK" 2>&1 | tee -a "$PL"
TASK="$TASK" RUNDIR="$V5RUN" FLAG=--walking PREFIX=velstand_v5 PREKEYS="Up Up" bash "$S/pushtest.sh" "$V5PICK" 2>&1 | tee -a "$PL"
bash "$S/posture_probe.sh" "exports/velstand_v5_${V5PICK}.onnx" --walking 3 2>&1 | tee -a "$PL"
log "body-pose probe v5 (single-launch)"; bash "$S/bodypose_probe_single.sh" "exports/velstand_v5_${V5PICK}.onnx" --walking 2>&1 | tee -a "$PL"
log "turn probe v5"; bash "$S/turn_probe.sh" "exports/velstand_v5_${V5PICK}.onnx" --walking 12 2>&1 | grep -E "^(turn|walk) " | tee -a "$PL"
log "PIPELINE DONE (part 4)"
echo "Pipeline part 4 finished. Press Enter to close."; read -r _
