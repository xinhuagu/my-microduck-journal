#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pipeline_round6_candidate_C.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Pipeline 7 — v0.2 candidate C, logged to logs/pipeline7.log.
#   Patch (env-gated, default unchanged): MICRODUCK_GROUND_MIX=1 -> more episodes start lying on the side / back:
#     final ground-spawn stage prone 0.45 -> 0.55, face-down share 0.50 -> 0.30, side share 0.5 -> 0.6 in every stage
#     (side 22 -> 33 % of envs, face-up 11 -> 15 %, face-down 11 -> 7 %).
#   C = A's configuration (v4 warm start, fast walk anchor, no in-network speed term) + stand-expert imitation x3
#       on fallen frames (MICRODUCK_EXPERT_BC_COEF=3.0) + MICRODUCK_GROUND_MIX=1; 3000 iterations, Rough-Backlash.
#   Then: smoke first; battery sweep (tracking + push) 2000..2999 -> pick -> 3-seed battery -> export -> release gate.
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/microduck_rl" || exit 1
PL=logs/pipeline7.log
log() { echo "=== $(date -Is) $*" | tee -a "$PL"; }
ckpts_between() { ls "$1" | grep -E 'model_[0-9]+\.pt' | sed -E 's/model_([0-9]+)\.pt/\1/' | sort -n | awk -v a="$2" -v b="$3" '$1 >= a && $1 <= b'; }
TEACHER=logs/rsl_rl/microduck_stand/2026-10-11_17-53-45_microduck_stand/model_2999.pt
WALK3=logs/rsl_rl/velocity/2026-10-15_22-09-39_velocity/model_2000.pt
V4RUN=2026-10-11_19-13-04_velstand
TASK=Mjlab-VelStand-Rough-Backlash-MicroDuck
VS=src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py

# ---------- patch ----------
python3 - "$VS" <<'EOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
if "MICRODUCK_GROUND_MIX" in s:
    print("velstand cfg: ground-mix knob already present"); sys.exit(0)
anchor = '''    {"step": _PRONE_ITERS[3] * NUM_STEPS_PER_ENV, "params": {"prone_prob": 0.45, "face_down_prob": 0.50, "side_prob": PRONE_SIDE_PROB, "crouch_prob": 0.20}},
]
'''
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''# Local tweak (my-microduck-journal): MICRODUCK_GROUND_MIX=1 — more episodes start lying on the side / back. The
# fast walk anchor erodes the get-up from those states (push failures end at z 64 mm, tilt ~97 deg); keep it practised.
if os.environ.get("MICRODUCK_GROUND_MIX"):
    for _st in PRONE_RAMP_STAGES:
        _st["params"]["side_prob"] = 0.6
    PRONE_RAMP_STAGES[3]["params"].update({"prone_prob": 0.40, "face_down_prob": 0.45})
    PRONE_RAMP_STAGES[4]["params"].update({"prone_prob": 0.55, "face_down_prob": 0.30})
    print("[velstand] MICRODUCK_GROUND_MIX: ground spawns final prone 0.55, side 0.6, face-down 0.30 of the rest")
''', 1)
p.write_text(s); print("velstand cfg: MICRODUCK_GROUND_MIX added")
EOF
export MICRODUCK_TURN=1 MICRODUCK_SPEED=1 MICRODUCK_BODY_WIDE=1 MICRODUCK_SPEED_FINE_W=0 MICRODUCK_EXPERT_BC_COEF=3.0 MICRODUCK_GROUND_MIX=1
export MICRODUCK_STAND_EXPERT="$TEACHER" MICRODUCK_WALK_ANCHOR="$WALK3" EXPERT="$TEACHER" ANCHOR="$WALK3"
log "PIPELINE 7 START (v0.2 candidate C); GPU power limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"

# ---------- smoke ----------
MICRODUCK_WARM_START=1 uv run train "$TASK" --env.scene.num-envs 64 --agent.max-iterations 3 \
  --agent.resume True --agent.load-run "$V4RUN" --agent.load-checkpoint model_2999.pt > logs/pipeline7_smoke.log 2>&1
RC=$?
if [ $RC -ne 0 ] || grep -q Traceback logs/pipeline7_smoke.log; then log "PIPELINE STOP: smoke failed (exit $RC)"; tail -5 logs/pipeline7_smoke.log | tee -a "$PL"; exit 1; fi
log "smoke OK: $(sed 's/\x1b\[[0-9;]*m//g' logs/pipeline7_smoke.log | grep -aoE '\[velstand\] MICRODUCK_GROUND_MIX[^$]*' | head -1); $(sed 's/\x1b\[[0-9;]*m//g' logs/pipeline7_smoke.log | grep -aoE 'stand expert \(iter [0-9]+\) on tilt>[0-9.]+° coef=[0-9.]+' | head -1)"

# ---------- C ----------
log "C: warm start $V4RUN/model_2999.pt, anchor $WALK3, SPEED_FINE_W=0, EXPERT_BC_COEF=3.0, GROUND_MIX=1, 3000 iterations"
MICRODUCK_WARM_START=1 uv run train "$TASK" --env.scene.num-envs 4096 --agent.resume True --agent.load-run "$V4RUN" --agent.load-checkpoint model_2999.pt \
  --agent.max-iterations 3000 > logs/cand_C_train.log 2>&1
log "C: exited with $? ($(grep -E 'Learning iteration' logs/cand_C_train.log | tail -1 | tr -s ' '))"
CRUN=$(ls -dt logs/rsl_rl/velstand/*/ | grep -vE "first-walk-src|walk-v2-src" | head -1); CRUN=${CRUN%/}
log "C: run $CRUN"

SW=logs/eval_sweep_cand_C_v02; mkdir -p "$SW"
log "C: battery sweep (tracking + push) on checkpoints 2000..2999"
for it in $(ckpts_between "$CRUN" 2000 2999); do
  bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$CRUN/model_${it}.pt" 256 tracking,push "$SW/cand_C_v02_${it}.json" 0 "cand_C_v02_${it}" > /dev/null 2>&1
done
python3 "$S/pick_from_battery.py" "$SW" release | tee -a "$PL"
PICK=$(python3 "$S/pick_from_battery.py" "$SW" release | grep '^PICK' | awk '{print $2}')
log "C: pick = $CRUN/model_${PICK}.pt; full battery (3 seeds, incl. goal)"
bash "$S/eval_battery.sh" Mjlab-VelStand-Flat-MicroDuck "$CRUN/model_${PICK}.pt" 256 getup,push,tracking,body,goal logs/eval/cand_C_v02.json 0,1,2 cand_C_v02 2>&1 \
  | grep -E "^\[battery\] (done|wrote)|Traceback" | tee -a "$PL"
uv run scripts/export.py "$TASK" --checkpoint-file "$CRUN/model_${PICK}.pt" --onnx-file "exports/cand_C_v02_${PICK}.onnx" > /tmp/export_C.log 2>&1 \
  && log "C: exported exports/cand_C_v02_${PICK}.onnx" || log "C: export failed (/tmp/export_C.log)"

log "RELEASE GATE"
python3 "$S/release_gate.py" logs/eval/cand_C_v02.json logs/eval/cand_A_v02.json logs/eval/cand_B_v02.json logs/eval/ours_velstand_v4.json 2>&1 | tee -a "$PL"
log "PIPELINE DONE (7)"
echo "Pipeline 7 finished. Press Enter to close."; read -r _
