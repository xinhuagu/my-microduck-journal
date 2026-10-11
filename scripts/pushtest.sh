#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/pushtest.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Generic push test. Exports checkpoints of a run, loads each ONNX in the CPU viewer (tmux, pane
# captured with pipe-pane), optionally sends pre-keys (e.g. Up = walk), then N random pushes.
#   TASK=<task id> RUNDIR=<run dir> FLAG=--walking PREFIX=velstand PREKEYS="Up" pushtest.sh 1250 2000 2999
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd "$HOME/microduck_rl" || exit 1
TASK="${TASK:?task id}"; RUNDIR="${RUNDIR:?run dir}"; FLAG="${FLAG:---walking}"; PREFIX="${PREFIX:-policy}"
PREKEYS="${PREKEYS:-}"; PUSHES="${PUSHES:-8}"; WINDOW_S="${WINDOW_S:-6}"; RECOVER_MM="${RECOVER_MM:-100}"
mkdir -p exports
for it in "$@"; do
  onnx="exports/${PREFIX}_${it}.onnx"
  if [ ! -s "$onnx" ]; then
    # VelStand's runner resolves its BC teachers at construction time (also during export) -> point it at local files
    MICRODUCK_STAND_EXPERT="${EXPERT:-}" MICRODUCK_WALK_ANCHOR="${ANCHOR:-}" \
    uv run scripts/export.py "$TASK" --checkpoint-file "$RUNDIR/model_${it}.pt" --onnx-file "$onnx" > /tmp/export_${PREFIX}_${it}.log 2>&1 || { echo "iter $it: EXPORT FAILED"; tail -2 /tmp/export_${PREFIX}_${it}.log; continue; }
  fi
  pkill -f infer_policy.py 2>/dev/null; tmux kill-session -t pt 2>/dev/null; sleep 1
  LOG=/tmp/pushtest_${PREFIX}_${it}.log; : > "$LOG"
  tmux new-session -d -s pt -x 160 -y 50 "export PATH=\$HOME/.local/bin:\$PATH MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12; cd \$HOME/microduck_rl; uv run scripts/infer_policy.py $FLAG $onnx --new-cmd-obs; sleep 300"
  tmux pipe-pane -o -t pt "cat >> $LOG"
  for i in $(seq 1 240); do grep -q "Keyboard controls" "$LOG" 2>/dev/null && break; sleep 0.5; done
  sleep 5
  for k in $PREKEYS; do tmux send-keys -t pt "$k"; sleep 0.3; done
  [ -n "$PREKEYS" ] && sleep 4
  base=$(grep -oE 'trunk_z=[0-9.]+' "$LOG" | tail -1 | sed 's/trunk_z=//')
  fwd=$(grep -oE 'fwd=[-+0-9.]+/[-+0-9.]+' "$LOG" | tail -1)
  ok=0; fell=0; moved=0
  for p in $(seq 1 "$PUSHES"); do
    n0=$(wc -l < "$LOG")
    tmux send-keys -t pt p
    sleep "$WINDOW_S"
    tr=$(tail -n +$((n0+1)) "$LOG" | grep -oE 'trunk_z=[0-9.]+' | sed 's/trunk_z=//')
    mn=$(echo "$tr" | sort -n | head -1); last=$(echo "$tr" | tail -1)
    lastfwd=$(tail -n +$((n0+1)) "$LOG" | grep -oE 'fwd=[-+0-9.]+/[-+0-9.]+' | tail -1)
    [ -n "$last" ] && [ "$(echo "$last >= $RECOVER_MM" | bc)" = "1" ] && ok=$((ok+1))
    [ -n "$mn" ] && [ "$(echo "$mn < 90" | bc)" = "1" ] && fell=$((fell+1))
    [ -n "$mn" ] && [ -n "$base" ] && [ "$(echo "($base - $mn) > 3" | bc)" = "1" ] && moved=$((moved+1))
    echo "  iter $it push $p: min=${mn:-?} mm  end=${last:-?} mm  ${lastfwd:-}"
  done
  echo "iter $it: start trunk_z ${base:-?} mm ${fwd:-} | moved ${moved}/${PUSHES} | went down (<90 mm) ${fell}/${PUSHES} | upright at end ${ok}/${PUSHES}"
  [ "$moved" -eq 0 ] && echo "  WARNING: no push changed trunk_z - keys probably not delivered"
  tmux kill-session -t pt 2>/dev/null; pkill -f infer_policy.py 2>/dev/null
done
