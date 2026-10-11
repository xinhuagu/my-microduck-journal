#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/tmux_viewer.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Start infer_policy.py inside tmux session "duck" with the given args, wait until it is ready.
#   tmux_viewer.sh --standing exports/standup_500.onnx --new-cmd-obs
export PATH="$HOME/.local/bin:$PATH"
pkill -f infer_policy.py 2>/dev/null; sleep 2
tmux kill-session -t duck 2>/dev/null
cd "$HOME/microduck_rl" || exit 1
ARGS="$*"
tmux new-session -d -s duck -x 140 -y 50 \
  "export PATH=\$HOME/.local/bin:\$PATH MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12; cd \$HOME/microduck_rl; uv run scripts/infer_policy.py $ARGS; echo VIEWER_EXITED; sleep 600"
for i in $(seq 1 240); do
  if tmux capture-pane -p -S -200 -t duck 2>/dev/null | grep -qE "Keyboard controls|VIEWER_EXITED|Traceback"; then break; fi
  sleep 0.5
done
echo "ready after ~$((i/2)) s"
tmux capture-pane -p -S -200 -t duck | grep -E "Active policy|Walking policy|Standing policy|Observation size|Traceback|Error" | head -6
