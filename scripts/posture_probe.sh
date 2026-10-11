#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/posture_probe.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Measure body pitch and leg joint angles before and after pushes, using the viewer's --debug prints.
#   posture_probe.sh <policy.onnx> [flag=--walking] [pushes=3]
export PATH="$HOME/.local/bin:$PATH"
ONNX="${1:?onnx}"; FLAG="${2:---walking}"; PUSHES="${3:-3}"; PREKEYS="${4:-}"   # e.g. "Up Up" = walk at 0.30 m/s before the pushes
cd "$HOME/microduck_rl" || exit 1
pkill -f infer_policy.py 2>/dev/null; tmux kill-session -t pp 2>/dev/null; sleep 1
LOG=/tmp/posture_probe.log; : > "$LOG"
tmux new-session -d -s pp -x 200 -y 50 "export PATH=\$HOME/.local/bin:\$PATH MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12; cd \$HOME/microduck_rl; uv run scripts/infer_policy.py $FLAG $ONNX --new-cmd-obs --debug; sleep 300"
tmux pipe-pane -o -t pp "cat >> $LOG"
for i in $(seq 1 240); do grep -q "Keyboard controls" "$LOG" 2>/dev/null && break; sleep 0.5; done
sleep 8
for k in $PREKEYS; do tmux send-keys -t pp "$k"; sleep 0.3; done
[ -n "$PREKEYS" ] && sleep 4
echo "MARK_PUSH_START" >> "$LOG"
for p in $(seq 1 "$PUSHES"); do tmux send-keys -t pp p; sleep 7; echo "MARK_PUSH_$p" >> "$LOG"; done
tmux kill-session -t pp 2>/dev/null; pkill -f infer_policy.py 2>/dev/null
python3 - "$LOG" <<'EOF'
import re, sys, math
txt = open(sys.argv[1], errors="ignore").read()
# split into debug blocks
blocks = re.split(r"={70}\nStep (\d+) DEBUG:", txt)
# blocks: [pre, step, body, step, body, ...]; find marker positions by scanning text order
marks = {m.start(): m.group(0) for m in re.finditer(r"MARK_PUSH_\w+", txt)}
def phase_of(pos):
    ph = "before"
    for mp, name in sorted(marks.items()):
        if mp < pos: ph = name.replace("MARK_PUSH_", "after_push_") if name != "MARK_PUSH_START" else "before"
    return ph
rows = []
for m in re.finditer(r"Step (\d+) DEBUG:(.*?)(?=\n={70}\nStep|\Z)", txt, re.S):
    step = int(m.group(1)); body = m.group(2)
    g = re.search(r"Proj grav \[3:6\]:\s*\[([^\]]+)\]", body)
    j = re.search(r"Joint pos \[6:20\]:\s*\[([^\]]+)\]", body)
    z = re.search(r"CoM height:\s*([-\d.]+)", body)
    if not (g and j and z): continue
    gv = [float(x) for x in g.group(1).split()]
    jv = [float(x) for x in j.group(1).split()]
    pitch = math.degrees(math.atan2(gv[0], -gv[2])) if len(gv) == 3 else float("nan")
    roll = math.degrees(math.atan2(-gv[1], -gv[2])) if len(gv) == 3 else float("nan")
    rows.append((step, phase_of(m.start()), float(z.group(1)), pitch, roll, jv))
names = ["l_hip_yaw","l_hip_roll","l_hip_pitch","l_knee","l_ankle","neck_pitch","head_pitch","head_yaw","head_roll","r_hip_yaw","r_hip_roll","r_hip_pitch","r_knee","r_ankle"]
print(f"{'step':>6} {'phase':<14} {'z[m]':>6} {'pitch°':>7} {'roll°':>6}  hip_pitch(L/R)  knee(L/R)  ankle(L/R)  neck/head_pitch")
for step, ph, z, pitch, roll, jv in rows:
    if len(jv) < 14: continue
    print(f"{step:>6} {ph:<14} {z:6.3f} {pitch:7.1f} {roll:6.1f}  {math.degrees(jv[2]):6.1f}/{math.degrees(jv[11]):6.1f}  {math.degrees(jv[3]):6.1f}/{math.degrees(jv[12]):6.1f}  {math.degrees(jv[4]):6.1f}/{math.degrees(jv[13]):6.1f}  {math.degrees(jv[5]):6.1f}/{math.degrees(jv[6]):6.1f}")
EOF
