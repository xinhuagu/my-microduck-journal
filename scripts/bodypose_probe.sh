#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/bodypose_probe.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Body-pose command probe: how far does the policy actually crouch / tilt when commanded (standing)?
#   bodypose_probe.sh <policy.onnx> [flag=--walking]
# Uses the viewer's --debug prints (CoM height, projected gravity) and tmux key injection:
# B (body-pose mode), Down = z -10 mm/key (viewer cap +-30 mm), Left/Right = pitch -/+10 deg, A/E = roll +/-10 deg.
export PATH="$HOME/.local/bin:$PATH"
ONNX="${1:?onnx}"; FLAG="${2:---walking}"
cd "$HOME/microduck_rl" || exit 1
pkill -f infer_policy.py 2>/dev/null; tmux kill-session -t bp 2>/dev/null; sleep 1
LOG=/tmp/bodypose_probe.log; : > "$LOG"
tmux new-session -d -s bp -x 200 -y 50 "export PATH=\$HOME/.local/bin:\$PATH MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12; cd \$HOME/microduck_rl; uv run scripts/infer_policy.py $FLAG $ONNX --new-cmd-obs --debug; sleep 300"
tmux pipe-pane -o -t bp "cat >> $LOG"
for i in $(seq 1 240); do grep -q "Keyboard controls" "$LOG" 2>/dev/null && break; sleep 0.5; done
sleep 6
mark() { echo "MARK_$1" >> "$LOG"; }
mark baseline; sleep 4
tmux send-keys -t bp b; sleep 0.5
# crouch
tmux send-keys -t bp Down; sleep 0.3; tmux send-keys -t bp Down; mark z-20; sleep 4
tmux send-keys -t bp Down; mark z-30; sleep 4
tmux send-keys -t bp Space; mark z0; sleep 3
# raise
tmux send-keys -t bp Up; sleep 0.3; tmux send-keys -t bp Up; mark z+20; sleep 4
tmux send-keys -t bp Space; sleep 3
# pitch
tmux send-keys -t bp Left; mark pitch-10; sleep 4
tmux send-keys -t bp Left; mark pitch-20; sleep 4
tmux send-keys -t bp Space; sleep 3
tmux send-keys -t bp Right; mark pitch+10; sleep 4
tmux send-keys -t bp Space; sleep 3
# roll
tmux send-keys -t bp a; mark roll+10; sleep 4
tmux send-keys -t bp Space; sleep 3
mark end; sleep 2
tmux kill-session -t bp 2>/dev/null; pkill -f infer_policy.py 2>/dev/null
python3 - "$LOG" <<'EOF'
import re, sys, math
txt = open(sys.argv[1], errors="ignore").read()
marks = sorted((m.start(), m.group(1)) for m in re.finditer(r"MARK_(\S+)", txt))
def phase_of(pos):
    ph = "start"
    for mp, name in marks:
        if mp < pos: ph = name
    return ph
cmds = {}
for m in re.finditer(r"Body cmd: x=\s*([-\d.]+)mm\s+y=\s*([-\d.]+)mm\s+z=\s*([-\d.]+)mm\s+roll=\s*([-\d.]+)°\s+pitch=\s*([-\d.]+)°", txt):
    cmds[phase_of(m.start())] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
rows = {}
for m in re.finditer(r"Step (\d+) DEBUG:(.*?)(?=\n={70}\nStep|\Z)", txt, re.S):
    body = m.group(2)
    g = re.search(r"Proj grav \[3:6\]:\s*\[([^\]]+)\]", body)
    z = re.search(r"CoM height:\s*([-\d.]+)", body)
    if not (g and z): continue
    gv = [float(x) for x in g.group(1).split()]
    if len(gv) != 3: continue
    pitch = math.degrees(math.atan2(gv[0], -gv[2])); roll = math.degrees(math.atan2(-gv[1], -gv[2]))
    rows.setdefault(phase_of(m.start()), []).append((float(z.group(1)), pitch, roll))
base = rows.get("baseline") or rows.get("start") or [(0.0, 0.0, 0.0)]
bz = sum(r[0] for r in base) / len(base)
print(f"{'phase':<10} {'cmd z[mm]':>9} {'cmd roll':>8} {'cmd pitch':>9} | {'meas dz[mm]':>11} {'meas pitch':>10} {'meas roll':>9}  (last 2 s of each phase; dz relative to baseline z={bz:.3f})")
order = [n for _, n in marks]
for ph in order:
    rs = rows.get(ph, [])
    if not rs: continue
    last = rs[-2:] if len(rs) >= 2 else rs
    mz = sum(r[0] for r in last) / len(last); mp = sum(r[1] for r in last) / len(last); mr = sum(r[2] for r in last) / len(last)
    c = cmds.get(ph, (0.0, 0.0, 0.0))
    print(f"{ph:<10} {c[0]:9.1f} {c[1]:8.1f} {c[2]:9.1f} | {(mz - bz) * 1000:11.1f} {mp:10.1f} {mr:9.1f}")
EOF
