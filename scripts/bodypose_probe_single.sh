#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/bodypose_probe_single.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Body-pose probe, one command per fresh viewer launch (no order effects): for each case, start the viewer,
# enter body mode (B), send the keys, wait, and report the measured crouch / pitch / roll of the last 2 s.
#   bodypose_probe_single.sh <policy.onnx> [flag=--standing]
export PATH="$HOME/.local/bin:$PATH"
ONNX="${1:?onnx}"; FLAG="${2:---standing}"
cd "$HOME/microduck_rl" || exit 1
OUT=/tmp/bodypose_single_summary.txt; : > "$OUT"
run_case() {  # <label> <keys...>
  local label="$1"; shift
  pkill -f infer_policy.py 2>/dev/null; tmux kill-session -t bs 2>/dev/null; sleep 1
  local LOG=/tmp/bodypose_single.log; : > "$LOG"
  tmux new-session -d -s bs -x 200 -y 50 "export PATH=\$HOME/.local/bin:\$PATH MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12; cd \$HOME/microduck_rl; uv run scripts/infer_policy.py $FLAG $ONNX --new-cmd-obs --debug; sleep 300"
  tmux pipe-pane -o -t bs "cat >> $LOG"
  for i in $(seq 1 240); do grep -q "Keyboard controls" "$LOG" 2>/dev/null && break; sleep 0.5; done
  sleep 5
  local n0; n0=$(wc -l < "$LOG")
  tmux send-keys -t bs b; sleep 0.5
  for k in "$@"; do tmux send-keys -t bs "$k"; sleep 0.4; done
  sleep 5
  tmux kill-session -t bs 2>/dev/null; pkill -f infer_policy.py 2>/dev/null
  python3 - "$label" "$LOG" "$n0" "$OUT" <<'EOF'
import re, sys, math
label, path, n0, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
lines = open(path, errors="ignore").read().split("\n")
pre, post = "\n".join(lines[:n0]), "\n".join(lines[n0:])
def rows(txt):
    r = []
    for m in re.finditer(r"Step (\d+) DEBUG:(.*?)(?=\n={70}\nStep|\Z)", txt, re.S):
        g = re.search(r"Proj grav \[3:6\]:\s*\[([^\]]+)\]", m.group(2)); z = re.search(r"CoM height:\s*([-\d.]+)", m.group(2))
        if not (g and z): continue
        gv = [float(x) for x in g.group(1).split()]
        if len(gv) != 3: continue
        r.append((float(z.group(1)), math.degrees(math.atan2(gv[0], -gv[2])), math.degrees(math.atan2(-gv[1], -gv[2]))))
    return r
b = rows(pre)[-3:]; a = rows(post)[-2:]
cmd = re.findall(r"Body cmd: x=\s*[-\d.]+mm\s+y=\s*[-\d.]+mm\s+z=\s*([-\d.]+)mm\s+roll=\s*([-\d.]+)°\s+pitch=\s*([-\d.]+)°", post)
c = cmd[-1] if cmd else ("?", "?", "?")
if b and a:
    bz = sum(r[0] for r in b)/len(b)
    mz = sum(r[0] for r in a)/len(a); mp = sum(r[1] for r in a)/len(a); mr = sum(r[2] for r in a)/len(a)
    line = f"{label:<10} cmd z={c[0]} roll={c[1]} pitch={c[2]} | measured dz={(mz-bz)*1000:+.1f} mm pitch={mp:+.1f} roll={mr:+.1f} (baseline z={bz:.3f})"
else:
    line = f"{label:<10} no data"
print(line); open(out, "a").write(line + "\n")
EOF
}
run_case "pitch+10" Left
run_case "pitch-10" Right
run_case "pitch+20" Left Left
run_case "pitch-20" Right Right
run_case "roll+10" a
run_case "roll-10" e
run_case "z-30" Down Down Down
run_case "z+20" Up Up
echo "===== $ONNX ($FLAG)"; cat "$OUT"
