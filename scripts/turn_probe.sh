#!/usr/bin/env bash
# Exact script used for the runs in docs/training-log.md (my-microduck-journal). Run inside WSL from
# any directory: `bash scripts/turn_probe.sh` — it cd's into ~/microduck_rl itself. Companion scripts are
# looked up next to this file.
# Turn / yaw probe: achieved yaw rate for in-place turn commands, and yaw drift + forward speed for
# straight walking. Uses the viewer's --ang-vel-z / --lin-vel-x initial commands and its --debug prints
# (Ang vel [0:3] = base angular velocity, body frame, rad/s; fwd= status = achieved / commanded m/s).
#   turn_probe.sh <policy.onnx> [flag=--walking] [window_s=12]
export PATH="$HOME/.local/bin:$PATH"
ONNX="${1:?onnx}"; FLAG="${2:---walking}"; WIN="${3:-12}"
cd "$HOME/microduck_rl" || exit 1
OUT=/tmp/turn_probe_summary.txt; : > "$OUT"
run_case() {  # <label> <extra viewer args>
  local label="$1"; shift
  pkill -f infer_policy.py 2>/dev/null; tmux kill-session -t tp 2>/dev/null; sleep 1
  local LOG=/tmp/turn_probe_${label// /_}.log; : > "$LOG"
  tmux new-session -d -s tp -x 200 -y 50 "export PATH=\$HOME/.local/bin:\$PATH MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12; cd \$HOME/microduck_rl; uv run scripts/infer_policy.py $FLAG $ONNX --new-cmd-obs --debug $*; sleep 300"
  tmux pipe-pane -o -t tp "cat >> $LOG"
  for i in $(seq 1 240); do grep -q "Keyboard controls" "$LOG" 2>/dev/null && break; sleep 0.5; done
  sleep 3
  local n0; n0=$(wc -l < "$LOG")
  sleep "$WIN"
  tmux kill-session -t tp 2>/dev/null; pkill -f infer_policy.py 2>/dev/null
  tail -n +$((n0+1)) "$LOG" > "${LOG}.win"
  python3 - "$label" "${LOG}.win" "$OUT" <<'EOF'
import re, sys, statistics
label, path, out = sys.argv[1:4]
txt = open(path, errors="ignore").read()
wz = [float(m.group(1).split()[2]) for m in re.finditer(r"Ang vel \[0:3\]:\s*\[([^\]]+)\]", txt) if len(m.group(1).split()) == 3]
fwd = [float(m.group(1)) for m in re.finditer(r"fwd=([-+0-9.]+)/", txt)]
z = [float(m.group(1)) for m in re.finditer(r"trunk_z=([0-9.]+)", txt)]
half = len(wz) // 2
wz_l = wz[half:] if half else wz
fwd_l = fwd[len(fwd)//2:] if fwd else []
line = (f"{label:<12} yaw_rate={statistics.mean(wz_l):+.2f} rad/s (n={len(wz_l)})  "
        f"fwd={statistics.mean(fwd_l):+.2f} m/s  min_trunk_z={min(z) if z else float('nan'):.0f} mm")
print(line); open(out, "a").write(line + "\n")
EOF
}
CASES="${CASES:-turn0.3 turn0.6 turn1.0 turn-1.0 walk0.30 walk0.45}"   # env CASES="walk0.30 turn0.6" = subset
for c in $CASES; do
  case "$c" in
    turn0.3)  run_case "turn +0.3" --ang-vel-z 0.3 ;;
    turn0.6)  run_case "turn +0.6" --ang-vel-z 0.6 ;;
    turn1.0)  run_case "turn +1.0" --ang-vel-z 1.0 ;;
    turn-1.0) run_case "turn -1.0" --ang-vel-z -1.0 ;;
    walk0.30) run_case "walk 0.30" --lin-vel-x 0.30 ;;
    walk0.45) run_case "walk 0.45" --lin-vel-x 0.45 ;;
  esac
done
echo "===== $ONNX"; cat "$OUT"
