# -*- coding: utf-8 -*-
"""Pick a checkpoint from a folder of eval_battery JSONs (one per checkpoint, named <label>_<iter>.json).
usage: pick_from_battery.py <dir> walk|velstand
  walk:     score = 3*speed@0.30 + turn@0.3 + 0.5*turn@0.6 - 0.5*drift_p90@0.30
  velstand: score = push_rec@1.0 + push_rec@1.2 + 4*speed@0.30 + 0.5*turn@0.6 - 0.5*drift_p90@0.30
Prints one line per checkpoint and a final 'PICK <iter> ...' line with the gate verdict."""
import glob
import json
import os
import re
import sys

d, kind = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(os.path.join(d, "*.json")):
    m = re.search(r"_(\d+)\.json$", f)
    if not m:
        continue
    a = json.load(open(f))["aggregate"]
    g = lambda k: a.get(k, {}).get("mean", 0.0)
    if kind == "walk":
        sc = 3 * g("speed_at_0.30") + g("turn_at_0.3") + 0.5 * g("turn_at_0.6") - 0.5 * g("drift_p90_at_0.30")
    elif kind == "release":   # v0.2 candidates: recovery first, speed and turning capped at the bar
        sc = 2 * (g("push_recovered_1.0") + g("push_recovered_1.2")) + 2 * min(g("speed_at_0.30"), 0.30) \
             + 0.5 * min(g("turn_at_0.6"), 0.60) - 0.5 * g("drift_p90_at_0.30")
    else:
        sc = g("push_recovered_1.0") + g("push_recovered_1.2") + 4 * g("speed_at_0.30") + 0.5 * g("turn_at_0.6") - 0.5 * g("drift_p90_at_0.30")
    rows.append((int(m.group(1)), sc, g("speed_at_0.30"), g("turn_at_0.3"), g("turn_at_0.6"), g("drift_p90_at_0.30"),
                 g("push_recovered_1.0"), g("push_recovered_1.2")))
rows.sort()
for it, sc, sp, t3, t6, dr, p1, p2 in rows:
    print(f"  {it:5d}  score {sc:+.3f}  speed@0.30 {sp:.2f}  turn 0.3/0.6 {t3:.2f}/{t6:.2f}  drift p90 {dr:.2f}  push rec 1.0/1.2 {p1:.2f}/{p2:.2f}")
if not rows:
    print("PICK none"); sys.exit(1)
best = max(rows, key=lambda r: r[1])
it, sc, sp, t3, t6, dr, p1, p2 = best
if kind == "walk":
    ok = sp >= 0.20 and t6 >= 0.50
    gate = f"speed@0.30 {sp:.2f} >= 0.20 and turn@0.6 {t6:.2f} >= 0.50"
elif kind == "release":
    ok = sp >= 0.28 and p1 >= 0.95 and p2 >= 0.95
    gate = f"single-seed preview: speed@0.30 {sp:.2f} >= 0.28, push rec 1.0/1.2 {p1:.2f}/{p2:.2f} >= 0.95 (the 3-seed release gate decides)"
else:
    ok = sp >= 0.17 and p2 >= 0.90
    gate = f"speed@0.30 {sp:.2f} >= 0.17 and push rec@1.2 {p2:.2f} >= 0.90"
print(f"PICK {it} {'PASS' if ok else 'FAIL'} ({gate})")
