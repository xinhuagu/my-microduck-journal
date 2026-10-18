# -*- coding: utf-8 -*-
"""Leaderboard for the background search: one row per variant JSON (1 seed) in a folder.
usage: search_board.py <dir>   -> prints the table and a final 'BEST <name> <preview PASS|FAIL>' line
Preview gate (single seed): min push cell >= 0.95, get-up >= 0.98, speed@0.30 >= 0.28, turn@0.6 >= 0.55."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_gate  # noqa: E402

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f)); a = d["aggregate"]
    g = lambda k: a.get(k, {}).get("mean", float("nan"))
    cells = release_gate.push_cells(d)
    mn = min(cells.values()) if cells else float("nan"); mean = sum(cells.values()) / max(len(cells), 1)
    ok = mn >= 0.95 and g("getup_recovered") >= 0.98 and g("speed_at_0.30") >= 0.28 and g("turn_at_0.6") >= 0.55
    rows.append((d["name"], mn, mean, g("getup_recovered"), g("speed_at_0.30"), g("turn_at_0.6"), ok, d.get("checkpoint", "")))
rows.sort(key=lambda r: (r[1], r[4]), reverse=True)
print("| variant | push min / mean | get-up | speed@0.30 | turn@0.6 | preview gate |\n|---|---|---|---|---|---|")
for n, mn, mean, gu, sp, t6, ok, ck in rows:
    print(f"| {n} | {100*mn:.1f} / {100*mean:.1f} % | {100*gu:.1f} % | {sp:.2f} | {t6:.2f} | {'PASS' if ok else 'fail'} |")
if rows:
    b = rows[0]
    print(f"BEST {b[0]} {'PASS' if b[6] else 'FAIL'} {b[7]}")
