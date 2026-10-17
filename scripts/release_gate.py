# -*- coding: utf-8 -*-
"""v0.2 release gate: check 3-seed battery JSONs against the agreed bar and print a table.
usage: release_gate.py <json> [<json> ...]
Bar (agreed 2026-10-17): push recovery >= 95 % in each of standing/walking x 1.0/1.2 m/s; get-up >= 98 %;
speed@0.30 >= 0.28 m/s; turn@0.6 >= 0.55 rad/s; crouch for -30 mm <= -27 mm; goal reached after a push >= 96 %."""
import json
import sys


def push_cells(d):
    acc = {}
    for ps in d["per_seed"]:
        for r in ps.get("push", []):
            acc.setdefault((r["mode"], r["push_mps"]), []).append(r["recovered_of_fallen"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def check(path):
    d = json.load(open(path)); a = d["aggregate"]
    m = lambda k: a.get(k, {}).get("mean", float("nan"))
    rows = []
    for (mode, spd), v in sorted(push_cells(d).items()):
        rows.append((f"push {spd:.1f} m/s {mode}: recovered", v, v >= 0.95, ">= 95 %", f"{100*v:.1f} %"))
    rows += [
        ("get-up recovered", m("getup_recovered"), m("getup_recovered") >= 0.98, ">= 98 %", f"{100*m('getup_recovered'):.1f} %"),
        ("speed at 0.30 m/s", m("speed_at_0.30"), m("speed_at_0.30") >= 0.28, ">= 0.28", f"{m('speed_at_0.30'):.2f} m/s"),
        ("turn at 0.6 rad/s", m("turn_at_0.6"), m("turn_at_0.6") >= 0.55, ">= 0.55", f"{m('turn_at_0.6'):.2f} rad/s"),
        ("crouch for -30 mm", m("crouch_mm_for_-30"), m("crouch_mm_for_-30") <= -27, "<= -27 mm", f"{m('crouch_mm_for_-30'):.1f} mm"),
        ("goal reached after a push", m("goal_reached_push"), m("goal_reached_push") >= 0.96, ">= 96 %", f"{100*m('goal_reached_push'):.1f} %"),
    ]
    ok = all(r[2] for r in rows)
    print(f"\n{d['name']}  ({path})  -> {'PASS' if ok else 'FAIL'}")
    for name, _, passed, bar, shown in rows:
        print(f"  {'ok  ' if passed else 'MISS'} {name:<34} {shown:>12}   (bar {bar})")
    return ok


if __name__ == "__main__":
    results = [check(p) for p in sys.argv[1:]]
    print("\nRELEASE GATE:", "at least one candidate passes" if any(results) else "no candidate passes")
