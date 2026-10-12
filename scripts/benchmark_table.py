# -*- coding: utf-8 -*-
"""Build the benchmark table (markdown) from eval_battery.py JSON outputs.

usage: python benchmark_table.py <dir with *.json> [--out docs/benchmark.md]
Each JSON is one policy (possibly several seeds); cells show mean ± std across seeds.
"""
import argparse
import glob
import json
import os

COLS = [
    ("getup_recovered", "get-up", "pct"),
    ("getup_median_s", "get-up median", "s"),
    ("push_recovered_1.0", "push 1.0 m/s recovered", "pct"),
    ("push_recovered_1.2", "push 1.2 m/s recovered", "pct"),
    ("speed_at_0.30", "speed at 0.30", "mps"),
    ("drift_p90_at_0.30", "yaw drift p90", "rads"),
    ("turn_at_0.3", "turn 0.3", "rads"),
    ("turn_at_0.6", "turn 0.6", "rads"),
    ("turn_at_1.0", "turn 1.0", "rads"),
    ("crouch_mm_for_-30", "crouch −30 mm", "mm"),
    ("pitch_deg_for_10", "pitch ±10°", "deg"),
    ("pitch_deg_for_20", "pitch ±20°", "deg"),
    ("roll_deg_for_10", "roll ±10°", "deg"),
]


def cell(agg: dict, key: str, kind: str) -> str:
    v = agg.get(key)
    if not v:
        return "—"
    m, sd, n = v["mean"], v["std"], v["n"]
    if kind == "pct":
        return f"{100*m:.0f} %" + (f" ± {100*sd:.0f}" if n > 1 else "")
    if kind == "s":
        return f"{m:.2f} s" + (f" ± {sd:.2f}" if n > 1 else "")
    if kind == "mps":
        return f"{m:.2f} m/s" + (f" ± {sd:.2f}" if n > 1 else "")
    if kind == "rads":
        return f"{m:.2f}" + (f" ± {sd:.2f}" if n > 1 else "")
    if kind == "mm":
        return f"{m:.1f} mm" + (f" ± {sd:.1f}" if n > 1 else "")
    if kind == "deg":
        return f"{m:.1f}°" + (f" ± {sd:.1f}" if n > 1 else "")
    return f"{m:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--order", default=None, help="comma list of names to put first, in this order")
    a = ap.parse_args()
    runs = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        d = json.load(open(f))
        if "aggregate" not in d:
            continue
        runs.append(d)
    if a.order:
        order = [x.strip() for x in a.order.split(",")]
        runs.sort(key=lambda d: order.index(d["name"]) if d["name"] in order else len(order))
    lines = ["| policy | seeds | " + " | ".join(c[1] for c in COLS) + " |",
             "|---|---|" + "|".join("---" for _ in COLS) + "|"]
    for d in runs:
        lines.append(f"| {d['name']} | {len(d['seeds'])} | " + " | ".join(cell(d["aggregate"], k, kind) for k, _, kind in COLS) + " |")
    md = "\n".join(lines)
    print(md)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(md + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
