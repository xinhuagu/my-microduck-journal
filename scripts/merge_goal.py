# -*- coding: utf-8 -*-
"""Merge goal-trial JSONs (logs/eval_goal/<name>.json) into the benchmark JSONs (logs/eval/<name>.json):
per-seed 'goal' rows and summary keys are added, the aggregate recomputed. usage: merge_goal.py <eval dir> <goal dir>"""
import glob
import json
import os
import sys

eval_dir, goal_dir = sys.argv[1], sys.argv[2]
for gf in glob.glob(os.path.join(goal_dir, "*.json")):
    name = os.path.basename(gf)
    ef = os.path.join(eval_dir, name)
    if not os.path.exists(ef):
        print("no benchmark json for", name); continue
    g = json.load(open(gf)); e = json.load(open(ef))
    by_seed = {r["seed"]: r for r in g["per_seed"]}
    for r in e["per_seed"]:
        gr = by_seed.get(r["seed"])
        if not gr:
            continue
        r["goal"] = gr["goal"]
        r["summary"].update({k: v for k, v in gr["summary"].items() if k.startswith("goal_")})
    keys = sorted({k for r in e["per_seed"] for k in r["summary"]})
    agg = {}
    for k in keys:
        v = [r["summary"][k] for r in e["per_seed"] if k in r["summary"] and r["summary"][k] == r["summary"][k]]
        if not v:
            continue
        m = sum(v) / len(v); sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        agg[k] = {"mean": m, "std": sd, "n": len(v)}
    e["aggregate"] = agg
    json.dump(e, open(ef, "w"), indent=1)
    print("merged", name, {k: round(v["mean"], 3) for k, v in agg.items() if k.startswith("goal_reached")})
