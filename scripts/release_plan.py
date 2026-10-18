# -*- coding: utf-8 -*-
"""Decide the v0.2 release contents by rule and write the release notes from the battery JSONs.

usage: release_plan.py <eval dir> <out dir>
Rule (agreed 2026-10-18):
  default = cand_C_v02 if it passes the release gate, else ours_velstand_v4
  fast    = among the fast candidates (speed@0.30 >= 0.28) the one with the best mean push recovery
            (tie-break: get-up); omitted if it is the default
Writes <out>/plan.json (names, onnx files, checkpoints) and <out>/RELEASE_NOTES.md (without checksums; the shell
step appends them)."""
import json
import os
import sys

E, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_gate  # noqa: E402

ONNX = {
    "ours_velstand_v4": "exports/velstand_v4_2999.onnx",
    "ours_velstand_v6": "exports/velstand_v6_2250.onnx",
    "cand_A_v02": "exports/cand_A_v02_2000.onnx",
    "cand_B_v02": "exports/cand_B_v02_4249.onnx",
}


def load(n):
    p = os.path.join(E, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def onnx_for(n, d):
    if n in ONNX:
        return ONNX[n]
    it = os.path.basename(d["checkpoint"]).replace("model_", "").replace(".pt", "")
    return f"exports/{n}_{it}.onnx"


def push_mean(d):
    cells = release_gate.push_cells(d)
    return sum(cells.values()) / max(len(cells), 1), cells


def m(d, k):
    return d["aggregate"].get(k, {}).get("mean", float("nan"))


C = load("cand_C_v02")
c_pass = bool(C) and release_gate.check(os.path.join(E, "cand_C_v02.json"))
default = "cand_C_v02" if c_pass else "ours_velstand_v4"
fast_pool = [n for n in ("cand_C_v02", "cand_B_v02", "ours_velstand_v6", "cand_A_v02") if load(n) and m(load(n), "speed_at_0.30") >= 0.28]
fast_pool.sort(key=lambda n: (push_mean(load(n))[0], m(load(n), "getup_recovered")), reverse=True)
fast = next((n for n in fast_pool if n != default), None)

plan = {"default": default, "default_passed_gate": c_pass, "fast": fast,
        "files": {default: onnx_for(default, load(default))}}
if fast:
    plan["files"][fast] = onnx_for(fast, load(fast))
json.dump(plan, open(os.path.join(OUT, "plan.json"), "w"), indent=1)

LABEL = {"ours_velstand_v4": "v4", "ours_velstand_v6": "v6", "cand_A_v02": "candidate A", "cand_B_v02": "candidate B", "cand_C_v02": "candidate C"}
REL_NAME = {default: "velstand_default.onnx"}
if fast:
    REL_NAME[fast] = "velstand_fast.onnx"


def row(name, d):
    pm, cells = push_mean(d)
    st = [cells.get(("standing", 1.0)), cells.get(("standing", 1.2))]
    wk = [cells.get(("walking", 1.0)), cells.get(("walking", 1.2))]
    f = lambda v: f"{100*v:.0f} %" if v == v else "—"
    return (f"| {name} | {f(m(d,'getup_recovered'))} | {f(st[0])} / {f(st[1])} | {f(wk[0])} / {f(wk[1])} | "
            f"{m(d,'speed_at_0.30'):.2f} / {m(d,'speed_at_0.45'):.2f} | {m(d,'turn_at_0.6'):.2f} / {m(d,'turn_at_1.0'):.2f} | "
            f"{m(d,'crouch_mm_for_-30'):.1f} mm | {f(m(d,'goal_reached_push'))}, {m(d,'goal_median_s_nopush'):.1f} s |")


hdr = ("| policy | get-up | push recovered, standing 1.0 / 1.2 m/s | walking 1.0 / 1.2 m/s | speed at 0.30 / 0.45 m/s | "
       "turn at 0.6 / 1.0 rad/s | crouch for −30 mm | 1.5 m goal after a push, time unpushed |\n|---|---|---|---|---|---|---|---|")
rows = []
off = load("official_velstand_v7")
if off:
    rows.append(row("official `velstand.onnx` (v7)", off))
rows.append(row(f"**`velstand_default.onnx`** = {LABEL[default]}", load(default)))
if fast:
    rows.append(row(f"**`velstand_fast.onnx`** = {LABEL[fast]}", load(fast)))
if default != "ours_velstand_v4":
    rows.append(row("v4 (release v0.1)", load("ours_velstand_v4")))

if c_pass:
    headline = ("**`velstand_default.onnx` passes the release gate**: it is at least as good as v4 on every push-recovery "
                "cell, get-up, crouch and the pushed goal, and it walks at the commanded speed.")
else:
    headline = ("No candidate passed the release gate (push recovery ≥ 95 % in every cell, get-up ≥ 98 %, speed ≥ 0.28 m/s, "
                "turn ≥ 0.55 rad/s, crouch ≤ −27 mm, pushed goal ≥ 96 %), so **the default stays v4**. The fast variant is "
                "published next to it with its trade-off stated: it walks at the commanded speed and turns well, and recovers "
                "fewer push falls than v4.")

notes = f"""## v0.2 — walking speed, plus a fast variant

{headline}

### Files

| file | what it is |
|---|---|
| `velstand_default.onnx` | the policy to use: walking, get-up from any fall, body-pose commands (crouch / tilt) — {LABEL[default]} (`{load(default)['checkpoint']}`) |
"""
if fast:
    notes += f"| `velstand_fast.onnx` | the fast variant — {LABEL[fast]} (`{load(fast)['checkpoint']}`): walks at the commanded speed; recovers fewer push falls than the default |\n"
notes += """| `walk_teacher_v3.onnx` | the walk teacher that tracks speed (0.15 / 0.30 / 0.45 m/s → 0.16 / 0.31 / 0.44): the anchor the fast policies are distilled from |
| `standup_teacher_v3.onnx` | the stand-up teacher (get-up in 0.3–0.5 s, crouch −40 mm, tilt ±20°) |
| `eval_*.json` | raw output of `scripts/eval_battery.py`, 256 envs × 3 seeds |
| `SHA256SUMS.txt` | checksums |

### Measured with the benchmark (same environment, 256 randomised ducks, 3 seeds)

""" + hdr + "\n" + "\n".join(rows) + """

### What changed since v0.1

- **Walking speed.** Every walk policy, the official one included, walked at about 60 % of the command. A fine
  forward-speed reward next to the stock one (`patches/speed.patch`) gives a walk teacher that tracks the
  command; distilled as the anchor, it carries the speed into the walk + get-up policy.
- **What it costs, and why.** With the fast walk teacher as the anchor the policy recovers fewer push falls. The
  battery shows why: the fast policies do not fall more often, but more of them stay lying on the side or back
  (trunk at 64 mm, tilted ~97°) — the get-up from those states erodes while the network learns the faster gait.
  Stronger imitation of the stand-up teacher on fallen frames and more episodes that start on the ground
  recover part of it. Details: `docs/training-log.md`.
- **Release gate.** A policy becomes the default only if it is at least as good as the previous default on every
  push-recovery cell, get-up, crouch and the pushed goal (`scripts/release_gate.py`).

### Use

```bash
uv run scripts/infer_policy.py --walking velstand_default.onnx --new-cmd-obs
# or the fast one:
uv run scripts/infer_policy.py --walking velstand_fast.onnx --new-cmd-obs
```

Upstream `microduck_rl` at `ceca0170` with the patches in `patches/`; every run, command and number is in
`docs/training-log.md`.
"""
open(os.path.join(OUT, "RELEASE_NOTES.md"), "w", encoding="utf-8").write(notes)
print(json.dumps(plan, indent=1))
