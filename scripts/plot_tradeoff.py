# -*- coding: utf-8 -*-
"""Speed vs push recovery for every walk + get-up policy measured with 3 seeds; colour = turning.

usage (from the journal root, any Python with matplotlib):
    python scripts/plot_tradeoff.py [--out results/tradeoff.png]
Each point is one policy: x = speed achieved for a 0.30 m/s command, y = the worst of the four push-recovery
cells of the release gate (standing / walking x 1.0 / 1.2 m/s) - the number the gate checks; the bar reaches up
to the best cell; colour = yaw rate achieved for a 0.6 rad/s turn command. The shaded box is the release gate's
target (speed >= 0.28 m/s, every push cell >= 95 %)."""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import release_gate  # noqa: E402

B = os.path.join(ROOT, "results", "benchmark")
POLICIES = [  # (label, json, kind)
    ("official v7", "official_velstand_v7.json", "official"),
    ("v2", "ours_velstand_v2.json", "ours"),
    ("v4 = default", "ours_velstand_v4.json", "release"),
    ("v5", "ours_velstand_v5_turning.json", "ours"),
    ("v6", "ours_velstand_v6.json", "ours"),
    ("A", "v02_candidates/cand_A_v02.json", "ours"),
    ("B", "v02_candidates/cand_B_v02.json", "ours"),
    ("C = fast", "ours_velstand_fast_v02.json", "release"),
    ("S2", "search_v03/full_S2_expert_x5.json", "ours"),
    ("S3", "search_v03/full_S3_anchor_half_expert_x5.json", "ours"),
    ("S5", "search_v03/full_S5_anchor_075_expert_x5.json", "ours"),
]
OFFSET = {"S5": (-24, 5), "S2": (8, -6), "C = fast": (-58, -4), "S3": (8, 2), "B": (8, -4), "v6": (8, -4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tradeoff.png"))
    a = ap.parse_args()
    pts = []
    for label, f, kind in POLICIES:
        p = os.path.join(B, f)
        if not os.path.exists(p):
            print(f"skip {label}: {f} missing")
            continue
        d = json.load(open(p))
        cells = list(release_gate.push_cells(d).values())
        agg = d["aggregate"]
        pts.append(dict(label=label, kind=kind, x=agg["speed_at_0.30"]["mean"],
                        y=100 * min(cells), mean=100 * sum(cells) / len(cells), hi=100 * max(cells),
                        turn=agg["turn_at_0.6"]["mean"]))

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=130)
    ax.add_patch(plt.Rectangle((0.28, 95), 0.06, 5, color="#2e7d32", alpha=0.10, lw=0))
    ax.text(0.307, 99.5, "release gate:\nspeed ≥ 0.28 and\nevery push cell ≥ 95 %", ha="center", va="top", fontsize=8, color="#2e7d32")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.75)
    cmap = plt.get_cmap("viridis")
    for p in pts:
        c = cmap(norm(p["turn"]))
        ax.vlines(p["x"], p["y"], p["hi"], color=c, lw=2, alpha=0.6)
        marker = {"official": "s", "release": "*", "ours": "o"}[p["kind"]]
        size = {"official": 80, "release": 260, "ours": 70}[p["kind"]]
        ax.scatter(p["x"], p["y"], s=size, marker=marker, color=c, edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(p["label"], (p["x"], p["y"]), xytext=OFFSET.get(p["label"], (7, 4)), textcoords="offset points", fontsize=9)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("turn achieved for a 0.6 rad/s command (rad/s)")
    ax.set_xlabel("speed achieved for a 0.30 m/s command (m/s)")
    ax.set_ylabel("push falls recovered, worst of 4 cells (%)\nbar = up to the best cell")
    ax.set_xlim(0.10, 0.335)
    ax.set_ylim(80, 100)
    ax.grid(alpha=0.3)
    ax.set_title("Walk + get-up policies: speed vs push recovery (256 ducks × 3 seeds each)", fontsize=11)
    ax.scatter([], [], marker="s", color="grey", edgecolor="black", label="official")
    ax.scatter([], [], marker="*", s=160, color="grey", edgecolor="black", label="released here")
    ax.scatter([], [], marker="o", color="grey", edgecolor="black", label="experiments here")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(a.out)
    for p in sorted(pts, key=lambda p: p["x"]):
        print(f"{p['label']:14s} speed {p['x']:.2f}  push worst {p['y']:.1f} (mean {p['mean']:.1f})  turn {p['turn']:.2f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
