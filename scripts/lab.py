# -*- coding: utf-8 -*-
"""The experiment loop as one tool: an experiment is a small TOML file, the tool does the rest.

    train -> exam every checkpoint (sweep) -> pick -> full exam (3 seeds) -> export ONNX -> release gate -> ledger

usage (inside WSL; any working directory):
    python3 scripts/lab.py run experiments/<name>.toml [more.toml ...]   # run one or several, in order
    python3 scripts/lab.py run --smoke experiments/<name>.toml           # 3 iterations, 64 envs: checks the plumbing
    python3 scripts/lab.py run --dry-run experiments/<name>.toml         # print the commands, run nothing
    python3 scripts/lab.py board                                          # leaderboard of everything in the ledger

Every step writes its result under logs/lab/<name>/ inside microduck_rl and is skipped when that result exists, so
an interrupted run continues where it stopped (training itself is not resumed: delete the half run and start again).
One line per finished experiment goes to logs/lab/ledger.jsonl. Spec format: see experiments/README.md."""
import argparse
import contextlib
import datetime as dt
import glob
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.expanduser(os.environ.get("MICRODUCK_RL", "~/microduck_rl"))
sys.path.insert(0, HERE)
import release_gate  # noqa: E402

DEFAULTS = {
    "num_envs": 4096,
    "start": {"mode": "scratch"},
    "env": {},
    "sweep": {"from": 0, "every": 1, "trials": "tracking,push", "seeds": "0", "num_envs": 256, "pick": "release"},
    "exam": {"trials": "getup,push,tracking,body,goal", "seeds": "0,1,2", "num_envs": 256},
    "gate": {"compare": []},
}


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_spec(path):
    with open(path, "rb") as f:
        spec = tomllib.load(f)
    base = {}
    for inc in spec.pop("include", []):
        base = merge(base, load_spec(os.path.join(os.path.dirname(os.path.abspath(path)), inc)))
    return merge(merge(DEFAULTS, base), spec)


class Lab:
    def __init__(self, spec, smoke=False, dry=False):
        self.s, self.smoke, self.dry = spec, smoke, dry
        self.name = spec["name"] + ("_smoke" if smoke else "")
        self.dir = os.path.join(RL, "logs", "lab", self.name)
        self.state_path = os.path.join(self.dir, "state.json")
        os.makedirs(self.dir, exist_ok=True)
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.env = dict(os.environ, WANDB_MODE="offline", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                        PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", ""))
        self.env.update({k: str(v) for k, v in spec["env"].items()})
        # the exam wrapper takes the teachers from EXPERT / ANCHOR
        if "MICRODUCK_STAND_EXPERT" in self.env:
            self.env["EXPERT"] = self.env["MICRODUCK_STAND_EXPERT"]
        if "MICRODUCK_WALK_ANCHOR" in self.env:
            self.env["ANCHOR"] = self.env["MICRODUCK_WALK_ANCHOR"]

    # ---------- helpers ----------
    def log(self, msg):
        line = f"=== {dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        if not self.dry:
            with open(os.path.join(self.dir, "lab.log"), "a") as f:
                f.write(line + "\n")

    def save(self):
        if not self.dry:
            json.dump(self.state, open(self.state_path, "w"), indent=1)

    def sh(self, cmd, out=None, extra_env=None):
        env = dict(self.env, **(extra_env or {}))
        shown = " ".join(f"{k}={v}" for k, v in (extra_env or {}).items()) + (" " if extra_env else "") + " ".join(shlex.quote(c) for c in cmd)
        if self.dry:
            print("  $", shown + (f" > {out}" if out else ""))
            return 0
        with open(out, "w") if out else open(os.devnull, "w") as f:
            return subprocess.call(cmd, cwd=RL, env=env, stdout=f, stderr=subprocess.STDOUT)

    def exam(self, ckpt, trials, seeds, n, out, label):
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return
        wrapper = os.path.join(HERE, "eval_battery.sh")
        if not os.path.exists(wrapper):
            wrapper = os.path.join(HERE, "wsl_eval_battery.sh")
        self.sh(["bash", wrapper, self.s["eval_task"], ckpt, str(n), trials, out, seeds, label],
                out=out.replace(".json", ".log"))

    @staticmethod
    def run_dirs():
        return set(glob.glob(os.path.join(RL, "logs", "rsl_rl", "*", "*/")))

    # ---------- steps ----------
    def train(self):
        if self.state.get("run_dir"):
            self.log(f"train: done before ({self.state['run_dir']})")
            return
        s = self.s
        iters = 3 if self.smoke else s["iterations"]
        n = 64 if self.smoke else s["num_envs"]
        cmd = ["uv", "run", "train", s["task"], "--env.scene.num-envs", str(n), "--agent.max-iterations", str(iters)]
        extra = {}
        mode = s["start"]["mode"]
        if mode in ("resume", "warm"):
            ck = s["start"]["from"]
            cmd += ["--agent.resume", "True", "--agent.load-run", os.path.basename(os.path.dirname(ck)),
                    "--agent.load-checkpoint", os.path.basename(ck)]
            if mode == "warm":
                extra["MICRODUCK_WARM_START"] = "1"   # weights only; curricula and the iteration counter start fresh
        elif mode != "scratch":
            raise SystemExit(f"start.mode must be scratch, warm or resume, not {mode!r}")
        self.log(f"train: {s['task']}, {mode}{' from ' + s['start'].get('from', '') if mode != 'scratch' else ''}, "
                 f"{iters} iterations, {n} envs, env {s['env']}")
        before = self.run_dirs()
        rc = self.sh(cmd, out=os.path.join(self.dir, "train.log"), extra_env=extra)
        if self.dry:
            return
        new = sorted(self.run_dirs() - before, key=os.path.getmtime)
        if rc != 0 or not new:
            raise SystemExit(f"training failed (exit {rc}); see {self.dir}/train.log")
        self.state["run_dir"] = os.path.relpath(new[-1].rstrip("/"), RL)
        self.save()
        self.log(f"train: done, run {self.state['run_dir']}")

    def checkpoints(self):
        rd = os.path.join(RL, self.state.get("run_dir", "RUN_DIR"))
        its = sorted(int(m.group(1)) for f in glob.glob(os.path.join(rd, "model_*.pt"))
                     if (m := re.search(r"model_(\d+)\.pt$", f)))
        return rd, its

    def sweep(self):
        sw = self.s["sweep"]
        rd, its = self.checkpoints()
        its = [i for i in its if i >= sw["from"]] if not self.smoke else its[-1:]
        its = its[::max(1, sw["every"])] if len(its) > 1 else its
        sdir = os.path.join(self.dir, "sweep")
        os.makedirs(sdir, exist_ok=True)
        self.log(f"sweep: {len(its)} checkpoints ({sw['trials']}, seeds {sw['seeds']})")
        for it in its:
            self.exam(os.path.join(rd, f"model_{it}.pt"), sw["trials"], sw["seeds"], 16 if self.smoke else sw["num_envs"],
                      os.path.join(sdir, f"{self.name}_{it}.json"), f"{self.name}_{it}")
        if self.dry:
            self.state["pick"] = "PICK"
            return
        out = subprocess.run([sys.executable, os.path.join(HERE, "pick_from_battery.py"), sdir, sw["pick"]],
                             capture_output=True, text=True).stdout
        with open(os.path.join(self.dir, "lab.log"), "a") as f:
            f.write(out)
        print(out, end="")
        m = re.search(r"^PICK (\d+)", out, re.M)
        if not m:
            raise SystemExit("pick failed: no exam results in the sweep (see the .log files next to them)")
        self.state["pick"] = int(m.group(1))
        self.save()
        self.log(f"pick: model_{self.state['pick']}.pt")

    def full_exam(self):
        ex = self.s["exam"]
        rd, _ = self.checkpoints()
        ck = os.path.join(rd, f"model_{self.state['pick']}.pt")
        out = os.path.join(self.dir, "exam.json")
        self.log(f"exam: {ex['trials']}, seeds {ex['seeds']}")
        self.exam(ck, self.s["sweep"]["trials"] if self.smoke else ex["trials"], "0" if self.smoke else ex["seeds"],
                  16 if self.smoke else ex["num_envs"], out, self.name)
        if not self.dry:
            if not os.path.exists(out):
                raise SystemExit(f"exam failed; see {out.replace('.json', '.log')}")
            shutil.copy(out, os.path.join(RL, "logs", "eval", f"{self.name}.json"))

    def export(self):
        rd, _ = self.checkpoints()
        onnx = os.path.join("exports", f"{self.name}_{self.state['pick']}.onnx")
        if os.path.exists(os.path.join(RL, onnx)):
            return
        rc = self.sh(["uv", "run", "scripts/export.py", self.s["task"], "--checkpoint-file",
                      os.path.relpath(os.path.join(rd, f"model_{self.state['pick']}.pt"), RL), "--onnx-file", onnx],
                     out=os.path.join(self.dir, "export.log"))
        self.log(f"export: {onnx}" if rc == 0 else f"export failed (see {self.dir}/export.log)")
        self.state["onnx"] = onnx if rc == 0 else None
        self.save()

    def gate(self):
        bars = self.s["gate"].get("bars")
        check = (lambda p: check_bars(p, bars)) if bars else release_gate.check
        if self.dry:
            print("  $ gate", "bars " + json.dumps(bars) if bars else "release_gate.py", "exam.json", *self.s["gate"]["compare"])
            return
        out = os.path.join(self.dir, "exam.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            verdict = check(out)
            for other in self.s["gate"]["compare"]:
                p = os.path.join(RL, other)
                if os.path.exists(p):
                    check(p)
        print(buf.getvalue(), end="")
        with open(os.path.join(self.dir, "lab.log"), "a") as f:
            f.write(buf.getvalue())
        self.state["gate"] = "PASS" if verdict else "FAIL"
        self.save()
        self.log(f"gate: {self.state['gate']}")

    def ledger(self):
        if self.dry or self.smoke:
            return
        d = json.load(open(os.path.join(self.dir, "exam.json")))
        a = d["aggregate"]
        cells = release_gate.push_cells(d)
        row = {"name": self.name, "finished": dt.datetime.now().isoformat(timespec="seconds"),
               "task": self.s["task"], "start": self.s["start"], "iterations": self.s["iterations"],
               "env": self.s["env"], "run_dir": self.state["run_dir"], "pick": self.state["pick"],
               "onnx": self.state.get("onnx"), "gate": self.state.get("gate"),
               "push_cells": {f"{m} {v}": round(x, 4) for (m, v), x in cells.items()},
               "metrics": {k: round(v["mean"], 4) for k, v in a.items() if isinstance(v, dict) and "mean" in v}}
        path = os.path.join(RL, "logs", "lab", "ledger.jsonl")
        rows = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
        rows = [r for r in rows if r["name"] != self.name] + [row]
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        self.log(f"ledger: {path}")

    def run(self):
        self.log(f"LAB START {self.name}" + (" (smoke)" if self.smoke else "") + (" (dry run)" if self.dry else ""))
        self.train()
        if "pick" not in self.state or self.dry:
            self.sweep()
        else:
            self.log(f"sweep: done before (pick model_{self.state['pick']}.pt)")
        self.full_exam()
        self.export()
        self.gate()
        self.ledger()
        self.log(f"PIPELINE DONE (lab {self.name})")


def check_bars(path, bars):
    """A gate given in the spec: {metric: "<= -14" | ">= 0.8"}. Metrics are the exam's aggregate means, plus
    push_worst (the worst of the four push-recovery cells). Prints one ok / MISS line per bar."""
    d = json.load(open(path))
    vals = {k: v["mean"] for k, v in d["aggregate"].items() if isinstance(v, dict) and "mean" in v}
    cells = release_gate.push_cells(d)
    if cells:
        vals["push_worst"] = min(cells.values())
    ok_all = True
    lines = []
    for metric, rule in bars.items():
        op, bar = rule.split()
        v = vals.get(metric, float("nan"))
        ok = (v <= float(bar)) if op == "<=" else (v >= float(bar))
        ok_all &= ok
        lines.append(f"  {'ok  ' if ok else 'MISS'} {metric:<26} {v:>9.3f}   (bar {op} {bar})")
    print(f"\n{d['name']}  ({path})  -> {'PASS' if ok_all else 'FAIL'}")
    print("\n".join(lines))
    return ok_all


def board():
    path = os.path.join(RL, "logs", "lab", "ledger.jsonl")
    if not os.path.exists(path):
        print("ledger is empty")
        return
    rows = [json.loads(line) for line in open(path)]
    print("| experiment | push worst / mean | get-up | speed at 0.30 | turn at 0.6 | crouch −30 mm | pushed goal | gate |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: min(r["push_cells"].values() or [0]), reverse=True):
        c = list(r["push_cells"].values())
        m = r["metrics"]
        print(f"| {r['name']} | {100*min(c):.1f} / {100*sum(c)/len(c):.1f} % | {100*m.get('getup_recovered', float('nan')):.1f} % "
              f"| {m.get('speed_at_0.30', float('nan')):.2f} | {m.get('turn_at_0.6', float('nan')):.2f} "
              f"| {m.get('crouch_mm_for_-30', float('nan')):.1f} | {100*m.get('goal_reached_push', float('nan')):.0f} % | {r['gate']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("specs", nargs="+")
    r.add_argument("--smoke", action="store_true", help="3 iterations, 64 envs, tiny exams: checks the plumbing")
    r.add_argument("--dry-run", action="store_true", help="print the commands only")
    sub.add_parser("board")
    a = ap.parse_args()
    if a.cmd == "board":
        board()
        return
    for p in a.specs:
        Lab(load_spec(p), smoke=a.smoke, dry=a.dry_run).run()


if __name__ == "__main__":
    main()
