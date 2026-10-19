# Experiments

One TOML file per experiment, run by [`scripts/lab.py`](../scripts/lab.py):

```bash
# inside WSL, with upstream microduck_rl in ~/microduck_rl (or MICRODUCK_RL=/path) and the patch applied
python3 scripts/lab.py run --smoke experiments/s5_anchor_075_expert_x5.toml   # 3 iterations: checks the plumbing (~5 min)
python3 scripts/lab.py run experiments/s5_anchor_075_expert_x5.toml           # the real thing (~2 h for 1 000 iterations)
python3 scripts/lab.py run a.toml b.toml c.toml                                # a queue, one after the other
python3 scripts/lab.py board                                                   # leaderboard of every finished experiment
```

What one run does, each step skipped if its result already exists (so an interrupted run continues):

1. **train** `task` for `iterations`, from scratch, warm-started or resumed (`[start]`), with the `[env]` knobs;
2. **sweep**: examine every saved checkpoint from `sweep.from` on (1 seed, quick trials);
3. **pick** the best by `sweep.pick` ([`pick_from_battery.py`](../scripts/pick_from_battery.py));
4. **exam** the pick with the full battery, 3 seeds;
5. **export** it to ONNX;
6. **gate**: the release gate ([`release_gate.py`](../scripts/release_gate.py)), next to the experiments in `gate.compare`;
7. **ledger**: one line in `logs/lab/ledger.jsonl` — settings, run directory, pick, every metric, the verdict.

Everything lands in `logs/lab/<name>/` inside microduck_rl: `lab.log`, `train.log`, `sweep/`, `exam.json`, `state.json`.

## Spec

```toml
include = ["_velstand_fast_base.toml"]   # optional: shared settings, merged first (tables merge key by key)
name = "S5_anchor_075_expert_x5"         # unique; names the folder, the ONNX and the ledger row
task = "Mjlab-VelStand-Rough-Backlash-MicroDuck"   # training task
eval_task = "Mjlab-VelStand-Flat-MicroDuck"        # task the exam runs in
iterations = 1000
num_envs = 4096                          # default 4096; half that on a 16 GB card

[start]
mode = "resume"                          # scratch | warm (weights only, curricula restart) | resume (curricula continue)
from = "logs/rsl_rl/velstand/<run>/model_2250.pt"  # for warm / resume

[env]                                    # MICRODUCK_* knobs from patches/all-in-one.patch, all off by default
MICRODUCK_ANCHOR_COEF = "0.75"

[sweep]
from = 2500                              # first checkpoint to examine (default 0)
every = 1                                # every n-th saved checkpoint
trials = "tracking,push"                 # quick trials for the sweep
seeds = "0"
pick = "release"                         # walk | velstand | release (see pick_from_battery.py)

[exam]
trials = "getup,push,tracking,body,goal"
seeds = "0,1,2"

[gate]
compare = ["logs/eval/<other>.json"]     # printed next to this experiment's verdict
```

## Files here

| file | what |
|---|---|
| `_velstand_fast_base.toml` | the fast walk + get-up setup shared by candidate C and the search (teachers, patches, sweep and exam) |
| `s5_anchor_075_expert_x5.toml` | S5 written as a spec: produces the same training command as `scripts/pipeline_round8_anchor_075.sh` |
| `_velstand_default_base.toml` | the default policy's setup (v4: walk teacher v1, stand-up teacher v3, body-wide) |
| `bodywalk_v1.toml` | the second skill: crouch / tilt while walking, with its own gate in `[gate.bars]` |
| `sideroll_v1.toml` | the side roll (new skill, released in v0.3): task, sweep, exam and its gate, with the two-run budget in the header |

A gate can be the release gate (default) or a list of bars in the spec — any metric of the exam, plus
`push_worst` (the worst of the four push-recovery cells):

```toml
[gate.bars]
walk_speed_ratio_body = ">= 0.8"
push_worst = ">= 0.94"
```
