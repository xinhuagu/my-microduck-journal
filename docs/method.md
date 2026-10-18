# Training your own skill with this loop

The policies in this repository came out of one loop, repeated: change one thing, train, examine every
checkpoint, pick by the exam, examine the pick properly, and let a rule fixed in advance decide whether it
is released. This page is that loop written down so it can be reused for another skill — with the
reason for every rule, because each one exists because something went wrong without it.

## The loop

```
  teacher(s) ──► gate the teacher ──► train / distil ──► sweep every checkpoint ──► pick
                                                                                    │
     release ◄── release gate (fixed in advance) ◄── full exam, 3 seeds ◄───────────┘
```

Steps 3–7 are one command: [`scripts/lab.py`](../scripts/lab.py) runs an experiment described in a small TOML
file ([`experiments/README.md`](../experiments/README.md)) — train, sweep, pick, exam, export, gate, ledger — and
continues where it stopped if interrupted.

## The rules, and where each came from

| rule | what went wrong without it |
|---|---|
| **Measure a behaviour before trying to change it.** | The first get-up policy "stood up" into a 34° lean; nobody saw it until a posture probe measured trunk pitch. Body commands: neither the policy nor the teacher crouched at all — found only by commanding −30 mm and measuring. |
| **Gate a teacher before anything learns from it.** | VelStand v5 was distilled from a walk teacher nobody had measured for speed; it inherited 0.12 m/s for a 0.30 command. Since then a teacher passes its own exam first (walk teacher v3: speed and turning, every checkpoint). |
| **Pick checkpoints by the exam, never "the last one".** | v6: push recovery 0.86 at iteration 2250, 0.80 at 2999, speed unchanged. The search control (S0): 1 000 more iterations with unchanged settings made recovery worse. |
| **One change per experiment.** | v6 changed the teacher and a reward together and lost push recovery; only candidate A (teacher alone) showed that the teacher was the cause. |
| **Fix the release gate before the run, and never move it after.** | Three candidates (C, S3, S5) missed by 0.2–3 points. Moving the bar each time would have made "passed the gate" mean nothing; the default stayed v4 and the near-misses were published as what they are. |
| **Record where failures end up, not just how many.** | "The fast policy is less stable" was wrong. Recording the final height and tilt of every failed recovery showed the ducks were lying on the side or back — a get-up problem, which the next candidate then fixed. |
| **Know what a logged number is before reading a trend into it.** | `expert_bc_*_frac` in the training log looked like imitation losses; they are shares of frames. |
| **When one knob trades two goals steeply, stop tuning it.** | Walk-teacher weight 0.5 / 0.75 / 1.0: push recovery or turning, never both. The next step is to separate the goals, not a finer grid. |
| **A published release never changes.** | Others hold its checksums. Improvements go into a new version that says what changed. |
| **Every change behind an environment variable, off by default, against one fixed upstream commit.** | Lets every past run be re-created from the same code, and anyone apply all changes with one patch. |

## Adding a new skill

What the loop needs from a new skill, in the order to do it:

1. **A task** that trains it — an upstream one, or a config change behind a new `MICRODUCK_*` variable in
   `microduck_rl` (keep it off by default; add it to the patch).
2. **A measurement first.** Before training, write the exam trial that measures the skill and run it on the
   existing policies (`eval_battery.py` has get-up, push, tracking, body-command and goal trials; a new skill
   usually needs one new trial function). That number is the baseline, and the bar for the gate.
3. **A teacher, if the skill goes into a combined network.** Train it alone first, gate it on the new trial,
   then distil it.
4. **A spec** in `experiments/`: start point, knobs, sweep, exam, and the gate the result has to pass.
5. **`lab.py run --smoke`**, then **`lab.py run`**. Read the sweep, not just the pick.
6. **Write it down** in `docs/training-log.md`: command, settings, the table, what it means, what is next —
   including when it failed.

## What it costs

On one consumer GPU (timings in the README): a 1 000-iteration resume of the walk + get-up network is about
70 minutes, the sweep over four checkpoints about 10, the 3-seed exam about 16 — about 1 h 40 min per
experiment, unattended. A queue of five runs overnight answers one question per run.
