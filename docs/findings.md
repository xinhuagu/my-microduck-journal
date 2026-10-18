# Findings: walking speed, and what it costs

A summary of what the speed rounds (2026-10-15 … 2026-10-18) showed, for anyone training their own
Microduck skills on top of `microduck_rl`. Every number comes from the evaluation battery
([`scripts/eval_battery.py`](../scripts/eval_battery.py): 256 randomised ducks, 3 seeds, the same
environment for every policy); every run, command and raw JSON is linked from
[`training-log.md`](training-log.md).

## 1. Every walk policy walked at about 60 % of the command — a reward-width problem

| command | official walk | our walk v1 | **walk teacher v3** |
|---|---|---|---|
| 0.15 / 0.30 / 0.45 m/s | 0.00 / 0.15 / 0.21 | 0.10 / 0.20 / 0.27 | **0.16 / 0.31 / 0.44** |

The stock forward-velocity term has a width (std) of 0.32 m/s, so walking 0.1 m/s too slowly costs
only about 10 % of that reward, while longer steps cost smoothness, foot and effort penalties. PPO
finds the cheaper compromise: a short, slow gait. The fix
([`patches/speed.patch`](../patches/speed.patch), `MICRODUCK_SPEED=1`) adds a narrow forward-only term
(std 0.10, weight 2.0) **next to** the stock one and widens the forward command range to −0.4…0.5 m/s.
The same reasoning fixed the in-place turning dead zone earlier ([`patches/turn.patch`](../patches/turn.patch)).

**Take-away:** when a policy under-tracks a command, check the width of the tracking kernel before
anything else; the policy is usually doing exactly what the reward prices.

## 2. In the walk + get-up network, the walk teacher decides the gait

The combined policy (VelStand) is trained with PPO plus two imitation terms: the stand-up teacher on
fallen frames (tilt > 35°) and a walk teacher, the *anchor*, on upright frames (tilt < 25°). Three
rounds changed only the anchor, and each time the combined policy inherited the anchor's gait almost
exactly:

| round | anchor | combined policy |
|---|---|---|
| v5 | walk teacher v2 (turns, slow) | turns, slow |
| v6 | walk teacher v3 (turns, fast) | 0.30 → 0.30 m/s, turn 0.6 → 0.60 rad/s |
| A | walk teacher v3, **no** speed reward inside VelStand | still fast (0.28 m/s) |

**Take-away:** to change how the combined policy walks, train and *gate* a new walk teacher first
(battery tracking trial on every checkpoint, pass/fail before any distillation). v5 was distilled from
an ungated teacher and lost speed nobody had measured.

## 3. The fast anchor costs push recovery — and the battery shows exactly where

Every fast combined policy recovered fewer push falls than v4 (anchored to the old, slow teacher):

| push recovered | v4 (release v0.1) | v6 | A | B |
|---|---|---|---|---|
| standing 1.0 / 1.2 m/s | 95 / 97 % | 89 / 84 % | 89 / 83 % | 89 / 87 % |
| walking 1.0 / 1.2 m/s | 95 / 97 % | 89 / 92 % | 87 / 88 % | 93 / 92 % |
| speed at 0.30 m/s | 0.18 | 0.30 | 0.28 | 0.29 |

The obvious explanation — a faster gait is less stable — turned out to be wrong. Two cheap battery
runs ([`results/benchmark/diag_push/`](../results/benchmark/diag_push/)) separated *falling* from
*getting up*:

1. **Falling:** the fast walk teacher falls about as often as the old one under the same pushes, and
   A falls about as often as v4.
2. **Getting up:** the ducks that are not recovered end up in one state — **lying on the side or back**
   (trunk at 64 mm, tilted ~97°) for the whole 6 s window; not half-up, not leaning. The get-up from
   ground spawns erodes the same way: face-up 97.5 → 93.9 %, on-side 98.7 → 96.9 % from v4 to A.

So the get-up from side / back — frames the anchor never imitates — erodes while the network learns
the faster gait; the two skills compete for the same network and the same learning signal. The push
trial now reports this split for every run (`failed_lying` / `failed_low` / `failed_up_but_tilted`,
median height and tilt of the failures).

**Take-away:** report *where* the failures end up, not just the failure rate. One extra field in the
push trial turned "it is less stable" (wrong) into "it forgets the side / back get-up" (testable).

## 4. Aimed at the diagnosis: candidate C

C = A's configuration with the stand-up teacher's imitation on fallen frames ×3
(`MICRODUCK_EXPERT_BC_COEF=3.0`) and more episodes that start lying on the side or back
(`MICRODUCK_GROUND_MIX=1`), so the get-up keeps getting signal while the gait changes.

Battery, 3 seeds (`model_2250`, picked by battery sweep over 2000–2999):

| | v4 (release v0.1) | A | **C** | official v7 |
|---|---|---|---|---|
| get-up face-down / face-up / on-side | 99.7 / 97.5 / 98.7 % | 99.7 / 93.9 / 96.9 % | **100 / 97.7 / 98.7 %** | 100 / 91 / 94 % |
| push recovered, standing 1.0 / 1.2 m/s | 95.3 / 96.7 % | 89.2 / 82.9 % | **93.3 / 91.9 %** | 93 / 92 % |
| push recovered, walking 1.0 / 1.2 m/s | 94.9 / 96.6 % | 87.2 / 88.1 % | **93.4 / 94.8 %** | 96 / 94 % |
| speed at 0.30 / 0.45 m/s | 0.18 / 0.23 | 0.28 / 0.39 | **0.29 / 0.40** | 0.14 / 0.22 |
| turn at 0.6 / ±1.0 rad/s | 0.39 / 1.10, −0.90 | 0.55 / 1.00, −0.99 | **0.58 / 1.02, −1.00** | 0.05 / 0.38, −0.32 |
| crouch for −30 mm | −28.7 | −27.8 | **−28.7** | −0.7 |
| 1.5 m goal after a push: reached; median time unpushed | 96 %; 9.8 s | 93 %; 6.6 s | **95 %; 6.5 s** | 94 %; 14.3 s |

- **The targeted fix worked on its target.** The get-up from ground spawns is fully back (face-up
  93.9 → 97.7 %, on-side 96.9 → 98.7 %, level with v4), crouch is back to −28.7 mm, and the speed and
  turning stayed (0.29 m/s, 0.58 rad/s).
- **Push recovery came back most of the way, not all of it:** 92–95 % against A's 83–89 % and v4's
  95–97 %. The ducks that are still not recovered after a push are again mostly lying (83–95 % of the
  failures), so a get-up after a *push* differs from a get-up from a *spawn* in a way the extra
  ground starts do not cover (momentum, joint configuration on landing, or the time it takes).
- C misses the agreed release gate by 0.2–3 points on four push cells and the pushed goal (94.7 vs
  96 %). By the rule fixed in advance **v4 stays the default** and C is published as the fast
  variant. Against the official policy C is level on push recovery and ahead on get-up, speed,
  turning, crouch and time to goal.

The next round (running) resumes from C with one change per variant (anchor weight halved, stand-up
imitation ×5, both, speed term at half weight) and ranks them with the same battery
([`scripts/search_board.py`](../scripts/search_board.py)).

## 5. Method notes that held up

- **Pick checkpoints by battery, never "the last one".** Push recovery peaked mid-run and dropped again
  in v6 (0.86 at 2250, 0.80 at 2750–2999) while speed stayed flat.
- **Agree the release gate before the run** ([`scripts/release_gate.py`](../scripts/release_gate.py)):
  a candidate replaces the default only if it is at least as good on every push-recovery cell, get-up,
  crouch and the pushed goal. Nothing gets promoted on a good-looking headline number.
- **One change per round.** A (anchor only, no in-network speed term) answered in one run what v6
  could not: the cost comes with the anchor, not with the reward.
- **Know what a logged number is.** `expert_bc_fallen_frac` / `expert_bc_anchor_frac` in the training
  log are the *shares* of frames each imitation term sees, not imitation losses; the loss is
  `Mean expert_bc loss` and scales with the coefficient.

## Reproduce

Upstream `microduck_rl` at `ceca0170`, then `git apply patches/all-in-one.patch` (every change is
behind a `MICRODUCK_*` environment variable and off by default). The pipelines that ran these rounds
are in [`scripts/`](../scripts/) (`pipeline_round4_speed.sh` … `pipeline_round7_release_prep_and_search.sh`).
