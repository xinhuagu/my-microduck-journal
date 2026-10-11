# Training log

All runs: `microduck_rl`, RTX 5090, WSL2 Ubuntu 24.04, `WANDB_MODE=offline`. Iterations are PPO
iterations (24 env-steps each). Checkpoints every 250 iterations.

| Date | Run | Task | Envs | Iterations | Wall-clock | Outcome |
|---|---|---|---|---|---|---|
| 2026-09-06 | `velocity` | `Mjlab-Velocity-Flat-MicroDuck` | 4096 | 4 (aborted) | — | first launch, stopped early |
| 2026-09-07 | `smoke-test` | `Mjlab-Velocity-Flat-MicroDuck` | 64 | 5 | < 1 min | stack verified |
| 2026-09-07 → 09-08 | **`first-walk`** | `Mjlab-Velocity-Flat-MicroDuck` | 4096 | **10 000** | 0→1000: 30 min · 0→2500: 69 min · paused overnight · ended 09-08 23:42 | walks; falls on a `P` push and cannot get up (expected for Velocity) |
| 2026-10-10 | smoke after `git pull` to `ceca0170` | `Mjlab-Velocity-Flat-MicroDuck` | 64 | 5 | 42 s | OK, 1.5–1.75 k steps/s |
| 2026-10-10 | export-all | — | — | 41 checkpoints → ONNX | 41 × ~13 s ≈ 9 min | `exports/first_walk_{0..10000}.onnx`, 793 766 B each |
| 2026-10-10 | smoke | `Mjlab-VelStand-Flat-MicroDuck` | 64 | — | 14 s | **failed**: wandb 403 on the private teacher checkpoints |
| 2026-10-10 23:57 → 10-11 02:28 | **`microduck_stand`** (stand-up teacher) | `Mjlab-StandUp-Flat-MicroDuck` | 4096 | 6 000 (last checkpoint `model_5999.pt`) | **2 h 30 min**, 1.42–1.5 s/iteration, 25 checkpoints | push test below: 4500 and 5999 get up 8/8, 3000 sits 8/8 |
| 2026-10-11 02:41 → 04:16 | **`velstand`** (walk + get-up) | `Mjlab-VelStand-Flat-MicroDuck` | 4096 | 3 000 (last checkpoint `model_2999.pt`) | **1 h 35 min**, 1.70 s/iteration, 13 checkpoints | warm start from `first-walk/model_10000.pt`, expert `microduck_stand/model_5999.pt`, anchor `model_10000.pt`; **gets up 8/8 standing and 8/8 walking** — push tests below |
| 2026-10-11 07:59 → 08:35 | **`microduck_stand` v2** (stand straight) | `Mjlab-StandUp-Flat-MicroDuck` + `MICRODUCK_STANDUP_STRAIGHT=1` | 4096 | 1 500 | 36 min, warm start from `model_5999.pt` | **ends upright** — posture probe after pushes: mean \|pitch\| 0.2° (v1: 34°), neck back at 0°; `upright_sharp` reward 0.10 → 4.75 of 5, mean reward 4 → 89. Fix: upright_sharp 5.0 (curriculum pinned), composite upright_std 0.25, head_pose_tracking 2.0 / fine_std 0.15 |
| 2026-10-11 08:37 → 11:02 | **`microduck_stand` v2** (continued) | same | 4096 | +6 000 (→ 7 498) | 2 h 26 min, resume from `model_1499.pt` | **final teacher** `model_7498.pt`, total budget 6 000 + 1 500 + 6 000 = 13 500 iterations. Probe: stands at pitch 0.6° throughout, neither probe push takes it down; push test: went down 1/8 (to 46 mm, back up), upright at end **8/8** |
| 2026-10-11 11:04 → | **`velstand` v2** (walk + get-up) | `Mjlab-VelStand-Flat-MicroDuck` | 4096 | 6 000, **running** | warm start from `first-walk/model_10000.pt`, expert = teacher v2 `model_7498.pt`, anchor = walk | 1.67 s/iteration → ≈ 2 h 50 min; then VelStand v3 (Rough-Backlash, 3 000) follows automatically |

## `microduck_stand` — iteration 500 check (2026-10-11 00:12)

Exported `model_500.pt` while training continued and loaded it as the *standing* policy in the
CPU viewer (`infer_policy.py --standing exports/standup_500.onnx --new-cmd-obs`), then sent a
random 1 m/s push (`P`).

| before the push | after the push |
|---|---|
| ![standing, 500 iterations](../results/standup-500-before-push.jpg) | ![sitting after a push, 500 iterations](../results/standup-500-after-push.jpg) |
| stands, slightly crouched, `trunk_z ≈ 117 mm` | does not fall over — but settles into a **sit**, `trunk_z ≈ 63 mm`, and stays there |

So at 500 iterations the policy has learned "don't topple" but not "get back to standing": a
sit is a safe, cheap equilibrium. This is exactly the local optimum upstream describes in
`microduck_velstand_env_cfg.py` ("the policy learned to sit and farm upright … gating a positive
reward on a bad state rewards entering the state") and the reason their recovery gates are on
tilt only. Whether the StandUp curriculum climbs out of it on its own by 2 000–3 000 iterations is
the thing to watch next; checkpoints are every 250 iterations.

## `first-walk` details

- Config as shipped in `microduck_velocity_env_cfg.py` at commit `29e887e`: 16 reward terms
  (`track_linear_velocity` 2.0, `track_angular_velocity` 2.0, `upright` 2.0, `air_time` 3.0,
  `foot_clearance` −2.0, `action_rate_l2` −0.1 → curriculum, ...), command ranges ±0.4 m/s forward,
  ±0.3 m/s lateral, ±1.0 rad/s yaw; BAM `m6` XL330 actuator model with `vin_range=(6.5, 8.2)`,
  `delay 3–6` steps; PPO `(512, 256, 128)` ELU, clip 0.2, entropy 0.01, adaptive LR 1e-3.
- `max_iterations` was set to 50 000; the run was stopped manually at 10 000.
- Exported checkpoints used in the viewer: 2500, 5000, 10 000. Subjectively 10 000 tracks the
  0.30 m/s command more closely and wobbles less than 5000; a proper walk sweep
  (`achieved` vs `cmd` at 0.3 / 0.4 m/s, yaw drift) is still to be recorded here.


## `microduck_stand` — push test, all checkpoints (2026-10-11 02:40)

Method: export the checkpoint, load it as the *standing* policy in the CPU viewer
(`infer_policy.py --standing exports/standup_<N>.onnx --new-cmd-obs`, driven inside `tmux`), send
8 random 1 m/s pushes (`P`) 6 s apart, read `trunk_z` from the status lines. Standing is
≈116 mm, sitting ≈63 mm, prone ≈50 mm. "Moved" = the push changed `trunk_z` by more than 3 mm
(the push direction is random, some barely touch it).

| checkpoint | pushes that moved it | went below 90 mm | upright (≥100 mm) 6 s later | behaviour |
|---|---|---|---|---|
| 500 | — | — | — | sits down after one push and stays (see above) |
| 2000 | 8/8 | 6/8 | **2/8** | sits after the first hard push, never gets up |
| 3000 | 8/8 | 8/8 | **0/8** | sits every time |
| 4500 | 4/8 | 3/8 | **8/8** | goes down to 58–79 mm and stands back up within the window |
| **5999** | 4/8 | 3/8 | **8/8** | goes down as far as 51 mm (near prone) and stands back up every time |

So the sit-instead-of-stand local optimum that showed at 500 persisted through 3000 and was
trained out between 3000 and 4500 — consistent with the config comments ("ground_state_mix ramps
face-down/face-up until iter 2500", recovery economics after that). The final checkpoint
`model_5999.pt` is the teacher for VelStand. Caveat: 8 pushes from one viewer session is a smoke
test, not the per-spawn-type battery upstream uses.

Training-curve note: `Episode_Reward/standing_composite` *falls* from 2.34 (iteration 1000) to
0.73 (5999) while the policy clearly gets better — because the curriculum lowers the share of
episodes that start already standing from 40 % to 15 %, so the average height reward over an
episode drops. Read these curves against the curriculum stage, not on their own.


## `velstand` — push tests (2026-10-11 04:30)

Same method as above (8 random 1 m/s pushes, 6 s apart, `trunk_z` from the status lines), now with
the VelStand checkpoints loaded as the *walking* policy (`infer_policy.py --walking
exports/velstand_<N>.onnx --new-cmd-obs`). Exporting a VelStand checkpoint needs
`MICRODUCK_STAND_EXPERT` / `MICRODUCK_WALK_ANCHOR` set as well, because `export.py` builds the
full runner, teachers included.

**Standing still** (no command):

| checkpoint | pushes that moved it | went below 90 mm | upright 6 s later |
|---|---|---|---|
| 1250 | 7/8 | 6/8 (down to 48 mm) | **8/8** |
| 2000 | 4/8 | 0/8 (never below 107 mm) | **8/8** |
| 2999 | 4/8 | 1/8 (75 mm) | **8/8** |

**Walking at 0.30 m/s** (two `Up` presses before the pushes), with the original walk policy as the
control:

| policy | speed before pushes (achieved / cmd) | went below 90 mm | upright 6 s later |
|---|---|---|---|
| `velstand_1250` | 0.14 / 0.30 | 7/8 (down to 43 mm) | **8/8** |
| `velstand_2999` | 0.15 / 0.30 | 5/8 (down to 43 mm) | **8/8** |
| `first_walk_10000` (Velocity, control) | 0.18 / 0.30 | 8/8 | **0/8** — stays on the ground at 40–50 mm |

Reading: by 2000 iterations the policy mostly *resists* a 1 m/s push while standing; when it does
go down — and while walking it still goes down most of the time — it stands back up within the
6 s window every time, from as low as 43 mm (prone). The control run shows what the pure walking
policy does with the same pushes: flat on the floor, 8/8. Cost: achieved forward speed at the
0.30 m/s command dropped from 0.18 (walk policy) to 0.14–0.15 m/s, and right after a recovery the
duck stands for a moment before walking again (the `fwd=+0.00/+0.30` lines at the end of the
windows). A proper walk sweep (0.3 / 0.4 m/s, yaw drift) and a per-spawn-type get-up battery are
still to be done; 8 pushes is a smoke test.

Training curves for this run are misleading on their own: `expert_bc_fallen_frac` sits at 0.6–0.7
and `recovery_success` at ≈0.003 for most of the run, because 45 % of episodes start prone and
`topple_push` ramps to 1.2 m/s — the policy spends most of its time on the ground *by design*.


## Posture probe: the get-up ends in a lean, not a stand (2026-10-11 07:5x)

Measured with the viewer's `--debug` output (projected gravity → trunk pitch, joint angles relative
to the default standing pose), one line per second, pushes injected through tmux:

| policy | before any push | after a push |
|---|---|---|
| `first_walk_10000` (walk) | z 0.117 m, pitch 1°, joints ≈ default | falls, stays down |
| `standup_5999` (teacher) | z 0.116, pitch 0°, joints ≈ default | **z 0.115, pitch 34°, neck_pitch −110° (joint limit), legs ≈ default — and it stays there** |
| `standup_4500` (teacher) | same | same lean, 34° / −110° |
| `velstand_2999` | z 0.117, pitch 1° | after a get-up: **pitch 35°, neck −110°**; after a later push it happened to return to 1° |

So "gets up" in the push tests meant "trunk back at standing height" — the policy stands at the
right height but pitched 34° forward with the head tucked into the chest, and never straightens.
The StandUp config even names this attractor: the *lean basin* ("lean trunk at the right
height"). Two things in the stock recipe let it win here:

1. `upright_sharp` (Gaussian on tilt, gated by height) is **faded by a curriculum**:
   1.5 → 1.0 at iteration 3000 → 0.5 at 4000. By the time the get-up skill exists, standing
   vertical pays almost nothing extra; `standing_composite` (upright_std 0.40 ≈ 23°) still gives
   ~0.3 at 34°, `height_stand*` are fully paid, and the head-pose term (std 0.5 rad) barely prices a
   tucked neck.
2. VelStand inherits the pose through the expert BC and sits exactly in the gap between its two
   teachers: the walk anchor covers tilt < 25°, the stand expert tilt > 35°. At 34° nobody teaches,
   and `RECOVERED_UP_TILT_DEG = 25` means the recovery bounty never fires — which is why
   `recovery_success` stayed ≈ 0.003 all run.

Fix (teacher side, [`patches/standup-stand-straight.patch`](../patches/standup-stand-straight.patch),
enabled with `MICRODUCK_STANDUP_STRAIGHT=1`): `upright_sharp` weight 5.0 with std 0.2 and the fade
curriculum pinned at 5.0; `standing_composite` upright_std 0.25; `head_pose_tracking` weight 2.0 with
`fine_std` 0.15. Fine-tune from `model_5999.pt` with `MICRODUCK_WARM_START=1`, 1 500 iterations.
Then VelStand gets re-trained with the new teacher.

### Result: teacher v2 stands straight (2026-10-11 08:37)

1 500 iterations, 36 minutes. Same probe as above on `standup_v2_1499.onnx`, two pushes, one row
per second (50 control steps):

| step | phase | z (m) | pitch | roll | neck_pitch | note |
|---|---|---|---|---|---|---|
| 650 | before | 0.116 | 0.3° | −0.2° | −0.1° | standing, joints ≈ default |
| 700 | push 1 | 0.089 | 32.6° | 22.4° | −54.5° | going down |
| 750 | push 1 | 0.088 | 70.4° | 82.5° | −112.7° | on the ground |
| 800 | push 1 | 0.116 | **0.7°** | 0.0° | −11.6° | back up after ≈ 1 s |
| 850 | push 1 | 0.116 | −0.1° | −0.1° | −0.7° | neck straight |
| 1050 | push 2 | 0.064 | 40.0° | 67.8° | −51.4° | down, lower than any v1 spawn |
| 1100 | push 2 | 0.100 | 43.6° | 6.1° | −110.5° | *the old lean pose — now a transit state* |
| 1150 | push 2 | 0.106 | 14.9° | 1.2° | −106.1° | straightening |
| 1200 | push 2 | 0.118 | −0.1° | −3.6° | −55.0° | upright, head coming up |
| 1250 | push 2 | 0.116 | **−0.2°** | 0.0° | −0.5° | standing straight |

Mean |pitch| over the last four rows: **0.2°** (v1 teacher: 34°). The lean pose still shows up on
the way up (step 1100: pitch 44°, neck at its −110° limit), but it no longer pays enough to stay
in — with `upright_sharp` at weight 5 / std 0.2 the policy keeps pushing to vertical and lifts the
head. Reward terms over the fine-tune: `upright_sharp` 0.10 → 4.75 (of a possible 5),
`head_pose_tracking` 0.06 → 1.84, mean reward 4.1 → 88.9, episode length 21 → 300. Read these as
convergence of *this* run only: the first iteration is low because the warm start resets the
curricula and the terminations, and the reward *definition* changed (weights 5.0 / 2.0 instead of
1.5 / 0.5), so the totals are not comparable with the v1 teacher's. The posture probe is the
comparison. Also open: v2 changed three reward terms, pinned a curriculum and added budget at once;
which of those did the work needs an ablation (one change at a time from `model_5999.pt`, 1 500
iterations each) — planned.

What happens next is scripted (one shell script, logged to `logs/pipeline.log`): continue this
teacher for another 6 000 iterations (the stock teacher in the upstream lineage was trained for
≈ 10 000; this one ends at 13 500 in total), probe + push test, then **VelStand v2** on Flat for
6 000 iterations (warm start from `first-walk/model_10000.pt`, expert = teacher v2, anchor = walk),
push tests, then **VelStand v3** on `Rough-Backlash` for 3 000 iterations warm-started from v2.

### Result: final teacher `model_7498.pt` (2026-10-11 11:02)

Plain `--agent.resume` (not a warm start) from `model_1499.pt`, so the iteration counter and the
stock curricula continue: 1 499 → 7 498 in 2 h 26 min (1.30 s collection + 0.07 s learning per
iteration). The reward totals *drop* at iteration 4 000 — mean reward 89.9 → 70.9 — but that is
the stock weight curricula, not a regression: `height_stand_sharp_weight` 0.5 → 0.2,
`standing_composite_weight` 2.5 → 1.5, `body_pose_tracking_weight` 3 → 4, `head_pose_bias_weight`
0.5 → 1.5, while the pinned `upright_sharp_weight` stays at 5.0 and the term itself stays at
4.57–4.67 of a possible 5.

| iteration | mean reward | `upright_sharp` | `head_pose_tracking` | `standing_composite` | episode length |
|---|---|---|---|---|---|
| 1 500 | 11.2 | 0.47 | 0.21 | 0.34 | 45 (just resumed) |
| 3 000 | 89.9 | 4.67 | 1.51 | 3.31 | 300 |
| 4 500 | 70.9 | 4.57 | 1.46 | 1.33 (weight 1.5) | 300 |
| 7 498 | 71.2 | 4.59 | 1.50 | 1.33 | 300 |

Posture probe on `standup_v2_7498.onnx`: pitch 0.5–1.0° the whole time, neck −3°; neither probe
push takes it down any more (roll −30° at the push, back to 0.2° one second later, min z 0.115 m).
Push test, 8 random 1 m/s pushes while standing: **moved 3/8, went down 1/8** (push 7: down to
46 mm, back to 117 mm), **upright at end 8/8**. So the final teacher mostly resists the push, and
gets up straight when it does fall — which is what VelStand will now be distilled from. The
teacher was never trained for walking; its forward speed in the test is 0.00 by design.

## Body-pose commands: what the policies actually do with them (2026-10-11 14:05)

The 13-D command vector has six body-pose slots (x, y, z, roll, pitch, yaw); the viewer drives
z / roll / pitch in body mode (`B`, then arrows and `A`/`E`, 10 mm or 10° per key). Measured
while standing, with the viewer's `--debug` output (`scripts/wsl_bodypose_probe.sh` in the
scratch tooling: z from CoM height, tilt from projected gravity, last 2 s of each command):

| command | `velstand_2999` (v1) | `standup_v2_7498` (teacher v2) |
|---|---|---|
| crouch −20 mm | −1.5 mm | −2.2 mm |
| crouch −30 mm | −3.1 mm | −1.9 mm |
| raise +20 mm | −0.6 mm | 0.0 mm |
| pitch 10° | **9.2°** | 3.0° |
| pitch 20° | 1.9° (outside the ±12° training range: gives up) | 0.2° |
| roll 10° | 1.7° | 1.0° |

Neither policy crouches; v1 tracks a pitch command only inside the ±12° it was trained for, and
the stand-straight teacher v2 answers 10° with 3°. Three things in the stock StandUp recipe
explain the teacher: `height_stand` / `height_stand_sharp` / `height_stand_l1`, `upright_sharp`
and `standing_composite` all pull toward **one** fixed pose (`STAND_Z`, vertical), so every
commanded crouch or tilt is taxed by the strongest terms in the stack. Upstream handles the
conflict by *fading* those terms (the `*_weight` curricula at iterations 3000/4000) while
`body_pose_tracking` ramps up to 4.0 — and the stand-straight patch above pinned `upright_sharp`
at 5.0 precisely to stop the lean basin, which also killed commanded tilt. Upstream's VelStand
notes agree on the crouch side: "z (crouch) is NOT learned" (its stand expert never crouched, and
PPO alone did not learn it).

Fix ([`patches/body-wide.patch`](../patches/body-wide.patch), enabled with `MICRODUCK_BODY_WIDE=1`):

- Four command-relative reward functions added to `mdp.py`: `height_target_gaussian_cmd`,
  `height_l1_penalty_cmd`, `upright_gaussian_at_height_cmd` (tilt measured against the commanded
  roll/pitch, same convention as upstream's `upright_body_cmd_relative`; height gate shifts with
  the commanded z) and `standing_composite_score_cmd` (height and upright targets follow the
  command; the joint-pose factor's std widens from 0.4 to 0.6 while a command is active, because a
  crouch bends the knees). At zero command all four are identical to the stock terms, so the
  straight stand from the previous round is kept.
- StandUp teacher: those terms swapped in, `body_pose_tracking` pinned at weight 4.0 from
  iteration 0 (warm start), the relaxed sharp-stand weights (0.2 / 1.5) pinned too, ranges crouch
  −40 mm / +30 mm and tilt ±15° from the start, ±20° from iteration 500.
- VelStand: ranges ±20° and −40 / +20 mm, tracking weight 3.0, and `body_no_teacher_axes = ()`
  in the distillation config so the stand expert teaches the crouch frames as well (upstream
  leaves z frames to PPO because its expert cannot crouch; ours is trained to).
- Viewer: `BODY_CMD_MAX_Z` 30 → 40 mm so the crouch can be commanded at all.

Smoke tests (3 iterations × 64 envs) pass for both tasks; the teacher shows
`Curriculum/body_pose_tracking_weight 4.0`, `body_pose_range 0.2618` (15°) and the relaxed sharp
weights at iteration 0. Queued as pipeline part 3, after VelStand v3 finishes: teacher v3 (3 000
iterations, warm start from `model_7498.pt`), posture probe (must still stand straight) + body
probe, then VelStand v4 (3 000 iterations on the v3 task, expert = teacher v3), push tests +
probes. Protective fall is a separate later round.

## Next

1. ~~StandUp v2 → posture probe: does it end at pitch ≈ 0° after a push?~~ Yes (0.2°).
2. ~~Teacher v2 continued to 7 498 → probe + push test.~~ Done: upright 8/8, down only 1/8.
3. VelStand v2 (Flat, 6 000, running) with the new teacher; posture probe + push tests; compare with v1.
4. VelStand v3 (Rough-Backlash, 3 000) warm-started from v2; push tests.
5. Body-pose round (pipeline part 3, queued): teacher v3 + VelStand v4 with crouch / tilt that
   actually track; then a protective-fall round (tuck the head, land softly) in the teacher.
6. Walk sweep, per-spawn-type get-up battery, README get-up sequence, Hub publish.
