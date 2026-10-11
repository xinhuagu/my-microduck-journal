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
| 2026-10-11 11:04 → 14:32 | **`velstand` v2** (walk + get-up) | `Mjlab-VelStand-Flat-MicroDuck` | 4096 | 6 000 | 1 h 30 + 1 h 50 (the PC went to sleep at iteration 2 784; resumed from `model_2750.pt` at 12:42, curricula continued) | warm start from `first-walk/model_10000.pt`, expert = teacher v2 `model_7498.pt`. **Gets up straight**: walking pushes down 5/8, back up 5/5 within ≈ 1 s at pitch 1–3°; standing pushes down 0/8. `recovery_success` 0.018 (v1: 0.003). Lost the tilt tracking v1 had (10° → 3.6°) — see the body-pose section |
| 2026-10-11 14:35 → 17:50 | **`velstand` v3** (rough + backlash) | `Mjlab-VelStand-Rough-Backlash-MicroDuck` | 4096 | 3 000 | **3 h 15 min**, 3.7 s/iteration (2.2× the flat task) | warm start from v2 `model_5999.pt`, same teachers. On the flat test scene: standing pushes down 2/8, walking (0.14 m/s) down 6/8, upright 8/8 both, get-ups straight (pitch 1–2°). Sweep → **use `model_2500`** (drift −0.01; 2999 −0.16). `recovery_success` 0.019 |
| 2026-10-11 17:53 → 19:11 | **`microduck_stand` v3** (body-wide) | `Mjlab-StandUp-Flat-MicroDuck` + `MICRODUCK_STANDUP_STRAIGHT=1` + `MICRODUCK_BODY_WIDE=1` | 4096 | 3 000 | 1 h 18 min, warm start from `model_7498.pt` | **stands straight (0.4°) and tracks the body command**: crouch −30 mm → −28.9 mm, pitch ±10° → 9.9 / −8.8°, ±20° → 19.5 / −19.3°, roll ±10° → 9.6 / −9.4°, raise +20 mm → +12.5 mm (teacher v2: −2 mm, 3°). `body_pose_tracking` 3.25 of 4 |
| 2026-10-11 19:13 → | **`velstand` v4** (body-wide) | `Mjlab-VelStand-Rough-Backlash-MicroDuck` + `MICRODUCK_BODY_WIDE=1` | 4096 | 3 000, **running** | warm start from v3 `model_2999.pt`, expert = teacher v3, anchor = walk | the stand expert now teaches the crouch frames too (`body_no_teacher_axes = ()`) |

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

## `velstand` v2 — results (2026-10-11 14:45)

Same 8-push smoke test as v1 (random direction, 1 m/s, 6 s window), `velstand_v2_5999.onnx`:

| mode | pushes that moved it | went below 90 mm | upright 6 s later |
|---|---|---|---|
| standing | 1/8 | **0/8** | 8/8 (never fell) |
| walking at 0.30 m/s (achieved 0.16) | 6/8 | **5/8** | **8/8** |

The question from the v1 round was *how* it stands afterwards. Posture probe while walking
(`posture_probe.sh … --walking 6 "Up Up"`, one row per second; pitch from projected gravity):

| event | lowest point | 1 s later | 2 s later |
|---|---|---|---|
| stumble while walking (no push) | z 0.054 m, pitch −165° (on its back) | z 0.051, −89° | z 0.122, **2.5°** |
| push 1 | z 0.073, −164° (on its back) | z 0.124, **4.0°** | 1.1°, walking again |
| push 2 | z 0.065, 61° | z 0.125, 5.9° (neck −97° for a moment) | **0.7°**, neck −7° |
| push 3 | z 0.112 (stumble only) | 3.0° | 2.6° |
| push 4 | z 0.092, 64° | z 0.122, **1.4°** | 3.1° |
| push 5 | z 0.076, 145° (face down) | z 0.122, **2.8°** | 1.1° |

Every fall — on the back, face down, sideways — ends standing within about a second, at the 1–3°
pitch of the walking gait, with the neck at −5°: the 34° lean basin of v1 is gone, and the gait
resumes (joint angles cycling, forward speed back at 0.16 m/s). The training curve agrees in its
own units: `recovery_success` (the bounty that only fires when the trunk is back within 25° of
vertical) averaged 0.018 per step over the last 2 000 iterations against 0.003 for v1, and the
expert-BC loss on fallen frames settled at 0.20 (v1 never got below 0.6, because its teacher's
targets were the lean).

What v2 lost: the body-pose probe gives pitch 10° → 3.6° (v1: 9.2°), 20° → 0.1°, crouch −30 mm
→ −1 mm. Expected — teacher v2 does not tilt either — and it is what the body-pose round below is
for. Cost of the recovery skill on walking is unchanged from v1: 0.16 m/s achieved at a 0.30 m/s
command (walk policy alone: 0.18).

## `velstand` v3 — rough terrain + backlash (2026-10-11 18:00)

Same recipe as v2, continued for 3 000 iterations on `Mjlab-VelStand-Rough-Backlash-MicroDuck`
(randomised terrain, actuator backlash) from `v2/model_5999.pt`. The rough task costs 3.7 s per
iteration against 1.7 on flat, so 3 000 iterations took 3 h 15 min. Tests are on the flat viewer
scene (the viewer has no rough terrain), so they say "nothing was lost on flat", not "works on rough":

| mode | pushes that moved it | went below 90 mm | upright 6 s later |
|---|---|---|---|
| standing | 6/8 | 2/8 | 8/8 |
| walking at 0.30 m/s (achieved 0.14) | 7/8 | 6/8 | 8/8 |

Posture probe (standing, 3 pushes): the one push that took it down to 93 mm ended at pitch 1.2°
two seconds later, neck −6°. Standing pitch is 1.2–1.6° (v2: 0.3°), a trace of the backlash
training. Body-pose commands: as v2 (crouch −30 mm → −1.5 mm, pitch 10° → 4.6°).

Sweep (`turn_sweep.sh`, yaw drift at 0.30 m/s / in-place 0.6 rad/s): 1500 −0.18 / 0, 1750 −0.05 / 0,
2000 −0.20 / 0, 2250 −0.04 / 0, **2500 −0.01 / 0.01**, 2750 +0.05 / 0, 2999 −0.16 / 0. So `model_2500`
is the v3 checkpoint to use. (Pipeline part 3 warm-starts v4 from the last checkpoint, 2999; drift
re-rolls with every training anyway and v4 gets its own sweep in part 4.)

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

## Stand-up teacher v3: straight *and* obedient (2026-10-11 19:15)

3 000 iterations from `model_7498.pt` with the command-relative rewards
([`patches/body-wide.patch`](../patches/body-wide.patch)), 1 h 18 min. Posture probe: pitch 0.4°
after pushes, so the command-relative `upright_sharp` kept the straight stand at zero command.
Body-pose probe, one command per fresh viewer launch (`scripts/bodypose_probe_single.sh`):

| command | teacher v2 `model_7498` | **teacher v3 `model_2999`** |
|---|---|---|
| crouch −30 mm | −1.9 mm | **−28.9 mm** |
| raise +20 mm | 0.0 mm | +12.5 mm (range is +30, the legs do not extend further) |
| pitch +10° / −10° | 3.0° / — | **+9.9° / −8.8°** |
| pitch +20° / −20° | 0.2° / — | **+19.5° / −19.3°** |
| roll +10° / −10° | 1.0° / — | **+9.6° / −9.4°** |

Training terms at 2 999: `body_pose_tracking` 3.25 (of a possible 4), `upright_sharp` 4.48 (of 5),
mean reward 74. That is the "crouch is not learned" line from the upstream notes answered on the
teacher side: once the stand rewards follow the command, PPO learns the crouch in 3 000 iterations
with no other change.

A probe artefact worth recording: the earlier sequential probe (`bodypose_probe.sh`, several
commands in one viewer session separated by `Space` resets) showed only the first command tracked
(pitch 10° → 10.0°, then 20° → 0.5°, roll → 0.1). Re-running each command from a fresh launch gives
the table above, so the sequential script, not the policy, was at fault (the viewer's body-mode
state after a reset is suspect). Pipeline part 3 still uses the sequential probe for v4; the v4 and
v5 numbers below are re-measured with the single-launch probe.

## Evaluation battery: first statistics (2026-10-11 21:10)

Until now every number came from one duck in the CPU viewer with keys injected through tmux — a
smoke test. [`scripts/eval_battery.py`](../scripts/eval_battery.py) (wrapper
[`eval_battery.sh`](../scripts/eval_battery.sh)) runs the policy from its rsl_rl checkpoint in
**256 parallel mjlab environments on the GPU, fixed seed**, with the task's domain randomisation on
and its interval events (training pushes) off, and forces spawn states and commands directly:

- *Get-up per spawn type*: face-down / face-up / on-side orientation from the StandUp recipe, a random
  sprawled joint pose (±0.5 rad), dropped from 12 cm and held for 1 s with the policy off; then the
  clock starts. "Recovered" = trunk ≥ 100 mm and tilt ≤ 15° for one continuous second.
- *Push falls*: standing or walking at 0.30 m/s for 3 s, then the trunk's horizontal velocity is set to
  the push speed in a random direction (the viewer's and the training event's definition); "fell" =
  trunk below 90 mm; recovery as above, timed from the push.
- *Tracking*: 8 s per command, mean over the last 5 s of body-frame forward speed and yaw rate; the
  per-env |yaw| median / p90 says whether a drift is systematic or a lottery across randomised ducks.
- *Body-pose commands while standing*: 5 s per command, last 2 s measured.

A full battery takes ≈ 15 min on the 5090 while a training run shares the GPU. The script needs a
`.pt` checkpoint (it builds the rsl_rl runner, teachers included); ONNX input is on the list so the
official policy can be measured with the same code.

**VelStand v2 `model_5750`** (`Mjlab-VelStand-Flat-MicroDuck`, seed 0, 256 envs):

| get-up from | started down | recovered | median | p90 | upright at end | end pitch |
|---|---|---|---|---|---|---|
| face-down | 100 % | **99 %** | 0.32 s | 0.75 s | 99 % | −0.1° |
| face-up | 100 % | **93 %** | 0.62 s | 0.84 s | 93 % | 0.0° |
| on side | 100 % | **93 %** | 0.66 s | 1.04 s | 93 % | −0.1° |

| push | standing: fell | recovered of fallen | median | walking: fell | recovered of fallen | median |
|---|---|---|---|---|---|---|
| 0.6 m/s | 20 % | 92 % | 0.84 s | 30 % | 95 % | 1.06 s |
| 0.8 m/s | 79 % | 88 % | 1.02 s | 77 % | 87 % | 1.08 s |
| 1.0 m/s | 93 % | 87 % | 1.06 s | 95 % | 84 % | 1.06 s |
| 1.2 m/s | 100 % | 76 % | 1.04 s | 97 % | 86 % | 1.12 s |

| command | `5750`: achieved | `5999`: achieved | falls |
|---|---|---|---|
| walk 0.15 / 0.30 / 0.45 m/s | 0.09 / 0.18 / 0.25 m/s | 0.10 / 0.21 / 0.30 m/s | 0 % |
| yaw drift while walking 0.30 / 0.45 | −0.04 / −0.04 rad/s | −0.07 / −0.16 rad/s | |
| turn in place +0.3 / +0.6 / +1.0 / −1.0 rad/s | 0.02 / **0.46** / 0.91 / −1.00 | 0.02 / **0.70** / 1.13 / −0.87 | 0 % |

Body-pose commands on `5750`: crouch −30 mm → −0.4 mm, pitch ±10° → +3.7 / −2.5°, ±20° → +4.2 /
−2.0°, roll ±10° → +1.9 / −1.3° — v2 does not track them, as the single-duck probe said.

Reading, and two corrections to earlier sections:

1. **Get-ups are sub-second** from every spawn type, straight at the end (pitch ≈ 0°); face-up and
   on-side fail 7 % of the time within 8 s. The push-recovery numbers (76–95 % of falls, median
   ≈ 1 s) are the ones to compare with the upstream table's "get-up after push fall 93 / 95 %" at 1.0 /
   1.2 m/s — ours are lower at 1.0 m/s standing (87 %) and clearly lower at 1.2 m/s standing (76 %).
   That is the honest baseline the next rounds have to beat.
2. **The viewer under-reports turning.** In the batched environment both v2 checkpoints turn at
   0.6 rad/s (0.46 / 0.70) where the viewer probe gave ≈ 0; the 0.3 rad/s dead zone (0.02) is real in
   both. So "VelStand lost the 0.6 turn" in the turning section below was a viewer artefact; the
   turning round is still justified by the 0.3 dead zone and the 1.0 overshoot / asymmetry
   (1.13 / −0.87 on 5999).
3. **The viewer's yaw-drift lottery shrinks in the batched environment**: 5999 drifts −0.07 rad/s
   over 256 randomised ducks, not +0.47. The per-checkpoint sweep stays useful, but it should be done
   with the battery's tracking trial, not the viewer; pipeline part 4 still uses the viewer sweep
   (it was launched before this), so its picks will be re-checked with the battery afterwards.

## Official policies measured with the same battery (2026-10-11 21:30)

Downloaded from the Hugging Face repo `pollen-robotics/microduck-policies` (main, 2026-10-11) into
`pretrained/official/`; `manifest.json` says `velstand.onnx` is the deployed v7 (`j4i6yoq2` @ 1250,
trained on `Mjlab-VelStand-Rough-Backlash-MicroDuck` at commit `53fb7d1`):

| file | sha256 |
|---|---|
| `alpha_walking.onnx` | `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c` |
| `alpha_stand.onnx` | `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b` |
| `velstand.onnx` | `ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a` |

`eval_battery.py` takes ONNX files directly (CPU onnxruntime, one call per env); on our own
`velstand_v2_5750.onnx` it reproduces the `.pt` numbers (0.30 → 0.18 m/s, ±1.0 rad/s → 0.93 /
−1.00), so the comparison below is the same code path for both sides.

**Walking policies**, `Mjlab-Velocity-Flat-MicroDuck`, 256 envs, seed 0, mean over the last 5 s of 8:

| command | official `alpha_walking` (per-env \|yaw\| p50 / p90) | ours `first_walk_10000` (p50 / p90) |
|---|---|---|
| walk 0.15 m/s | 0.00 m/s (does not step) | 0.10 m/s |
| walk 0.30 m/s | 0.15 m/s, drift 0.00 (0.04 / 0.10) | 0.20 m/s, drift −0.01 (0.06 / 0.17) |
| walk 0.45 m/s | 0.21 m/s, drift −0.02 (0.04 / 0.11) | 0.27 m/s, drift −0.05 (0.08 / 0.23) |
| turn +0.3 rad/s | 0.00 | 0.05 (0.00 / 0.26) |
| turn +0.6 rad/s | 0.02 (0.00 / 0.00) | **0.57** (0.58 / 0.65) |
| turn +1.0 / −1.0 rad/s | 0.35 / −0.17 | **1.00 / −1.00** |
| falls | 0 % | 0 % |

Reading: same recipe, 10 000 iterations here against 3 750 for the official checkpoint. The official
walk has a speed dead zone at 0.15 m/s and the turning dead zone its authors describe (0.6 → 0.02,
1.0 → 0.35, asymmetric); ours tracks turns from 0.6 rad/s up, symmetrically, and runs 30–50 % faster
at the same command — though both are well under the commanded speed, which is the speed round's
job. Our drift is slightly larger (p90 0.17–0.23 vs 0.10–0.11). Caveat: the official checkpoint was
trained at an older commit, so the environment's domain randomisation may differ a little from what
it saw.

**Walk + get-up policies**, `Mjlab-VelStand-Flat-MicroDuck`, 256 envs, seed 0 — official `velstand.onnx`
(deployed v7, `j4i6yoq2` @ 1250, trained on Rough-Backlash) against our v2 `model_5750` (Flat):

| | official v7 | ours v2 `5750` |
|---|---|---|
| get-up face-down: recovered, median, p90 | 100 %, 0.36 s, 0.54 s | 99 %, 0.32 s, 0.75 s |
| get-up face-up | 93 %, 0.72 s, 0.94 s | 93 %, 0.62 s, 0.84 s |
| get-up on side | 95 %, 0.74 s, 1.16 s | 93 %, 0.66 s, 1.04 s |
| push 1.0 m/s standing: fell / recovered of fallen / median | 98 % / **92 %** / 1.16 s | 93 % / 87 % / 1.06 s |
| push 1.2 m/s standing | 100 % / **92 %** / 1.18 s | 100 % / 76 % / 1.04 s |
| push 1.0 m/s walking | 94 % / **94 %** / 1.06 s | 95 % / 84 % / 1.06 s |
| push 1.2 m/s walking | 95 % / **93 %** / 1.08 s | 97 % / 86 % / 1.12 s |
| walk 0.15 / 0.30 / 0.45 m/s | 0.02 / 0.14 / 0.22 m/s | **0.09 / 0.18 / 0.25** m/s |
| yaw drift at 0.30 / 0.45 (per-env p90) | +0.05 / −0.03 (0.15 / 0.17) | −0.04 / −0.04 (0.17 / 0.19) |
| turn +0.3 / +0.6 / +1.0 / −1.0 rad/s | 0.00 / 0.05 / 0.39 / −0.34 | 0.02 / **0.46 / 0.91 / −1.00** |
| crouch −30 mm / raise +20 mm | −0.6 / +0.3 mm | −0.4 / +1.1 mm |
| roll +10 / −10° | **+7.8 / −7.9°** | +1.9 / −1.3° |
| pitch +10 / −10 / +20 / −20° | **+9.9 / −6.2 / +16.9 / −15.4°** | +3.7 / −2.5 / +4.2 / −2.0° |

Reading: get-ups are a tie (ours a little faster, theirs a little more reliable on the side spawn);
the official policy recovers clearly more push falls — 92–94 % against our 76–87 %, the biggest
gap at 1.2 m/s standing — and it tracks body tilt (it is distilled from `alpha_stand`, and even
extrapolates to ±20° beyond its ±12° training range), which v2 does not; ours walks faster at every
command and turns at 0.6 rad/s and above where theirs barely moves. Neither crouches. The official
push-recovery numbers here (92–94 %) are close to the 93–95 % in their own notes, which says the
battery is measuring the same thing they measured. Two gaps to close, in the order already planned:
body commands (v4, from a teacher that crouches) and push recovery (the protective-fall round, where
"recovered of fallen at 1.2 m/s standing" is the number to move from 76 % toward 92 %).


## Turning: where the bottleneck is, and what ours does (2026-10-11 15:20)

Upstream's own notes (in the VelStand config and `docs/velstand_policy.md`) call the deployed policy
"still very bad at turning": in place, 0.3 / 0.6 rad/s commands give ≈ 0, 1.0 gives 0.45. Their
diagnosis, which the numbers below confirm for our policies too:

1. **The reward makes small turns not worth it.** `track_angular_velocity` uses std √0.5 ≈ 0.71 rad/s,
   so ignoring a 0.3 rad/s command costs only 16 % of that term, while stepping costs smoothness,
   foot and effort penalties. Their counterfactual scoring showed standing still *is* the reward
   optimum at 0.3 rad/s and turning wins by only 4 % at 0.6.
2. **Slow in-place turns are never sampled.** The turn-in-place bucket draws |ω_z| from 40 % of the
   range upward (`turn_in_place_min_frac = 0.4`), so nothing below 0.4 rad/s is trained in place.
3. **The yaw error is noisy.** mjlab's term uses the full body-frame z rate, which during stepping
   carries roll/pitch coupling, so a small commanded yaw is buried.
4. **VelStand copies the walk teacher's dead zone.** The walk anchor (behaviour cloning on upright
   frames) is the floor; both of their attempts to sharpen the yaw reward *and* take turn frames off
   the anchor lost turning within 250 iterations, so they ship a runtime command remap instead
   (fed = sign(c)·(0.33 + |c|/0.6)).
5. **Yaw drift is a checkpoint lottery.** Their checkpoint 500 turns right when asked to walk straight
   (−0.20 rad/s), 1499 turns left (+0.27), 1250 neither — "always run the walk sweep".

Ours, measured in the viewer with `scripts/turn_probe.sh` (initial command via `--ang-vel-z` /
`--lin-vel-x`, body-frame yaw rate from the `--debug` gyro line, mean over the last 6 s of 12):

| command | upstream v7 `j4i6yoq2@1250` | our walk `first_walk_10000` | our `velstand_v2_5999` |
|---|---|---|---|
| turn in place +0.3 rad/s | 0.00 | 0.00 | 0.00 |
| turn in place +0.6 rad/s | 0.07 | **+0.70** | 0.03 |
| turn in place +1.0 / −1.0 rad/s | 0.45 / −0.37 | **+0.83 / −1.00** | +1.24 / −0.79 |
| straight walk 0.30 m/s: speed, yaw drift | 0.35 m/s, 0.05 rad/s | 0.19 m/s, −0.05 | 0.17 m/s, **+0.47** |
| straight walk 0.45 m/s: speed, yaw drift | (0.4 m/s: 0.03) | 0.27 m/s, −0.17 | 0.27 m/s, +0.24 |

Two things stand out. Our walk teacher turns better than upstream's walk policy at 0.6 and 1.0
rad/s (it was trained 10 000 iterations against their ≈ 3 750), but has the same dead zone at 0.3.
And VelStand v2's *last* checkpoint lost the 0.6 turn and walks in a left-hand circle at 0.30 m/s
(+0.47 rad/s ≈ 27°/s): the push tests above were done while it was veering. The sweep over the
last five checkpoints (`scripts/turn_sweep.sh`):

| v2 checkpoint | yaw drift at 0.30 m/s | turn in place 0.6 rad/s |
|---|---|---|
| 5000 | +0.09 | 0.00 |
| 5250 | −0.34 | 0.00 |
| 5500 | −0.11 | −0.01 |
| **5750** | **+0.02** | 0.00 |
| 5999 | +0.47 | 0.03 |

Same lottery as upstream's. So: **`model_5750` is the v2 checkpoint to use**, and the sweep becomes a
fixed step after every VelStand run instead of "take the last checkpoint". The 0.6 rad/s dead zone,
however, is in every v2 checkpoint — the anchor to a teacher that *can* turn at 0.6 did not carry it
over, which says the in-place turn frames are too rare in VelStand's command mix to matter.

Plan for turning, in order of leverage: the dead zone starts in the walk teacher, and ours is our
own, so fix it there (Velocity task, warm start from `first_walk_10000`): an additive yaw-only fine
term (std 0.35, weight 2 — additive, not a replacement, which is the variant that lost turning
upstream), `turn_in_place_min_frac` 0.4 → 0.1 so slow turns are sampled, and a sweep over every
250-iteration checkpoint for drift. Then VelStand warm-started from and anchored to that teacher
inherits the turning, with the in-place fraction raised so the anchor actually sees those frames.
Numbers to beat: 0.3 / 0.6 / 1.0 rad/s in place, drift < 0.05 rad/s at 0.30 and 0.45 m/s.

Queued (2026-10-11 15:28) as pipeline part 4, [`scripts/pipeline_round3_turning.sh`](../scripts/pipeline_round3_turning.sh),
after the body-pose round: [`patches/turn.patch`](../patches/turn.patch) adds `MICRODUCK_TURN=1`
to the walk recipe (turn-in-place bucket 25 % of envs sampled from 10 % of the range, plus
`track_yaw_rate_fine` std 0.35 / weight 2.0 next to the stock term — the additive variant);
smoke-tested on both tasks. Steps: sweep the v4 checkpoints → walk teacher v2 (3 000 iterations,
warm start from `first_walk_10000`) → sweep its checkpoints for drift and 0.3 / 0.6 / 1.0 rad/s →
VelStand v5 warm-started from the v4 pick with the new walk anchor and the same body-wide teacher →
sweep → push tests and all three probes on the pick. Checkpoints are now chosen by score
(turn 0.3 + turn 0.6 + ½ turn 1.0 − 2 |drift|), never "the last one".

## Next

1. ~~StandUp v2 → posture probe: does it end at pitch ≈ 0° after a push?~~ Yes (0.2°).
2. ~~Teacher v2 continued to 7 498 → probe + push test.~~ Done: upright 8/8, down only 1/8.
   ~~Teacher v3 (body-wide): straight + tracks crouch / tilt?~~ Yes: −29 mm, ±20°, ±10° roll.
3. ~~VelStand v2 (Flat, 6 000) with the new teacher; posture probe + push tests; compare with v1.~~ Done: stands up straight.
4. ~~VelStand v3 (Rough-Backlash, 3 000) warm-started from v2; push tests.~~ Done, pick 2500.
5. Body-pose round (pipeline part 3, queued): teacher v3 + VelStand v4 with crouch / tilt that
   actually track; then a protective-fall round (tuck the head, land softly) in the teacher.
6. Turning round (pipeline part 4, queued): walk teacher v2 with the yaw fixes, sweep, VelStand v5 anchored to it.
7. ~~Fixed-seed, batched evaluation battery~~ Done (first version; impact forces and ONNX input still to add) — the regression test for every later round.
8. ~~The official walk ONNX measured with the same probes (apples to apples).~~ Done for the walk and the walk+get-up policies; `alpha_stand.onnx` vs teacher v3 next.
9. GitHub Release per round: ONNX files + config + sha256.
10. Speed round (walk teacher v3: fine linear-velocity term, more high-speed commands) and the
    protective-fall rewards (head impact), distilled together into VelStand v6, judged by separate metrics.
11. Body-pose commands while walking (lift the standing gate).
12. Fillers for idle GPU time: ablation of the stand-straight fix; episodic tricks (one-leg stand,
    forward roll, bow).
