# my-microduck-journal

**A journal of re-training the Microduck's skills from scratch on a consumer PC** — one RTX
50-series card under WSL2 on Windows 11 — with my own policies and without the private teacher checkpoints the official recipe depends on.
Every step, number and dead end is written down, so that anyone with a single consumer GPU can
reproduce the pipeline. The exact card and the times it gave are in the hardware table below; a
16 GB card runs the same recipe with half the parallel environments and proportionally more time.

## Why train your own

The skills that come with the robot are finished ONNX files, and the viewer can even chain two of
them (walk while a velocity is commanded, stand up when it is zero). That is enough to *watch* a duck
get up. It is not enough to *change* anything:

- **The shipped policies are end points, not starting points.** The basic skills (walking,
  stand-up) train from scratch with the public recipe. The *composite* ones do not: VelStand (walk +
  get-up in one network) is distilled from, and warm-started on, teacher checkpoints that live in a
  private wandb project, and the official training command stops with an HTTP 403 without them. So
  the first job here was to rebuild those teachers; after that, composite skills can be trained on
  this PC too.
- **What you don't like, you can fix.** The first fall-recovery policy trained here "stood up" into
  a 34° forward lean. Owning the teacher meant changing its reward
  ([`patches/standup-stand-straight.patch`](patches/standup-stand-straight.patch)) and re-training,
  instead of living with it.
- **Training budget becomes a choice.** The stand-up teacher here has 13 500 iterations against the
  ≈ 10 000 of the upstream checkpoint it replaces; VelStand v2 gets 6 000 on top of a
  10 000-iteration walk, then 3 000 more on rough terrain with actuator backlash. On one consumer
  GPU each of those is a few hours.
- **One network instead of a hand-off.** Chaining walk + stand-up in the viewer switches networks in
  the middle of a fall. VelStand learns both in one network, which is what runs on the real robot,
  and training it is exactly what needs the teachers above.
- **Reproducible from the record.** Offline wandb, the exact scripts in `scripts/`, every number
  and dead end in `docs/`, so the next person needs neither the private project nor a workstation.
  (Still missing: the checkpoints themselves and a fixed-seed evaluation — see Status.)

What is different from the official setup:

| | Official (upstream) | Here |
|---|---|---|
| Hardware | Linux workstations (COACH2, RTX 5090) | Windows 11 + WSL2 Ubuntu 24.04, RTX 5090 + Intel iGPU (hence the WSLg pitfalls) |
| Walking policy | `alpha_walking` from wandb `441tzs6d` (3 750 iterations) | `first-walk`, trained here from scratch, 4096 envs × 10 000 iterations — faster and turns in place where the official one does not (table below) |
| Fall recovery | VelStand distilled from a **private** stand-up expert (`69u48n8l`) | own `StandUp` teacher (13 500 iterations) + VelStand with local checkpoints — v1 got back to standing height after every fall but into a 34° lean; v2 (teacher with command-independent upright rewards) gets up straight, 5/5 falls while walking in the 8-push smoke test |
| Body-pose control | tilt from the shipped stand expert, crouch "not learned" | teacher rewards made command-relative, crouch taught by the own teacher (in progress) |
| Tooling | wandb online, internal eval scripts | `WANDB_MODE=offline`, everything reproducible from this repo |

![64 Microducks in the mjlab StandUp environment, iteration-500 checkpoint](results/standup-500-64envs.jpg)

*64 of the 4096 parallel `Mjlab-StandUp-Flat-MicroDuck` environments, replayed with the
iteration-500 checkpoint in mjlab's native viewer (`uv run play ... --num-envs 64 --viewer native`)
while the same GPU keeps training. Each duck starts an episode on the ground and has to get up.*

## Status (2026-10-11)

✅ = verified in simulation here · 🔄 = training / being tested.

| Milestone | State | Notes |
|---|---|---|
| RTX 5090 (Blackwell, `sm_120`) runs the official `microduck_rl` stack | ✅ verified | `warp-lang 1.12.0` / `mjlab 1.3.0` / `torch 2.9.1+cu128`, WSL2 Ubuntu 24.04 — see [docs/setup-wsl2-rtx5090.md](docs/setup-wsl2-rtx5090.md) |
| Walking policy (`Mjlab-Velocity-Flat-MicroDuck`) | ✅ trained | 4096 envs × 10 000 iterations, all 41 checkpoints exported to ONNX — see [docs/training-log.md](docs/training-log.md) |
| Fall recovery (`Mjlab-VelStand-*`) | ✅ trained | The official recipe distils a stand-up expert from a **private** wandb project (HTTP 403), so a `StandUp` teacher was trained here first (6 000 iterations, 2 h 30) and VelStand warm-started from the walk policy with local teachers (3 000 iterations, 1 h 35). 8-push smoke test (1 m/s, random direction, 6 s window): v1 went **down 1/8 standing, 5/8 walking** and was back above 100 mm trunk height **every time it went down** (the walk policy alone stays down 8/8) — but v1 "stands" pitched 34° forward with the head tucked (the teacher's lean basin). Recovery took 1–3 s in the posture probe. Not yet done: a fixed-seed battery per spawn type (face-down / face-up / side) with recovery time and final posture. Teacher v2 with stronger upright rewards (13 500 iterations in total) ends every get-up at pitch ≈ 0°. **VelStand v2** (6 000 iterations with that teacher): walking pushes down 5/8, back up 5/5 within ≈ 1 s **at pitch 1–3°**, standing pushes down 0/8. Checkpoint to use: `model_5750`, picked by a walk sweep (yaw drift 0.02 rad/s at 0.30 m/s; the last checkpoint, 5999, veers at 0.47 — see the turning section of the log). v3 continues it on rough terrain with actuator backlash (3 000 iterations; on flat: walking pushes down 6/8, up 6/6, pick `model_2500`) — see [docs/velstand-without-private-teachers.md](docs/velstand-without-private-teachers.md) and [docs/training-log.md](docs/training-log.md) |
| Body-pose commands (crouch / tilt while standing) | 🔄 in progress | Measured first: neither the walk+get-up policy nor the teacher crouched, and tilt tracked only to ~10°. With the teacher's fixed-pose rewards made command-relative ([`patches/body-wide.patch`](patches/body-wide.patch)), **teacher v3 crouches 29 mm and tilts ±20° / rolls ±10° on command while still standing straight**; VelStand v4 is being distilled from it — see [docs/training-log.md](docs/training-log.md#body-pose-commands-what-the-policies-actually-do-with-them-2026-10-11-1405) |
| Evaluation battery (256 envs, fixed seed) | ✅ first statistics | VelStand v2: get-up 93–99 % from face-down / face-up / on-side, median 0.3–0.7 s, straight at the end; push falls recovered 76–95 % (0.6–1.2 m/s); 0.30 m/s command → 0.18 m/s; in-place turn 0.3 / 0.6 / 1.0 rad/s → 0.02 / 0.46 / 0.91 — see [docs/training-log.md](docs/training-log.md#evaluation-battery-first-statistics-2026-10-11-2110) |

## Measured against the official policies

The official ONNX policies (Hugging Face `pollen-robotics/microduck-policies`) run through the same
batched battery as ours: identical environment, 256 randomised ducks, seed 0, 8 s per command.
Walking policies first (`Mjlab-Velocity-Flat-MicroDuck` environment):

| command | official `alpha_walking` | ours `first_walk_10000` |
|---|---|---|
| walk 0.15 / 0.30 / 0.45 m/s → achieved | 0.00 / 0.15 / 0.21 m/s | **0.10 / 0.20 / 0.27 m/s** |
| turn in place 0.3 / 0.6 / 1.0 / −1.0 rad/s → achieved | 0.00 / 0.02 / 0.35 / −0.17 | 0.05 / **0.57 / 1.00 / −1.00** |
| falls during the trial | 0 % | 0 % |

- Same recipe, more training: the walk recipe is unchanged here; ours ran 10 000 iterations against
  the official checkpoint's 3 750. The official policy does not step at all for a 0.15 m/s command
  and has the in-place turning dead zone its own notes describe; ours tracks 0.6 rad/s and above,
  symmetrically. Both under-track speed (0.30 → 0.20 / 0.15) and both ignore 0.3 rad/s.
- Caveat: the official checkpoint comes from an older commit of the recipe, so the environment's
  randomisation may differ slightly from what it was trained in.

Walk + get-up policies (`Mjlab-VelStand-Flat-MicroDuck` environment; the official file is their
deployed v7, ours is v2 `model_5750`):

| | official `velstand.onnx` | ours `velstand v2` |
|---|---|---|
| get-up from face-down / face-up / on-side | 100 / 93 / 95 %, median 0.36 / 0.72 / 0.74 s | 99 / 93 / 93 %, median 0.32 / 0.62 / 0.66 s |
| recovered after a 1.0 / 1.2 m/s push, standing | **92 / 92 %** | 87 / 76 % |
| recovered after a 1.0 / 1.2 m/s push, walking | **94 / 93 %** | 84 / 86 % |
| walk 0.15 / 0.30 / 0.45 m/s → achieved | 0.02 / 0.14 / 0.22 m/s | **0.09 / 0.18 / 0.25 m/s** |
| turn in place 0.3 / 0.6 / 1.0 / −1.0 rad/s → achieved | 0.00 / 0.05 / 0.39 / −0.34 | 0.02 / **0.46 / 0.91 / −1.00** |
| body tilt command ±10° → achieved | **+9.9 / −6.2°** pitch, ±7.8° roll | +3.7 / −2.5° pitch, ±1.6° roll |
| crouch −30 mm → achieved | −0.6 mm | −0.4 mm |

Stand-up teachers (`Mjlab-StandUp-Flat-MicroDuck` environment; these are the policies the walk +
get-up networks are distilled from):

| | official `alpha_stand` | ours teacher v3 |
|---|---|---|
| get-up from face-down / face-up / on-side | 100 / 100 / 100 %, median 0.38 / 0.50 / 0.54 s | 100 / 100 / 100 %, median **0.28 / 0.44 / 0.50 s** |
| body tilt ±10° / ±20° → achieved | ±9.2° roll, +9.5 / −9.1°, +18.4 / −18.6° pitch | ±9.5° roll, +10.0 / −8.9°, +19.7 / −19.4° pitch |
| crouch −30 mm / raise +20 mm → achieved | +0.5 / +1.8 mm (none) | **−28.2 / +12.4 mm** |

So, measured the same way: equal at getting up, the official policy recovers more push falls
(5–16 points) and tilts on command, ours walks faster and turns where theirs does not; v2 does
not crouch and neither does any official policy, while our stand-up teacher v3 does — which is what
v4 is distilled from. The next versions target exactly the two gaps (v4: body commands from the own teacher,
which crouches; the push-recovery margin goes to the protective-fall round). Full tables with
per-env spread: [docs/training-log.md](docs/training-log.md#official-policies-measured-with-the-same-battery-2026-10-11-2130).

## Hardware & software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB, driver 617.42 (Windows) |
| CPU / RAM | 64 GB RAM, Windows 11 (build 26200) |
| WSL | WSL 2.7.13, kernel 6.18.33.2, WSLg 1.0.73, Ubuntu 24.04 LTS |
| Python stack | uv 0.12.10, Python 3.12.3, torch 2.9.1+cu128, warp-lang 1.12.0, mjlab 1.3.0, mujoco 3.10.0 |
| `microduck_rl` | `develop` @ `ceca0170` (2026-10-09); first runs were on `29e887e` (2026-09-02) |

## Headline numbers

- **Smoke test** (5 iterations × 64 envs): 42 s wall-clock, Warp reports `"cuda:0": "NVIDIA GeForce RTX 5090" (32 GiB, sm_120)`, CUDA Toolkit 12.9. No `no kernel image` errors.
- **Full walk training** (4096 envs): iteration 0 → 1000 in 30 min, 0 → 2500 in 69 min, i.e. roughly **2.5–3 h for the 6000-iteration budget** the upstream docs recommend for gaits. (Upstream README says "about 1–2 h"; a 4090 under WSL2 was reported at ~4 h 23 min.)
- **Export**: ~13 s per checkpoint (`scripts/export.py`, normalizer baked in), ONNX ≈ 775 KB each.
- **Stand-up teacher**: 6 000 iterations in 2 h 30 min, plus 7 500 stand-straight iterations in 3 h 02 min (1.30 s collection + 0.07 s learning per iteration); **VelStand** (walk + get-up): v1 3 000 iterations in 1 h 35 min, 1.70 s/iteration including the behaviour-cloning passes; v2 6 000 iterations in ≈ 3 h 20 min.

## Repository layout

```
docs/
  setup-wsl2-rtx5090.md             environment, install, smoke test
  pitfalls.md                       every problem hit so far, with the fix
  training-log.md                   one row per run: task, envs, iterations, time, outcome
  velstand-without-private-teachers.md   how to train fall recovery without the private wandb teachers
scripts/
  export_all.sh                     export every checkpoint of a run to ONNX
  run_viewer.sh / tmux_viewer.sh    CPU MuJoCo viewer with the right env vars (tmux variant takes injected keys)
  train_standup_teacher.sh          teacher v1 (6 000 iterations from scratch)
  train_standup_teacher_v2_straight.sh   teacher v2 fine-tune (stand-straight rewards)
  train_velstand_v1.sh              VelStand v1 (warm start from the walk, local teachers)
  pipeline_round1_*.sh              teacher v2 → continue → VelStand v2 → v3, with probes and push tests
  pipeline_round1b_resume_after_sleep.sh   the resume after the PC went to sleep mid-run
  pipeline_round2_body_pose.sh      teacher v3 (body-pose) → VelStand v4
  pipeline_round3_turning.sh        walk teacher v2 (yaw fixes) → sweeps → VelStand v5
  pushtest.sh                       8 random pushes in the viewer: moved / went down / upright at the end
  posture_probe.sh                  trunk height, pitch, roll and joint angles before and after pushes
  bodypose_probe.sh                 commanded vs measured crouch / tilt
  turn_probe.sh / turn_sweep.sh     in-place yaw rate and straight-walk drift, per checkpoint
  eval_battery.py / .sh             batched, fixed-seed battery: get-up per spawn type, push falls, tracking, body commands
  wslg_copymode.ps1                 check / fix the WSLg copy-mode blank window without wsl --shutdown
patches/                            config diffs against upstream microduck_rl, applied in this order:
  velstand-local-teachers.patch     MICRODUCK_STAND_EXPERT / MICRODUCK_WALK_ANCHOR = local checkpoint paths
  standup-stand-straight.patch      MICRODUCK_STANDUP_STRAIGHT=1 (teacher v2)
  body-wide.patch                   MICRODUCK_BODY_WIDE=1 (teacher v3 / VelStand v4, in progress)
  turn.patch                        MICRODUCK_TURN=1 (walk teacher v2 / VelStand v5, queued)
results/                            screenshots used in this README and the docs
```

Checkpoints (`.pt`) and exported policies (`.onnx`) are **not** in this repo yet. Plan: a GitHub
release per round with the ONNX files, their config and sha256 sums, once the body-pose round (v4) is
in; the Hub (`uv run publish`) after that.

## Quick start (what actually worked here)

```bash
# WSL2 Ubuntu 24.04, NVIDIA driver >= 580 on the Windows side
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl
git checkout ceca0170                             # the upstream commit every run below used
git apply /path/to/my-microduck-journal/patches/velstand-local-teachers.patch
git apply /path/to/my-microduck-journal/patches/standup-stand-straight.patch
git apply /path/to/my-microduck-journal/patches/body-wide.patch   # only needed for the body-pose round
git apply /path/to/my-microduck-journal/patches/turn.patch        # only needed for the turning round
UV_HTTP_TIMEOUT=600 uv sync --locked              # ~2 GB of CUDA wheels on first run

# smoke test (always first), then the real thing
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# export + drive it in the CPU viewer (see docs/pitfalls.md for the WSLg env vars)
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --checkpoint-file logs/rsl_rl/velocity/<run>/model_10000.pt --onnx-file walk.onnx
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12 uv run scripts/infer_policy.py --walking walk.onnx --new-cmd-obs
```

The full order of runs, each with the exact command, is in [docs/training-log.md](docs/training-log.md)
and the scripts above: walk (10 000 iterations) → `train_standup_teacher.sh` (6 000) →
`train_standup_teacher_v2_straight.sh` (1 500, then `pipeline_round1_*.sh` continues it by 6 000 and
trains VelStand v2 / v3) → `pipeline_round2_body_pose.sh`. Every pipeline step runs `pushtest.sh`
and `posture_probe.sh` on the result and appends to `logs/pipeline*.log`. Run names and checkpoint
numbers inside the scripts are the ones from this machine; change them to yours.

## Upstream

- Robot software: [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck)
- RL environments: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache-2.0; its `AGENTS.md` is the real training manual)
- Simulator / framework: [mujocolab/mjlab](https://github.com/mujocolab/mjlab), [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp)
- Official policies: [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) (ONNX only)

## License

Apache-2.0 for everything in this repository. Upstream `microduck_rl` code is Apache-2.0; its 3D
model files are CC BY-NC-SA 4.0 and are not redistributed here.
