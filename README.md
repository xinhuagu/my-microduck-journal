# my-microduck-journal

**A journal of re-training the Microduck's skills from scratch on a consumer PC** — an RTX 5090
under WSL2 on Windows 11 — with my own policies and without the private teacher checkpoints the official recipe depends on.
Every step, number and dead end is written down, so that anyone with a single consumer GPU can
reproduce the pipeline.

What is different from the official setup:

| | Official (upstream) | Here |
|---|---|---|
| Hardware | Linux workstations (COACH2, RTX 5090) | Windows 11 + WSL2 Ubuntu 24.04, RTX 5090 + Intel iGPU (hence the WSLg pitfalls) |
| Walking policy | `alpha_walking` from wandb `441tzs6d` | `first-walk`, trained here from scratch, 4096 envs × 10 000 iterations |
| Fall recovery | VelStand distilled from a **private** stand-up expert (`69u48n8l`) | own `StandUp` teacher (6 000 iterations) + VelStand with local checkpoints — gets up 8/8 in the push test |
| Tooling | wandb online, internal eval scripts | `WANDB_MODE=offline`, everything reproducible from this repo |

![64 Microducks in the mjlab StandUp environment, iteration-500 checkpoint](results/standup-500-64envs.jpg)

*64 of the 4096 parallel `Mjlab-StandUp-Flat-MicroDuck` environments, replayed with the
iteration-500 checkpoint in mjlab's native viewer (`uv run play ... --num-envs 64 --viewer native`)
while the same GPU keeps training. Each duck starts an episode on the ground and has to get up.*

## Status (2026-10-10)

| Milestone | State | Notes |
|---|---|---|
| RTX 5090 (Blackwell, `sm_120`) runs the official `microduck_rl` stack | ✅ verified | `warp-lang 1.12.0` / `mjlab 1.3.0` / `torch 2.9.1+cu128`, WSL2 Ubuntu 24.04 — see [docs/setup-wsl2-rtx5090.md](docs/setup-wsl2-rtx5090.md) |
| Walking policy (`Mjlab-Velocity-Flat-MicroDuck`) | ✅ trained | 4096 envs × 10 000 iterations, all 41 checkpoints exported to ONNX — see [docs/training-log.md](docs/training-log.md) |
| Fall recovery (`Mjlab-VelStand-*`) | ✅ trained | The official recipe distils a stand-up expert from a **private** wandb project (HTTP 403), so a `StandUp` teacher was trained here first (6 000 iterations, 2 h 30) and VelStand warm-started from the walk policy with local teachers (3 000 iterations, 1 h 35). Push test: **gets up 8/8** standing and walking, where the walk policy stays down 8/8 — see [docs/velstand-without-private-teachers.md](docs/velstand-without-private-teachers.md) and [docs/training-log.md](docs/training-log.md) |

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
- **Stand-up teacher**: 6 000 iterations in 2 h 30 min; **VelStand** (walk + get-up): 3 000 iterations in 1 h 35 min, 1.70 s/iteration including the behaviour-cloning passes.

## Repository layout

```
docs/
  setup-wsl2-rtx5090.md             environment, install, smoke test
  pitfalls.md                       every problem hit so far, with the fix
  training-log.md                   one row per run: task, envs, iterations, time, outcome
  velstand-without-private-teachers.md   how to train fall recovery without the private wandb teachers
scripts/
  export_all.sh                     export every checkpoint of a run to ONNX
  run_viewer.sh                     launch the CPU MuJoCo viewer with the right env vars
patches/                            (coming) config diffs against upstream microduck_rl
results/                            (coming) reward curves, GIFs
```

Checkpoints (`.pt`) and exported policies (`.onnx`) are **not** in this repo; they will be published
on the Hugging Face Hub via `uv run publish`.

## Quick start (what actually worked here)

```bash
# WSL2 Ubuntu 24.04, NVIDIA driver >= 580 on the Windows side
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl
UV_HTTP_TIMEOUT=600 uv sync --locked              # ~2 GB of CUDA wheels on first run

# smoke test (always first), then the real thing
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# export + drive it in the CPU viewer (see docs/pitfalls.md for the WSLg env vars)
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --checkpoint-file logs/rsl_rl/velocity/<run>/model_10000.pt --onnx-file walk.onnx
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12 uv run scripts/infer_policy.py --walking walk.onnx --new-cmd-obs
```

## Upstream

- Robot software: [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck)
- RL environments: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache-2.0; its `AGENTS.md` is the real training manual)
- Simulator / framework: [mujocolab/mjlab](https://github.com/mujocolab/mjlab), [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp)
- Official policies: [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) (ONNX only)

## License

Apache-2.0 for everything in this repository. Upstream `microduck_rl` code is Apache-2.0; its 3D
model files are CC BY-NC-SA 4.0 and are not redistributed here.
