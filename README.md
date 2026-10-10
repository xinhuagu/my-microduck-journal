# my-microduck-journal

Training a [Microduck](https://www.pollen-robotics.com/microduck) to walk and get back up with
[mjlab](https://github.com/mujocolab/mjlab) + PPO on an **RTX 5090 under WSL2** — setup notes,
pitfalls, and a training log. Everything here was run on one consumer PC; the robot itself is
still on pre-order, so all results are simulation (`microduck_rl`) until it ships.

## Status (2026-10-10)

| Milestone | State | Notes |
|---|---|---|
| RTX 5090 (Blackwell, `sm_120`) runs the official `microduck_rl` stack | ✅ verified | `warp-lang 1.12.0` / `mjlab 1.3.0` / `torch 2.9.1+cu128`, WSL2 Ubuntu 24.04 — see [docs/setup-wsl2-rtx5090.md](docs/setup-wsl2-rtx5090.md) |
| Walking policy (`Mjlab-Velocity-Flat-MicroDuck`) | ✅ trained | 4096 envs × 10 000 iterations, all 41 checkpoints exported to ONNX — see [docs/training-log.md](docs/training-log.md) |
| Fall recovery (`Mjlab-VelStand-*`) | 🚧 in progress | The official recipe distils a stand-up expert from Pollen's **private** wandb project (HTTP 403). Plan: train our own `StandUp` teacher, then VelStand with local checkpoints — see [docs/velstand-without-private-teachers.md](docs/velstand-without-private-teachers.md) |
| Deploy on the real robot | ⏳ waiting | pre-order; `sudo robotctl policy load walk <file>` once it arrives |

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

## Repository layout

```
docs/
  setup-wsl2-rtx5090.md             environment, install, smoke test
  pitfalls.md                       every problem hit so far, with the fix
  training-log.md                   one row per run: task, envs, iterations, time, outcome
  velstand-without-private-teachers.md   how to train fall recovery without Pollen's wandb
scripts/
  export_all.sh                     export every checkpoint of a run to ONNX
  run_viewer.sh                     launch the CPU MuJoCo viewer with the right env vars
patches/                            (coming) config diffs against upstream microduck_rl
results/                            (coming) reward curves, GIFs
```

Checkpoints (`.pt`) and exported policies (`.onnx`) are **not** in this repo; they will be published
on the Hugging Face Hub via `uv run publish` so that `sudo robotctl policy load` can install them.

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
