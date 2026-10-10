# Setup: microduck_rl on an RTX 5090 under WSL2

As of October 2026 the upstream docs only say "requires a CUDA GPU and uv"; mjlab's install page
lists "Linux + NVIDIA Turing or newer, driver 580+" and calls Windows/WSL "preliminary". Nothing
public confirmed Blackwell (`sm_120`) on the pinned `warp-lang 1.12.0` / `mjlab 1.3.0` stack. This
page records a working configuration.

## Host

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 607 MiB |
| Windows driver | 617.42 (the WSL side sees the same number via `/usr/lib/wsl/lib/libcuda.so`) |
| Windows | 11, build 26200 |
| RAM | 64 GB |
| WSL | 2.7.13.0, kernel 6.18.33.2-microsoft-standard-WSL2, WSLg 1.0.73, Direct3D 1.611.1 |
| Distro | Ubuntu 24.04 LTS (user `ubuntu`), WSL ext4 disk 1 TB |

## Install

```bash
# inside WSL
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv 0.12.10
git clone https://github.com/pollen-robotics/microduck_rl   # keep it on the ext4 side, NOT under /mnt/c
cd microduck_rl
UV_HTTP_TIMEOUT=600 uv sync --locked                     # ~2 GB CUDA wheels; .venv ends up at 7.9 GB
uv run list-envs | grep MicroDuck                        # 20+ tasks incl. Velocity, VelStand, StandUp, ...
```

Resolved versions in `.venv`: Python 3.12.3, `torch 2.9.1+cu128` (`torch.cuda.is_available() == True`),
`warp-lang 1.12.0`, `mjlab 1.3.0`, `mujoco 3.10.0`. `uv sync --locked` after `git pull` to
`ceca0170` (2026-10-09) was a no-op — the lock file had not changed since early September.

## Smoke test (do this before anything else)

`AGENTS.md` upstream insists on a 5-iteration / 64-env smoke test first.

```bash
WANDB_MODE=offline MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5
```

Result on this machine (2026-10-10): exit code 0, **42 s** wall-clock including module loads.
Relevant lines:

```
Warp 1.12.0 initialized:
   CUDA Toolkit 12.9, Driver 13.4
   Devices:
     "cuda:0"   : "NVIDIA GeForce RTX 5090" (32 GiB, sm_120, mempool enabled)
[INFO] Training with: device=cuda:0, seed=42, rank=0
Number of environments | 64
Learning iteration 0/5  Steps per second: 734   (first iteration includes graph capture)
Learning iteration 4/5  Steps per second: 1745
```

What the log also shows, and is worth reading once: the 61-D actor observation (48 proprio +
13 command), the 76-D critic observation, the 16 reward terms with weights, the 13 domain-randomised
fields, 7 curriculum terms, and the `(512, 256, 128)` ELU actor/critic MLPs.

## Full training

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

Checkpoints land in `logs/rsl_rl/velocity/<date>_<run>/model_<iter>.pt` every 250 iterations.
Timing from checkpoint mtimes of the `first-walk` run (4096 envs):

| iterations | wall-clock |
|---|---|
| 0 → 1000 | 30 min |
| 0 → 2500 | 69 min |
| 6000 (upstream gait budget) | ≈ 2.5–3 h (extrapolated) |

The run was paused overnight and resumed; the total of 10 000 iterations therefore has no clean
wall-clock figure.

## WSL-specific notes

- **Keep a terminal attached.** WSL shuts the VM down when no window is attached, which kills
  `nohup`/`setsid` training jobs. Leave the training terminal open.
- **Files on ext4.** Accessing a repo under `/mnt/c` from WSL is several times slower.
- **CUDA Graphs.** Older drivers only lose the "CUDA Graphs" optimisation (Datawhale reported this
  on driver 535); 617.42 had no such warning.
- Rendering (viewer) needs extra environment variables — see [pitfalls.md](pitfalls.md).
