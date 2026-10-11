# Pitfalls and fixes

Everything below actually happened on this machine (RTX 5090 + Intel iGPU, Windows 11, WSL2
Ubuntu 24.04, WSLg 1.0.73). Newest first within each section.

## Viewer / rendering (WSLg)

### Black MuJoCo window, then `D3D12: Removing Device.`

**Symptom.** `GALLIUM_DRIVER=d3d12 uv run scripts/infer_policy.py ...` opens a window that stays
black; a few seconds later the process dies with `D3D12: Removing Device.` while the simulation
loop (the `[vel 1s avg] ...` lines) was running fine.

**Cause.** The machine has two GPUs (Intel iGPU + RTX 5090). Mesa's D3D12 Gallium driver inside
WSLg picked the **Intel** adapter. Verified with a tiny GLFW probe:

```
GALLIUM_DRIVER=d3d12 -> renderer: D3D12 (Intel(R) Graphics)
```

**Fix.** Tell Mesa which adapter to use:

```bash
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12 uv run scripts/infer_policy.py ...
# -> renderer: D3D12 (NVIDIA GeForce RTX 5090)
```

Fallback that always works (software rendering, fine for a single duck): just omit
`GALLIUM_DRIVER` — Mesa falls back to `llvmpipe`.

### Window exists (penguin icon in the taskbar) but shows nothing; title starts with `[WARN:COPY MODE]`

**Symptom.** The MuJoCo window is listed by Windows (visible, not minimised), but clicking it shows
nothing. The window title is `[WARN:COPY MODE] MuJoCo : scene (Ubuntu-24.04)`.

**Cause.** Not the GPU. WSLg's compositor (Weston, in the hidden system distro) shares window
pixels with Windows through a virtiofs mount at `/mnt/shared_memory`. On this machine that mount
was missing after every Windows boot, so Weston logged

```
rdp_allocate_shared_memory: Failed to open "/mnt/shared_memory/{...}" with error: Input/output error
```

at start-up (`/mnt/wslg/weston.log`) and fell back to "copy mode" (RDP bitmap copies), in which
GPU-rendered windows stay blank. Restarting the Windows-side RDP client (`msrdc.exe`) does not
help: the allocation is only attempted when Weston starts. Known upstream bug:
[microsoft/WSL#40618](https://github.com/microsoft/WSL/issues/40618) (WSL 2.7.x; seen here on
2.7.13 / WSLg 1.0.73).

**Fix without restarting WSL** (training keeps running). Create the mount point in the distro — WSL
then mounts the virtiofs share itself — and restart only Weston; WSLGd respawns it:

```powershell
wsl -d Ubuntu-24.04 -u root -- mkdir -p /mnt/shared_memory
wsl --system -u root -- sh -c 'kill $(pidof weston)'
```

Scripted, with verification: [`scripts/wslg_copymode.ps1`](../scripts/wslg_copymode.ps1) `-Check`
(read-only diagnosis, exit 1 = copy mode) and `-Fix` (the steps above, then confirms the new Weston
did not log the failure). Relaunch the viewer afterwards (every GUI window dies with Weston). Check with
`wsl --system -- mount | grep shared_memory` (must say `virtiofs`) and the window title, which
loses the `[WARN:COPY MODE]` prefix.

**Durable fix.** The `mkdir` persists on the distro's disk; after it, the next `wsl --shutdown` or
Windows reboot comes up in the fast path directly. Before that was known, the only fix was
`wsl --shutdown`, which also kills any training running in the distro.

### Arrow keys do nothing in the viewer

`infer_policy.py` reads keys from the **terminal's stdin** (cbreak mode), not from the MuJoCo
window. Two separate causes were hit:

1. The terminal window was minimised, so keys went to the MuJoCo window, which ignores them.
2. Some embedded terminal panels (e.g. an IDE/assistant terminal) swallow arrow keys. Running the
   command in **Windows Terminal** (`wt.exe`) fixed it.

If keys arrive, the terminal prints `Vel cmd: [0.15, 0.00, 0.00] [walking]`.

A robust way to drive the viewer without caring about window focus at all: run it inside
`tmux` and inject keys from another shell —

```bash
tmux new-session -d -s duck "cd ~/microduck_rl && MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12 uv run scripts/infer_policy.py --walking walk.onnx --new-cmd-obs"
tmux send-keys -t duck Up        # walk
tmux send-keys -t duck p         # random push
tmux capture-pane -p -t duck | tail -3   # read the status lines
```

### `T` (pause inference) makes the duck fall

`T` freezes the motors at their last target. Mid-stride that is not a balanced pose, so the duck
tips over within a second. Pausing is fine for a *standing* duck; for a walking snapshot, take
screenshots while it walks instead.

### The viewer camera is static

`infer_policy.py` does not track the robot and mouse-wheel zoom had no effect through WSLg, so a
duck commanded forward leaves the frame after a few metres. Use a low command (one `Up` =
0.15 m/s) and take the pictures early.

## Observation contract

### `Got invalid dimensions for input: obs ... Got: 51 Expected: 61`

The policy family uses a 61-D observation (48 proprio + 13-D command block: twist 3, head pose 4,
body pose 6). `infer_policy.py` defaults to the old 51-D layout. Add `--new-cmd-obs`:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --new-cmd-obs
```

## Training / tooling

### `No API key configured. Use wandb login` when starting `Mjlab-VelStand-*`

VelStand distils a stand-up expert and a walk anchor from Pollen's wandb project
(`pollen-robotics/mjlab_microduck`, runs `69u48n8l` and `441tzs6d`). After `wandb login` the
request still fails with **HTTP 403** — the project is private. The Hugging Face Hub only hosts the
exported ONNX files, not the `.pt` checkpoints the distillation code needs. See
[velstand-without-private-teachers.md](velstand-without-private-teachers.md).

### Exporting a VelStand checkpoint fails with wandb 403 even though training worked

`scripts/export.py` constructs the full runner, and VelStand's PPO subclass resolves its two BC
teachers in the constructor — so the export needs the same `MICRODUCK_STAND_EXPERT` /
`MICRODUCK_WALK_ANCHOR` variables as training, or it goes to wandb and gets the 403.

### Push test showed identical `trunk_z` for every push

The first version of the push-test script ran the viewer as `infer_policy.py ... | tee log` inside
tmux. With the pipe in place the injected keys never reached the policy (and Python's stdout became
block-buffered). Run the viewer without a pipeline and capture the pane with
`tmux pipe-pane -o "cat >> log"` instead; always check that at least one push changed `trunk_z`.

### Walking policy falls and never gets up

Expected: `Mjlab-Velocity-*` has no fall-recovery curriculum. Fall recovery is the `VelStand`
family (walk + protective fall + get-up) or the separate `StandUp` / `Roulade` skills. The CPU
viewer has no reset key; quit with `Q` and relaunch.

### `uv run train --help` grep returned nothing in a script

Harmless; the flags are `--env.scene.num-envs N` and `--agent.max-iterations N` (tyro naming).

### Running WSL commands from Git Bash mangles paths

`wsl.exe -- bash /mnt/c/...` from Git Bash gets `/mnt/c` rewritten to a Windows path (MSYS path
conversion). Call `wsl.exe` from PowerShell, or set `MSYS_NO_PATHCONV=1`.

## Batched viewer (`uv run play`)

### Viser viewer crashes on start: `add_slider ... assert max >= value >= min`

`uv run play Mjlab-StandUp-Flat-MicroDuck --checkpoint-file ... --num-envs 64 --viewer viser`
(mjlab 1.3.0) starts the Viser server and then dies in `_gui_api.py:add_slider` with an
`AssertionError`. The native viewer works: `--viewer native` (with the WSLg env vars above) shows
all 64 environments in one MuJoCo window — that is where the README picture comes from.

### `http://localhost:8080` shows `Cannot GET /` instead of Viser

Something on the Windows side already owns `127.0.0.1:8080` — here it was **NVIDIA Broadcast** —
so WSL2's localhost forwarding never kicks in. Use the WSL IP directly (`hostname -I` inside WSL,
e.g. `http://172.29.134.39:8080`), or free the port.

### Mouse input does not reach the WSLg window through synthetic events

Scroll/drag sent by automation tools (RDP-hosted WSLg window) did not move the MuJoCo camera at
all; a real mouse does. For scripted screenshots, pick the camera in code or live with the default.
