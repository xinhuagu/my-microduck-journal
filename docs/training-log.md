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
| 2026-10-10 23:57 → | **`microduck_stand`** (stand-up teacher) | `Mjlab-StandUp-Flat-MicroDuck` | 4096 | 6 000 max, **running** | 1.42 s/iteration, ETA ≈ 2 h 40 min | smoke 5/64 OK (2 000 steps/s); first look at iteration 500 below |

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

## Next

1. `Mjlab-StandUp-Flat-MicroDuck` — our own stand-up teacher (upstream's took 9 750 iterations;
   aim for 2 000–3 000 first and evaluate).
2. `Mjlab-VelStand-Flat-MicroDuck` with `MICRODUCK_WARM_START=1` from `first-walk/model_10000.pt`
   and local teacher paths — see [velstand-without-private-teachers.md](velstand-without-private-teachers.md).
