# Fall recovery (VelStand) without Pollen's private teachers

**Status: in progress (2026-10-10).** This page states the problem and the plan; results will be
appended as they come in.

## The problem

`Mjlab-VelStand-*` is the policy family behind the robot's shipped `walk` slot: one network that
walks, stands still at zero command, falls protectively and gets back up. Upstream trains it by

1. **warm-starting** from a walking checkpoint (`MICRODUCK_WARM_START=1 ... --agent.resume True`),
   which keeps weights / normaliser / optimiser but resets the iteration counter so VelStand's own
   curricula start from stage 0, and
2. **distilling** a stand-up expert into the fallen frames (`ENABLE_EXPERT_BC = True`) while a
   frozen copy of the walk checkpoint anchors the upright frames.

Both teachers are fetched with `wandb.Api()` from `pollen-robotics/mjlab_microduck`
(`69u48n8l @ model_9750` = `alpha_stand`, `441tzs6d @ model_3750` = `alpha_walking`). That
project returns **HTTP 403** for an ordinary wandb account, and the Hugging Face Hub repo
`pollen-robotics/microduck-policies` publishes only `.onnx` files. `distill.py` needs `.pt`
checkpoints with an `actor_state_dict`.

Upstream's own notes say why the expert matters: without it (their "Run-2", 1 085 iterations)
the policy recorded **zero** recoveries.

## The plan

```
first-walk/model_10000.pt (ours, walks)  ── warm start + walk anchor ──┐
                                                                       ├─► VelStand
StandUp/model_<N>.pt (ours, gets up)     ── stand-up expert (BC) ──────┘
```

1. Train a stand-up teacher from scratch — pure PPO, no teacher needed:
   `uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096`
2. Point VelStand at local files. `distill._resolve_checkpoint` already prefers
   `checkpoint_path` / `anchor_checkpoint_path` over the wandb run paths, so the change is a few
   lines in `microduck_velstand_env_cfg.py` (`bc_cfg`) plus `WARM_START_RUN` → a local checkpoint.
   The diff will live in `patches/`.
3. `MICRODUCK_WARM_START=1 uv run train Mjlab-VelStand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.resume True --agent.load-checkpoint <first-walk/model_10000.pt>`
   for ~1 500 iterations (upstream's v7 used 1 500 from a walk checkpoint).
4. Evaluate like upstream: push falls at 1.0 / 1.2 m/s, get-up rate; walk sweep for yaw drift at
   0.3 / 0.4 m/s; pick the checkpoint by behaviour, not by reward (upstream saw checkpoints 500 and
   1 499 drift in opposite yaw directions while the metrics looked identical).

## Alternatives considered

- **Convert `alpha_stand.onnx` back into a `.pt` teacher.** The ONNX contains the normaliser and
  the three MLP layers, so a state-dict mapping is possible in principle; not attempted yet.
- **`ENABLE_EXPERT_BC = False`.** Documented upstream as not learning to get up; not worth a run.
