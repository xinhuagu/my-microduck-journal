# Fall recovery (VelStand) without Pollen's private teachers

**Status: VelStand training started 2026-10-11 ~02:50** with the self-trained teacher
`microduck_stand/model_5999.pt` (gets up 8/8 in the push test) and `first-walk/model_10000.pt` as warm start + anchor.

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

## The plan — and the plumbing, verified (2026-10-11 00:40)

```
first-walk/model_10000.pt (ours, walks)  ── warm start + walk anchor ──┐
                                                                       ├─► VelStand
microduck_stand/model_<N>.pt (ours)      ── stand-up expert (BC) ──────┘
```

1. **Stand-up teacher from scratch** — pure PPO, no teacher needed:
   `uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 6000`
   (running; see [training-log.md](training-log.md)).
2. **Local teachers instead of wandb.** `distill._resolve_checkpoint` already prefers
   `checkpoint_path` / `anchor_checkpoint_path` over the wandb run paths; the only change needed is
   to feed them in. [`patches/velstand-local-teachers.patch`](../patches/velstand-local-teachers.patch)
   (5 lines in `microduck_velstand_env_cfg.py`) reads two environment variables:

   | variable | meaning |
   |---|---|
   | `MICRODUCK_STAND_EXPERT` | path to the stand-up teacher `model_*.pt` → `bc_cfg["checkpoint_path"]` |
   | `MICRODUCK_WALK_ANCHOR` | path to the walk checkpoint `model_*.pt` → `bc_cfg["anchor_checkpoint_path"]` |

3. **Warm start from a local run.** mjlab resolves `--agent.load-run` by regex *inside*
   `logs/rsl_rl/<experiment_name>/`, so the walk run is exposed there with a symlink:
   `ln -s ../velocity/2026-09-07_20-38-41_first-walk logs/rsl_rl/velstand/first-walk-src`.
4. **Launch** (smoke-tested with 5 iterations × 64 envs — expert loaded, anchor loaded, counters
   reset by `MICRODUCK_WARM_START=1`, iterations ran):

   ```bash
   MICRODUCK_WARM_START=1 \
   MICRODUCK_STAND_EXPERT=logs/rsl_rl/microduck_stand/<run>/model_<N>.pt \
   MICRODUCK_WALK_ANCHOR=logs/rsl_rl/velocity/2026-09-07_20-38-41_first-walk/model_10000.pt \
   uv run train Mjlab-VelStand-Flat-MicroDuck --env.scene.num-envs 4096 \
     --agent.resume True --agent.load-run first-walk-src --agent.load-checkpoint model_10000.pt \
     --agent.max-iterations 3000
   ```

   Log excerpt from the smoke test:

   ```
   [distill] expert BC ON: stand expert (iter 1500) on tilt>35.0° coef=1.0; walk anchor ON on tilt<25.0° coef=1.0
   [INFO]: Loading model checkpoint from: logs/rsl_rl/velstand/first-walk-src/model_10000.pt
   [mdp] Patch 5: WARM START from .../model_10000.pt — common_step_counter 240024 → 0, iteration → 0
   ```

5. Evaluate like upstream: push falls, get-up rate, walk sweep for yaw drift; pick the checkpoint by
   behaviour, not by reward (upstream saw checkpoints 500 and 1 499 drift in opposite yaw directions
   while the metrics looked identical).

## Alternatives considered

- **Convert `alpha_stand.onnx` back into a `.pt` teacher.** The ONNX contains the normaliser and
  the three MLP layers, so a state-dict mapping is possible in principle; not attempted yet.
- **`ENABLE_EXPERT_BC = False`.** Documented upstream as not learning to get up; not worth a run.
