# -*- coding: utf-8 -*-
"""Fixed-seed, batched evaluation battery for Microduck walk / get-up policies (my-microduck-journal).

Runs N parallel mjlab environments headless on the GPU with the trained policy from an rsl_rl checkpoint,
forces spawn states and commands, and reports:
  A. get-up battery per spawn type (face-down / face-up / on-side): recovery rate, median time, final pitch
  B. push falls while standing and while walking (1.0 and 1.2 m/s): fall rate, recovery rate, time
  C. speed / yaw tracking: achieved vs commanded for straight walks and in-place turns, yaw drift, falls
  D. body-pose commands while standing: achieved crouch / raise / roll / pitch

usage (inside microduck_rl, teachers for VelStand must be given as env vars as for export):
  uv run python eval_battery.py --task Mjlab-VelStand-Flat-MicroDuck \
      --checkpoint logs/rsl_rl/velstand/<run>/model_5750.pt --num-envs 256 --seed 0 --out eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

STAND_Z = 0.116          # trunk height when standing (measured)
UP_Z = 0.100             # "back on its feet" height threshold
UP_TILT_DEG = 15.0       # ... and tilt threshold
DOWN_Z = 0.090           # "went down" threshold (same as the viewer push test)


# ----------------------------------------------------------------------------- env + policy
def build(task: str, ckpt: str, num_envs: int, seed: int, device: str, episode_s: float = 60.0,
          video: str | None = None, video_steps: int = 1700):
    configure_torch_backends()
    env_cfg = load_env_cfg(task, play=True)
    agent_cfg = load_rl_cfg(task)
    if video:   # close tracking camera on env 0, 720p, recorded from the first step for `video_steps` steps
        v = env_cfg.viewer
        v.distance, v.elevation, v.azimuth, v.height, v.width, v.max_extra_envs = 1.2, -20.0, 135.0, 720, 1280, 0
    # No surprise pushes during a trial: interval events stay registered (curricula may reference them) but never
    # fire; curricula are dropped (evaluation only); keep only the time-out termination.
    for v in env_cfg.events.values():
        if getattr(v, "mode", "") == "interval":
            v.interval_range_s = (1.0e6, 1.0e6)
    env_cfg.curriculum = {}
    env_cfg.terminations = {k: v for k, v in env_cfg.terminations.items() if "time_out" in k}
    env_cfg.episode_length_s = episode_s
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if video else None)
    if video:
        from mjlab.utils.wrappers import VideoRecorder
        env = VideoRecorder(env, video_folder=video, step_trigger=lambda step: step == 0,
                            video_length=video_steps, disable_logger=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if ckpt.endswith(".onnx"):
        policy = OnnxPolicy(ckpt, device)     # exported policy (normalizer baked in): also works for the official files
    else:
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
    env.seed(seed)
    torch.manual_seed(seed)
    return env, policy


class OnnxPolicy:
    """Run an exported [1,61]->[1,14] (or batched) ONNX policy on the CPU for every env."""

    def __init__(self, path: str, device: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.in_name, self.device = inp.name, device
        self.batched = not (isinstance(inp.shape[0], int) and inp.shape[0] == 1)
        if not self.batched:
            # Exported files are [1, 61]; rewrite the batch dimension to a symbol so one call serves all envs
            # (a plain MLP has no batch-dependent reshapes, so this is safe; verified with a 2-row run below).
            try:
                import numpy as np, onnx, tempfile
                m = onnx.load(path)
                for t in list(m.graph.input) + list(m.graph.output):
                    t.type.tensor_type.shape.dim[0].dim_param = "N"
                tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False); onnx.save(m, tmp.name); tmp.close()
                sess = ort.InferenceSession(tmp.name, providers=["CPUExecutionProvider"])
                x = np.zeros((2, inp.shape[1]), dtype=np.float32)
                y2 = sess.run(None, {self.in_name: x})[0]
                y1 = self.sess.run(None, {self.in_name: x[:1]})[0]
                if y2.shape[0] == 2 and np.allclose(y1[0], y2[0], atol=1e-5):
                    self.sess, self.batched = sess, True
            except Exception as e:  # keep the per-env loop
                print(f"[battery] dynamic batch not possible ({e.__class__.__name__}); using one call per env")
        print(f"[battery] ONNX policy {path}: input {inp.name} {inp.shape}, batched={self.batched}")

    def __call__(self, obs):
        x = obs["actor"] if "actor" in obs.keys() else obs["policy"]
        a = x.detach().cpu().numpy().astype("float32")
        if self.batched:
            try:
                y = self.sess.run(None, {self.in_name: a})[0]
            except Exception:
                self.batched = False
        if not self.batched:
            import numpy as np
            y = np.concatenate([self.sess.run(None, {self.in_name: a[i:i + 1]})[0] for i in range(a.shape[0])], 0)
        return torch.as_tensor(y, device=self.device)


class Duck:
    """Thin control layer over the batched env: force commands and states, step, measure."""

    def __init__(self, env, policy, seed: int):
        self.env, self.policy = env, policy
        self.u = env.unwrapped
        self.robot = self.u.scene["robot"]
        self.n = self.u.num_envs
        self.dev = self.u.device
        self.dt = self.u.step_dt
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.origins = self.u.scene.terrain.env_origins
        djp = self.robot.data.default_joint_pos
        self.default_joint_pos = (djp.expand(self.n, -1) if djp.dim() == 2 and djp.shape[0] == 1 else djp).clone()
        drs = self.robot.data.default_root_state
        self.stand_z = float(drs[0, 2]) if drs.dim() == 2 else float(drs[2])
        self.cm = self.u.command_manager

    # ---- commands
    def set_twist(self, vx: float, vy: float, wz: float, standing: bool | None = None):
        tw = self.cm.get_term("twist")
        tw.time_left[:] = 1e9
        for name in ("is_heading_env", "is_world_env"):
            if hasattr(tw, name):
                getattr(tw, name)[:] = False
        if standing is None:
            standing = (abs(vx) + abs(vy) + abs(wz)) < 1e-6
        tw.is_standing_env[:] = standing
        cmd = torch.tensor([vx, vy, wz], device=self.dev)
        tw.vel_command_b[:] = cmd
        if hasattr(tw, "vel_command_w"):
            tw.vel_command_w[:] = cmd

    def set_body(self, dz: float = 0.0, roll: float = 0.0, pitch: float = 0.0):
        if "body_pose" not in self.cm.active_terms:
            return
        b = self.cm.get_term("body_pose")
        b.time_left[:] = 1e9
        cmd = torch.tensor([0.0, 0.0, dz, roll, pitch, 0.0], device=self.dev)
        if hasattr(b, "_raw"):
            b._raw[:] = cmd
        b._command[:] = cmd

    def set_head_zero(self):
        if "head_pose" in self.cm.active_terms:
            h = self.cm.get_term("head_pose")
            h.time_left[:] = 1e9
            h._command[:] = 0.0

    # ---- states
    def _quats(self, kind: str) -> torch.Tensor:
        n = self.n
        yaw = (torch.rand(n, device=self.dev, generator=self.gen) * 2 - 1) * math.pi
        cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
        s = 2.0 ** -0.5
        if kind == "upright":
            return torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)
        if kind == "face_down":
            return torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
        if kind == "face_up":
            return torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)
        if kind == "on_side":
            sign = torch.where(torch.rand(n, device=self.dev, generator=self.gen) < 0.5, 1.0, -1.0)
            return torch.stack([s * cy, sign * s * cy, sign * s * sy, s * sy], dim=1)
        raise ValueError(kind)

    def action_for_targets(self, q_target: torch.Tensor) -> torch.Tensor:
        """Raw action that makes the joint-position action term target `q_target` (target = default + raw*scale + offset)."""
        term = self.u.action_manager.get_term("joint_pos")
        scale, offset = term.scale, term.offset
        return (q_target - self.default_joint_pos - offset) / scale

    def hold(self, seconds: float, q_target: torch.Tensor | None = None):
        """Step with the policy off: joint targets = `q_target` (default pose if None). Lets a ground spawn
        drop and settle in a sprawled posture, so the get-up clock starts from a real resting state."""
        act = torch.zeros(self.n, self.u.action_manager.total_action_dim, device=self.dev) if q_target is None \
            else self.action_for_targets(q_target)
        for _ in range(int(round(seconds / self.dt))):
            self.env.step(act)

    def spawn(self, kind: str, settle_steps: int = 0):
        """Reset every env, then overwrite the root pose / joints with the requested spawn.
        Ground spawns get a random sprawled joint pose (±0.5 rad around default, inside the soft limits),
        are dropped from 12 cm and held in that pose for 1 s (policy off) before the trial starts."""
        self.env.reset()
        z = 0.122 if kind == "upright" else 0.12   # default_root_state has z = 0 (root at the origin), so spawn above the ground
        state = torch.zeros(self.n, 13, device=self.dev)
        state[:, :3] = self.origins
        state[:, 2] += z
        state[:, 3:7] = self._quats(kind)
        self.robot.write_root_state_to_sim(state)
        q = self.default_joint_pos.clone()
        if kind != "upright":
            q = q + (torch.rand(q.shape, device=self.dev, generator=self.gen) * 2 - 1) * 0.5
            lim = self.robot.data.soft_joint_pos_limits
            q = torch.maximum(torch.minimum(q, lim[..., 1]), lim[..., 0])
        self._spawn_q = q
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.u.scene.write_data_to_sim()
        self.u.sim.forward()
        self.set_head_zero()
        self.set_body(0.0, 0.0, 0.0)
        if kind != "upright":
            self.hold(1.0, self._spawn_q)
        if os.environ.get("BATTERY_DEBUG"):
            m = self.measure()
            print(f"[spawn {kind}] z: min {float(m['z'].min()):.3f} med {float(m['z'].median()):.3f} max {float(m['z'].max()):.3f}  tilt med {float(m['tilt'].median()):.1f}")
        for _ in range(settle_steps):
            self.step()

    def push(self, speed: float):
        """Set the trunk's world xy velocity to `speed` m/s in a random direction (overwrites, like the
        viewer's `random_push` and the training event push_by_setting_velocity)."""
        ang = torch.rand(self.n, device=self.dev, generator=self.gen) * 2 * math.pi
        vel = self.robot.data.root_link_vel_w.clone()
        vel[:, 0] = speed * torch.cos(ang)
        vel[:, 1] = speed * torch.sin(ang)
        self.robot.write_root_link_velocity_to_sim(vel)
        self.u.scene.write_data_to_sim()

    # ---- stepping + measurement (the whole battery runs under torch.inference_mode(), see main)
    def step(self):
        obs = self.env.get_observations()
        act = self.policy(obs)
        self.env.step(act)

    def measure(self) -> dict[str, torch.Tensor]:
        d = self.robot.data
        z = d.root_link_pos_w[:, 2] - self.origins[:, 2]
        g = d.projected_gravity_b
        gz = torch.clamp(-g[:, 2], -1.0, 1.0)
        tilt = torch.rad2deg(torch.acos(gz))
        pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
        roll = torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2]))
        vx = d.root_link_lin_vel_b[:, 0]
        wz = d.root_link_ang_vel_b[:, 2]
        return {"z": z, "tilt": tilt, "pitch": pitch, "roll": roll, "vx": vx, "wz": wz}

    def rollout(self, seconds: float) -> dict[str, torch.Tensor]:
        """Step for `seconds`, return per-signal tensors of shape (T, N)."""
        T = int(round(seconds / self.dt))
        buf = {k: [] for k in ("z", "tilt", "pitch", "roll", "vx", "wz")}
        dbg = bool(os.environ.get("BATTERY_DEBUG"))
        for t in range(T):
            self.step()
            m = self.measure()
            for k in buf:
                buf[k].append(m[k])
            if dbg and (t % 25 == 0 or t < 5):
                print(f"  t={t*self.dt:5.2f}s z min/med/max {float(m['z'].min()):.3f}/{float(m['z'].median()):.3f}/{float(m['z'].max()):.3f}  tilt med {float(m['tilt'].median()):.1f}  vx med {float(m['vx'].median()):+.2f}  wz med {float(m['wz'].median()):+.2f}")
        return {k: torch.stack(v) for k, v in buf.items()}


# ----------------------------------------------------------------------------- analysis helpers
def up_mask(tr: dict[str, torch.Tensor]) -> torch.Tensor:
    return (tr["z"] >= UP_Z) & (tr["tilt"] <= UP_TILT_DEG)


def recovery_stats(tr: dict[str, torch.Tensor], dt: float, hold_s: float = 1.0, start: int = 0):
    """Per env: first time (s) the duck is up for `hold_s` continuously, from step `start`; NaN if never."""
    up = up_mask(tr)[start:]
    T, N = up.shape
    hold = max(1, int(round(hold_s / dt)))
    first = torch.full((N,), float("nan"), device=up.device)
    run = torch.zeros(N, dtype=torch.long, device=up.device)
    for t in range(T):
        run = torch.where(up[t], run + 1, torch.zeros_like(run))
        hit = (run == hold) & torch.isnan(first)
        first[hit] = (t - hold + 1) * dt
    return first


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def nanmedian(t: torch.Tensor) -> float:
    v = t[~torch.isnan(t)]
    return float(v.median()) if len(v) else float("nan")


# ----------------------------------------------------------------------------- trials
def trial_getup(duck: Duck, kinds=("face_down", "face_up", "on_side"), seconds: float = 8.0):
    rows = []
    for kind in kinds:
        duck.spawn(kind)
        m0 = duck.measure()
        down0 = (m0["z"] < DOWN_Z) | (m0["tilt"] > 45.0)   # only ducks that really start on the ground count
        duck.set_twist(0, 0, 0, standing=True)
        tr = duck.rollout(seconds)
        t_up = recovery_stats(tr, duck.dt)
        t_up[~down0] = float("nan")
        last = slice(-int(round(1.0 / duck.dt)), None)
        end_pitch = tr["pitch"][last].mean(0)
        end_up = up_mask(tr)[last].all(0) & down0
        n_down = max(int(down0.sum()), 1)
        ok = ~torch.isnan(t_up)
        rows.append({
            "spawn": kind, "n": duck.n, "started_down": float(down0.float().mean()),
            "recovered": float(ok.float().sum() / n_down),
            "median_time_s": nanmedian(t_up),
            "p90_time_s": float(torch.quantile(t_up[ok], 0.9)) if ok.any() else float("nan"),
            "upright_at_end": float(end_up.float().sum() / n_down),
            "end_pitch_deg_median": float(end_pitch[end_up].median()) if end_up.any() else float("nan"),
        })
    return rows


def trial_push(duck: Duck, speeds=(1.0, 1.2), modes=("standing", "walking"), seconds: float = 6.0):
    if os.environ.get("BATTERY_PUSH_SPEEDS"):
        speeds = tuple(float(x) for x in os.environ["BATTERY_PUSH_SPEEDS"].split(","))
    rows = []
    for mode in modes:
        for speed in speeds:
            duck.spawn("upright")
            if mode == "walking":
                duck.set_twist(0.30, 0, 0, standing=False)
            else:
                duck.set_twist(0, 0, 0, standing=True)
            duck.rollout(3.0)                      # settle / get walking
            duck.push(speed)
            tr = duck.rollout(seconds)
            fell = (tr["z"] < DOWN_Z).any(0)
            t_up = recovery_stats(tr, duck.dt)     # time from the push until up for 1 s
            rec = (~torch.isnan(t_up)) & fell
            # where do the non-recovered fallen ducks end up? (last 1 s): lying (z < 70 mm), low (70-100 mm), up but tilted
            fail = fell & ~rec
            last = slice(-int(round(1.0 / duck.dt)), None)
            z_end = tr["z"][last].mean(0); tilt_end = tr["tilt"][last].mean(0)
            nf = max(int(fail.sum()), 1)
            rows.append({
                "mode": mode, "push_mps": speed, "n": duck.n,
                "fell": float(fell.float().mean()),
                "recovered_of_fallen": float(rec.float().sum() / max(int(fell.sum()), 1)),
                "median_recovery_s": nanmedian(t_up[fell]) if fell.any() else float("nan"),
                "upright_at_end": float(up_mask(tr)[-1].float().mean()),
                "failed_n": int(fail.sum()),
                "failed_lying": float((fail & (z_end < 0.070)).float().sum() / nf),
                "failed_low": float((fail & (z_end >= 0.070) & (z_end < UP_Z)).float().sum() / nf),
                "failed_up_but_tilted": float((fail & (z_end >= UP_Z)).float().sum() / nf),
                "failed_z_mm_median": float(z_end[fail].median() * 1000) if fail.any() else float("nan"),
                "failed_tilt_deg_median": float(tilt_end[fail].median()) if fail.any() else float("nan"),
            })
    return rows


def trial_tracking(duck: Duck, seconds: float = 8.0):
    cases = [("walk 0.15", 0.15, 0, 0), ("walk 0.30", 0.30, 0, 0), ("walk 0.45", 0.45, 0, 0),
             ("turn +0.3", 0, 0, 0.3), ("turn +0.6", 0, 0, 0.6), ("turn +1.0", 0, 0, 1.0), ("turn -1.0", 0, 0, -1.0)]
    if os.environ.get("BATTERY_DEBUG"):
        cases = cases[1:2]
    rows = []
    for label, vx, vy, wz in cases:
        duck.spawn("upright")
        duck.set_twist(vx, vy, wz, standing=False)
        tr = duck.rollout(seconds)
        last = slice(int(round(3.0 / duck.dt)), None)   # average over the last 5 s
        warm = int(round(1.0 / duck.dt))                 # the first second is the spawn settling
        wz_env = tr["wz"][last].mean(0)                  # per-env mean yaw rate over the last 5 s
        rows.append({
            "case": label, "cmd_vx": vx, "cmd_wz": wz,
            "vx": float(tr["vx"][last].mean()), "wz": float(tr["wz"][last].mean()),
            "wz_abs_p50": float(wz_env.abs().median()), "wz_abs_p90": float(torch.quantile(wz_env.abs(), 0.9)),
            "fell": float((tr["z"][warm:] < DOWN_Z).any(0).float().mean()),
        })
    return rows


def trial_body(duck: Duck, seconds: float = 5.0):
    cases = [("crouch -30 mm", -0.03, 0, 0), ("raise +20 mm", 0.02, 0, 0),
             ("roll +10", 0, math.radians(10), 0), ("roll -10", 0, math.radians(-10), 0),
             ("pitch +10", 0, 0, math.radians(10)), ("pitch -10", 0, 0, math.radians(-10)),
             ("pitch +20", 0, 0, math.radians(20)), ("pitch -20", 0, 0, math.radians(-20))]
    rows = []
    duck.spawn("upright")
    duck.set_twist(0, 0, 0, standing=True)
    base = duck.rollout(3.0)["z"][-50:].mean(0)
    for label, dz, roll, pitch in cases:
        duck.spawn("upright")
        duck.set_twist(0, 0, 0, standing=True)
        duck.rollout(2.0)
        duck.set_body(dz, roll, pitch)
        tr = duck.rollout(seconds)
        last = slice(-int(round(2.0 / duck.dt)), None)
        rows.append({
            "case": label, "cmd_dz_mm": dz * 1000, "cmd_roll_deg": math.degrees(roll), "cmd_pitch_deg": math.degrees(pitch),
            "dz_mm": float(((tr["z"][last].mean(0) - base) * 1000).median()),
            "roll_deg": float(tr["roll"][last].mean(0).median()),
            "pitch_deg": float(tr["pitch"][last].mean(0).median()),
            # a commanded crouch legitimately goes below the 90 mm height threshold: judge falls by tilt here
            "fell": float((tr["tilt"] > 45.0).any(0).float().mean()),
        })
    return rows


def trial_bodywalk(duck: Duck, seconds: float = 6.0, vx: float = 0.20):
    """Body commands while walking at vx: crouch -20 mm, roll / pitch +-10 deg (half the standing range).
    Upstream zeroes body commands while walking; this trial lets them through for every env (the command's
    walking gate is bypassed for the trial only), so any policy can be measured. Each case is compared with
    the same walk under a zero body command (per env, mean over the last 3 s)."""
    if "body_pose" not in duck.cm.active_terms:
        return []
    b = duck.cm.get_term("body_pose")
    if not hasattr(b, "_raw"):
        return []

    def through():
        b._command[:] = b._raw
    b._gate = through   # instance attribute shadows the class method while the trial runs
    r10 = math.radians(10)
    cases = [("walk", 0, 0, 0), ("crouch -20 mm", -0.02, 0, 0), ("roll +10", 0, r10, 0), ("roll -10", 0, -r10, 0),
             ("pitch +10", 0, 0, r10), ("pitch -10", 0, 0, -r10)]
    rows, ref = [], None
    try:
        for label, dz, roll, pitch in cases:
            duck.spawn("upright")
            duck.set_twist(vx, 0, 0, standing=False)
            duck.set_body(0, 0, 0)
            duck.rollout(2.0)
            duck.set_body(dz, roll, pitch)
            tr = duck.rollout(seconds)
            last = slice(-int(round(3.0 / duck.dt)), None)
            z, r, p = tr["z"][last].mean(0), tr["roll"][last].mean(0), tr["pitch"][last].mean(0)
            if ref is None:
                ref = (z, r, p)
            rows.append({
                "case": label, "cmd_vx": vx, "cmd_dz_mm": dz * 1000, "cmd_roll_deg": math.degrees(roll), "cmd_pitch_deg": math.degrees(pitch),
                "dz_mm": float(((z - ref[0]) * 1000).median()),
                "roll_deg": float((r - ref[1]).median()),
                "pitch_deg": float((p - ref[2]).median()),
                "vx": float(tr["vx"][last].mean()),
                "fell": float((tr["tilt"] > 45.0).any(0).float().mean()),
            })
    finally:
        del b._gate
    return rows


# ----------------------------------------------------------------------------- goal reaching (navigation layer)
GOAL_DIST = 1.5          # m from the spawn point, random bearing
GOAL_TOL = 0.10          # m: "reached" when inside this for 1 s
TURN_IN_PLACE_ERR = 0.6  # rad: heading error above which the controller turns in place
TURN_RATE = 0.8          # rad/s: in-place turn command (well above the 0.3 rad/s dead zone)
WALK_V = 0.30            # m/s forward command while far from the goal


def goal_controller(dxy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Per-env (vx, vy, wz) from the goal vector in the world frame and the robot heading.
    Proportional in heading, bang-bang between turn-in-place and walking, stop inside the tolerance."""
    dist = torch.linalg.norm(dxy, dim=1)
    bearing = torch.atan2(dxy[:, 1], dxy[:, 0])
    err = torch.remainder(bearing - heading + math.pi, 2 * math.pi) - math.pi      # wrap to [-pi, pi)
    cmd = torch.zeros(dxy.shape[0], 3, device=dxy.device)
    turn = (err.abs() > TURN_IN_PLACE_ERR) | ((dist < 0.3) & (err.abs() > 0.3))
    walk = ~turn & (dist >= GOAL_TOL)
    # Turning: keep a little forward speed while far from the goal (yaw is tracked better while stepping
    # than in place, and a pure in-place turn can stall on policies with a turning dead zone).
    cmd[turn, 0] = torch.where(dist[turn] > 0.5, torch.full_like(dist[turn], 0.15), torch.zeros_like(dist[turn]))
    cmd[turn, 2] = TURN_RATE * torch.sign(err[turn])
    cmd[walk, 0] = torch.clamp(0.8 * dist[walk], 0.2, WALK_V)
    cmd[walk, 2] = torch.clamp(1.5 * err[walk], -0.8, 0.8)
    return cmd


def trial_goal(duck: Duck, seconds: float = 30.0, push_speed: float = 1.0, push_at_s: float = 2.0):
    """Walk to a goal GOAL_DIST away in a random direction; one variant with a push at push_at_s."""
    tw = duck.cm.get_term("twist")
    rows = []
    variants = (True,) if os.environ.get("BATTERY_GOAL_PUSH_ONLY") else (False, True)
    for with_push in variants:
        duck.spawn("upright")
        duck.set_twist(0, 0, 0, standing=True)
        duck.rollout(1.0)
        start = duck.robot.data.root_link_pos_w[:, :2].clone()
        ang = torch.rand(duck.n, device=duck.dev, generator=duck.gen) * 2 * math.pi
        goal = start + GOAL_DIST * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
        T = int(round(seconds / duck.dt)); hold = int(round(1.0 / duck.dt))
        dist_hist, z_hist = [], []
        for t in range(T):
            pos = duck.robot.data.root_link_pos_w[:, :2]
            heading = duck.robot.data.heading_w
            cmd = goal_controller(goal - pos, heading)
            tw.time_left[:] = 1e9
            tw.is_standing_env[:] = cmd.abs().sum(1) < 1e-6
            tw.vel_command_b[:] = cmd
            if hasattr(tw, "vel_command_w"):
                tw.vel_command_w[:] = cmd
            if with_push and t == int(round(push_at_s / duck.dt)):
                duck.push(push_speed)
            duck.step()
            m = duck.measure()
            dist_hist.append(torch.linalg.norm(goal - duck.robot.data.root_link_pos_w[:, :2], dim=1))
            z_hist.append(m["z"])
        dist = torch.stack(dist_hist); z = torch.stack(z_hist)                     # (T, N)
        inside = dist < GOAL_TOL
        run = torch.zeros(duck.n, dtype=torch.long, device=duck.dev)
        t_reach = torch.full((duck.n,), float("nan"), device=duck.dev)
        for t in range(T):
            run = torch.where(inside[t], run + 1, torch.zeros_like(run))
            hit = (run == hold) & torch.isnan(t_reach)
            t_reach[hit] = (t - hold + 1) * duck.dt
        reached = ~torch.isnan(t_reach)
        fell = (z < DOWN_Z).any(0)
        last = slice(-int(round(2.0 / duck.dt)), None)
        drift = (dist[last].max(0).values - dist[last].min(0).values)              # movement inside the last 2 s
        rows.append({
            "with_push": with_push, "push_mps": push_speed if with_push else 0.0, "n": duck.n,
            "reached": float(reached.float().mean()),
            "median_time_s": nanmedian(t_reach),
            "p90_time_s": float(torch.quantile(t_reach[reached], 0.9)) if reached.any() else float("nan"),
            "final_dist_m_median": float(dist[-1].median()),
            "fell": float(fell.float().mean()),
            "reached_of_fallen": float((reached & fell).float().sum() / max(int(fell.sum()), 1)) if with_push else float("nan"),
            "stop_drift_m_median": float(drift[reached].median()) if reached.any() else float("nan"),
            # diagnostics for the ones that did not make it: how far they ended up, and whether they were still moving
            "failed_final_dist_m_median": float(dist[-1][~reached].median()) if (~reached).any() else float("nan"),
            "failed_still_down": float(((z[last] < DOWN_Z).all(0) & ~reached).float().sum() / max(int((~reached).sum()), 1)) if (~reached).any() else float("nan"),
        })
    return rows


# ----------------------------------------------------------------------------- main
def run_trials(duck: Duck, only: set[str]) -> dict:
    out = {}
    with torch.inference_mode():   # resets write into buffers the policy/env created in inference mode; stay inside it
        if "getup" in only:
            out["getup"] = trial_getup(duck)
        if "push" in only:
            out["push"] = trial_push(duck)
        if "tracking" in only:
            out["tracking"] = trial_tracking(duck)
        if "body" in only:
            out["body"] = trial_body(duck)
        if "bodywalk" in only:
            out["bodywalk"] = trial_bodywalk(duck)
        if "goal" in only:
            out["goal"] = trial_goal(duck)
    return out


def summarize(res: dict) -> dict:
    """One row of headline numbers per policy (the leaderboard columns)."""
    s = {}
    g = res.get("getup")
    if g:
        s["getup_recovered"] = sum(r["recovered"] for r in g) / len(g)
        s["getup_median_s"] = sum(r["median_time_s"] for r in g) / len(g)
    p = res.get("push")
    if p:
        for spd in (1.0, 1.2):
            rows = [r for r in p if abs(r["push_mps"] - spd) < 1e-6]
            if rows:
                s[f"push_recovered_{spd:.1f}"] = sum(r["recovered_of_fallen"] for r in rows) / len(rows)
    t = res.get("tracking")
    if t:
        by = {r["case"]: r for r in t}
        if "walk 0.30" in by:
            s["speed_at_0.30"] = by["walk 0.30"]["vx"]; s["drift_p90_at_0.30"] = by["walk 0.30"].get("wz_abs_p90", float("nan"))
        if "walk 0.45" in by: s["speed_at_0.45"] = by["walk 0.45"]["vx"]
        if "turn +0.3" in by: s["turn_at_0.3"] = by["turn +0.3"]["wz"]
        if "turn +0.6" in by: s["turn_at_0.6"] = by["turn +0.6"]["wz"]
        if "turn +1.0" in by: s["turn_at_1.0"] = by["turn +1.0"]["wz"]
        if "turn -1.0" in by: s["turn_at_-1.0"] = by["turn -1.0"]["wz"]
    b = res.get("body")
    if b:
        by = {r["case"]: r for r in b}
        if "crouch -30 mm" in by: s["crouch_mm_for_-30"] = by["crouch -30 mm"]["dz_mm"]
        if "pitch +10" in by and "pitch -10" in by:
            s["pitch_deg_for_10"] = (by["pitch +10"]["pitch_deg"] - by["pitch -10"]["pitch_deg"]) / 2
        if "pitch +20" in by and "pitch -20" in by:
            s["pitch_deg_for_20"] = (by["pitch +20"]["pitch_deg"] - by["pitch -20"]["pitch_deg"]) / 2
        if "roll +10" in by and "roll -10" in by:
            s["roll_deg_for_10"] = (by["roll +10"]["roll_deg"] - by["roll -10"]["roll_deg"]) / 2
    bw = res.get("bodywalk")
    if bw:
        by = {r["case"]: r for r in bw}
        if "crouch -20 mm" in by: s["walk_crouch_mm_for_-20"] = by["crouch -20 mm"]["dz_mm"]
        if "roll +10" in by and "roll -10" in by:
            s["walk_roll_deg_for_10"] = (by["roll +10"]["roll_deg"] - by["roll -10"]["roll_deg"]) / 2
        if "pitch +10" in by and "pitch -10" in by:
            s["walk_pitch_deg_for_10"] = (by["pitch +10"]["pitch_deg"] - by["pitch -10"]["pitch_deg"]) / 2
        if "walk" in by and by["walk"]["vx"] > 1e-3:
            body = [r["vx"] for r in bw if r["case"] != "walk"]
            s["walk_speed_ratio_body"] = sum(body) / len(body) / by["walk"]["vx"]
        s["walk_body_fell"] = max(r["fell"] for r in bw)
    g = res.get("goal")
    if g:
        for r in g:
            tag = "push" if r["with_push"] else "nopush"
            s[f"goal_reached_{tag}"] = r["reached"]
            s[f"goal_median_s_{tag}"] = r["median_time_s"]
            s[f"goal_stop_drift_m_{tag}"] = r["stop_drift_m_median"]
    return s


def aggregate(per_seed: list[dict]) -> dict:
    """mean / std across seeds of every headline number."""
    keys = sorted({k for r in per_seed for k in r["summary"]})
    agg = {}
    for k in keys:
        v = [r["summary"][k] for r in per_seed if k in r["summary"] and r["summary"][k] == r["summary"][k]]
        if not v:
            continue
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        agg[k] = {"mean": m, "std": sd, "n": len(v)}
    return agg


def print_tables(res: dict):
    if "getup" in res:
        print("\n| spawn | started down | recovered | median s | p90 s | upright at end | end pitch |\n|---|---|---|---|---|---|---|")
        for r in res["getup"]:
            print(f"| {r['spawn']} | {fmt_pct(r['started_down'])} | {fmt_pct(r['recovered'])} | {r['median_time_s']:.2f} | {r['p90_time_s']:.2f} | {fmt_pct(r['upright_at_end'])} | {r['end_pitch_deg_median']:.1f}° |")
    if "push" in res:
        print("\n| mode | push | fell | recovered of fallen | median recovery s | upright at end |\n|---|---|---|---|---|---|")
        for r in res["push"]:
            print(f"| {r['mode']} | {r['push_mps']:.1f} m/s | {fmt_pct(r['fell'])} | {fmt_pct(r['recovered_of_fallen'])} | {r['median_recovery_s']:.2f} | {fmt_pct(r['upright_at_end'])} |")
            if r.get("failed_n"):
                print(f"    not recovered: {r['failed_n']} ducks — lying {fmt_pct(r['failed_lying'])}, low 70-100 mm {fmt_pct(r['failed_low'])}, "
                      f"up but tilted >15° {fmt_pct(r['failed_up_but_tilted'])}; median end z {r['failed_z_mm_median']:.0f} mm, tilt {r['failed_tilt_deg_median']:.0f}°")
    if "tracking" in res:
        print("\n| case | vx achieved / cmd | wz achieved / cmd | per-env \\|wz\\| p50 / p90 | fell |\n|---|---|---|---|---|")
        for r in res["tracking"]:
            print(f"| {r['case']} | {r['vx']:+.2f} / {r['cmd_vx']:+.2f} | {r['wz']:+.2f} / {r['cmd_wz']:+.2f} | {r['wz_abs_p50']:.2f} / {r['wz_abs_p90']:.2f} | {fmt_pct(r['fell'])} |")
    if "body" in res:
        print("\n| case | dz mm | roll ° | pitch ° | fell |\n|---|---|---|---|---|")
        for r in res["body"]:
            print(f"| {r['case']} | {r['dz_mm']:+.1f} | {r['roll_deg']:+.1f} | {r['pitch_deg']:+.1f} | {fmt_pct(r['fell'])} |")
    if res.get("bodywalk"):
        print("\n| walking 0.20 m/s + | dz mm | roll ° | pitch ° | vx | fell |\n|---|---|---|---|---|---|")
        for r in res["bodywalk"]:
            print(f"| {r['case']} | {r['dz_mm']:+.1f} | {r['roll_deg']:+.1f} | {r['pitch_deg']:+.1f} | {r['vx']:.2f} | {fmt_pct(r['fell'])} |")
    if "goal" in res:
        print(f"\n| goal {GOAL_DIST} m | reached | median s | p90 s | final dist (median) | fell | reached of fallen | stop drift (median) |\n|---|---|---|---|---|---|---|---|")
        for r in res["goal"]:
            label = f"push {r['push_mps']:.1f} m/s at 2 s" if r["with_push"] else "no push"
            print(f"| {label} | {fmt_pct(r['reached'])} | {r['median_time_s']:.1f} | {r['p90_time_s']:.1f} | {r['final_dist_m_median']:.2f} m | {fmt_pct(r['fell'])} | {fmt_pct(r['reached_of_fallen']) if r['reached_of_fallen'] == r['reached_of_fallen'] else '—'} | {r['stop_drift_m_median']*100:.1f} cm |")
            if r.get("failed_final_dist_m_median", float("nan")) == r.get("failed_final_dist_m_median", float("nan")):
                print(f"    not reached: final distance median {r['failed_final_dist_m_median']:.2f} m, still down at the end {fmt_pct(r['failed_still_down'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", default=None, help="label for the leaderboard (default: checkpoint file name)")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 0,1,2 (overrides --seed; reports mean ± std)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="getup,push,tracking,body")
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", default=None, help="folder: record env 0 (720p, close camera) from the first step")
    ap.add_argument("--video-steps", type=int, default=1700, help="recorded steps (50 per second)")
    a = ap.parse_args()
    t0 = time.time()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]
    only = set(a.only.split(","))
    name = a.name or os.path.basename(a.checkpoint)
    per_seed = []
    for seed in seeds:
        env, policy = build(a.task, a.checkpoint, a.num_envs, seed, a.device, video=a.video, video_steps=a.video_steps)
        duck = Duck(env, policy, seed)
        print(f"\n[battery] {name}  task={a.task}  envs={a.num_envs}  seed={seed}  dt={duck.dt:.3f}s  (setup {time.time()-t0:.0f} s)")
        res = run_trials(duck, only)
        env.close()
        print_tables(res)
        res["seed"] = seed
        res["summary"] = summarize(res)
        per_seed.append(res)
    out = {"name": name, "task": a.task, "checkpoint": a.checkpoint, "num_envs": a.num_envs, "seeds": seeds,
           "per_seed": per_seed, "aggregate": aggregate(per_seed), "seconds": time.time() - t0}
    print(f"\n[battery] headline ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}, mean ± std):")
    for k, v in out["aggregate"].items():
        print(f"  {k:<22} {v['mean']:+.3f} ± {v['std']:.3f}")
    print(f"[battery] done in {out['seconds']:.0f} s")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[battery] wrote {a.out}")


if __name__ == "__main__":
    main()
