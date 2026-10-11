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
def build(task: str, ckpt: str, num_envs: int, seed: int, device: str, episode_s: float = 60.0):
    configure_torch_backends()
    env_cfg = load_env_cfg(task, play=True)
    agent_cfg = load_rl_cfg(task)
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
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
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
            rows.append({
                "mode": mode, "push_mps": speed, "n": duck.n,
                "fell": float(fell.float().mean()),
                "recovered_of_fallen": float(rec.float().sum() / max(int(fell.sum()), 1)),
                "median_recovery_s": nanmedian(t_up[fell]) if fell.any() else float("nan"),
                "upright_at_end": float(up_mask(tr)[-1].float().mean()),
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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="getup,push,tracking,body")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    env, policy = build(a.task, a.checkpoint, a.num_envs, a.seed, a.device)
    duck = Duck(env, policy, a.seed)
    print(f"[battery] {a.task} {a.checkpoint} envs={a.num_envs} seed={a.seed} dt={duck.dt:.3f}s  (setup {time.time()-t0:.0f} s)")
    res = {"task": a.task, "checkpoint": a.checkpoint, "num_envs": a.num_envs, "seed": a.seed}
    only = set(a.only.split(","))
    with torch.inference_mode():   # resets write into buffers the policy/env created in inference mode; stay inside it
        if "getup" in only:
            res["getup"] = trial_getup(duck)
            print("\n| spawn | started down | recovered | median s | p90 s | upright at end | end pitch |\n|---|---|---|---|---|---|---|")
            for r in res["getup"]:
                print(f"| {r['spawn']} | {fmt_pct(r['started_down'])} | {fmt_pct(r['recovered'])} | {r['median_time_s']:.2f} | {r['p90_time_s']:.2f} | {fmt_pct(r['upright_at_end'])} | {r['end_pitch_deg_median']:.1f}° |")
        if "push" in only:
            res["push"] = trial_push(duck)
            print("\n| mode | push | fell | recovered of fallen | median recovery s | upright at end |\n|---|---|---|---|---|---|")
            for r in res["push"]:
                print(f"| {r['mode']} | {r['push_mps']:.1f} m/s | {fmt_pct(r['fell'])} | {fmt_pct(r['recovered_of_fallen'])} | {r['median_recovery_s']:.2f} | {fmt_pct(r['upright_at_end'])} |")
        if "tracking" in only:
            res["tracking"] = trial_tracking(duck)
            print("\n| case | vx achieved / cmd | wz achieved / cmd | per-env \\|wz\\| p50 / p90 | fell |\n|---|---|---|---|---|")
            for r in res["tracking"]:
                print(f"| {r['case']} | {r['vx']:+.2f} / {r['cmd_vx']:+.2f} | {r['wz']:+.2f} / {r['cmd_wz']:+.2f} | {r['wz_abs_p50']:.2f} / {r['wz_abs_p90']:.2f} | {fmt_pct(r['fell'])} |")
        if "body" in only:
            res["body"] = trial_body(duck)
            print("\n| case | dz mm | roll ° | pitch ° | fell |\n|---|---|---|---|---|")
            for r in res["body"]:
                print(f"| {r['case']} | {r['dz_mm']:+.1f} | {r['roll_deg']:+.1f} | {r['pitch_deg']:+.1f} | {fmt_pct(r['fell'])} |")
    res["seconds"] = time.time() - t0
    print(f"\n[battery] done in {res['seconds']:.0f} s")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[battery] wrote {a.out}")
    env.close()


if __name__ == "__main__":
    main()
