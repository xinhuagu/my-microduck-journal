# Microduck policy benchmark

One fixed exam for every walking / get-up policy that speaks the Microduck contract
(`obs[1, 61] → actions[1, 14]`, 50 Hz): the same simulated environment, the same 256 randomised
ducks, the same three seeds, the same forced spawns, pushes and commands. Official policies, the
ones trained in this repository and community policies all go through it unchanged, so the numbers
below are directly comparable. Everything is simulation (mjlab / MuJoCo Warp), flat ground.

## Run it on your own policy

```bash
# inside a microduck_rl checkout (see docs/setup-wsl2-rtx5090.md); copy scripts/eval_battery.py and
# scripts/eval_battery.sh from this repo next to each other, then:
bash eval_battery.sh Mjlab-VelStand-Flat-MicroDuck my_policy.onnx 256 getup,push,tracking,body out.json 0,1,2 my_policy
python benchmark_table.py <folder with the json files> --out table.md
```

About 15 minutes per seed on an RTX 5090 shared with nothing else. The script also takes rsl_rl
`.pt` checkpoints (`latest:<experiment>[:<iteration>]` picks the newest run directory). Use the
`Velocity` task for pure walking policies and the `StandUp` task for stand-up policies (then
`getup,body` only).

## What is measured

| trial | setup | reported |
|---|---|---|
| get-up | face-down / face-up / on-side orientation with a random sprawled joint pose, dropped from 12 cm, held 1 s with the policy off, then 8 s | recovered = trunk ≥ 100 mm and tilt ≤ 15° for one continuous second; rate, median and p90 time, final pitch |
| push | standing, or walking at 0.30 m/s for 3 s; then the trunk's horizontal velocity is set to 1.0 or 1.2 m/s in a random direction; 6 s | fell = trunk below 90 mm; recovered of fallen (timed from the push), upright at the end |
| tracking | 8 s per command: walk 0.15 / 0.30 / 0.45 m/s, turn in place ±0.3 / 0.6 / 1.0 rad/s | mean over the last 5 s of body-frame forward speed and yaw rate; per-env \|yaw\| median / p90 as drift |
| body | standing; crouch −30 mm, raise +20 mm, roll ±10°, pitch ±10° and ±20°; 5 s each | achieved after 3 s (last 2 s), falls judged by tilt |

The table shows the headline numbers: get-up rate (mean of the three spawns) and median time, push
recovery at 1.0 and 1.2 m/s (mean of standing and walking), achieved speed for a 0.30 m/s command,
yaw-drift p90 at 0.30 m/s, achieved in-place yaw rate for 0.3 / 0.6 / 1.0 rad/s, achieved crouch
for −30 mm, achieved tilt for ±10° / ±20° pitch and ±10° roll (mean of both directions). Cells are
mean ± standard deviation over the three seeds. Raw JSON per policy, with every trial row and seed,
is in [`results/benchmark/`](../results/benchmark/).

## Results (2026-10-13, upstream `microduck_rl` @ `ceca0170`)

| policy | seeds | get-up | get-up median | push 1.0 m/s recovered | push 1.2 m/s recovered | speed at 0.30 | yaw drift p90 | turn 0.3 | turn 0.6 | turn 1.0 | crouch −30 mm | pitch ±10° | pitch ±20° | roll ±10° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official_velstand_v7 | 3 | 96 % ± 1 | 0.60 s ± 0.01 | 94 % ± 1 | 93 % ± 1 | 0.14 m/s ± 0.00 | 0.16 ± 0.01 | 0.00 ± 0.00 | 0.05 ± 0.01 | 0.38 ± 0.03 | -0.7 mm ± 0.0 | 8.0° ± 0.0 | 16.1° ± 0.0 | 7.8° ± 0.0 |
| ours_velstand_v4 | 3 | 99 % ± 1 | 0.52 s ± 0.02 | 95 % ± 2 | 97 % ± 1 | 0.18 m/s ± 0.00 | 0.18 ± 0.01 | 0.00 ± 0.00 | 0.39 ± 0.01 | 1.10 ± 0.00 | -28.7 mm ± 0.1 | 10.0° ± 0.0 | 19.7° ± 0.0 | 9.6° ± 0.0 |
| ours_velstand_v5_turning | 3 | 95 % ± 0 | 0.57 s ± 0.01 | 89 % ± 2 | 89 % ± 1 | 0.12 m/s ± 0.00 | 0.06 ± 0.00 | 0.02 ± 0.00 | 0.68 ± 0.00 | 1.12 ± 0.00 | -25.8 mm ± 0.1 | 9.8° ± 0.0 | 19.4° ± 0.0 | 9.1° ± 0.0 |
| ours_velstand_v2 | 3 | 97 % ± 1 | 0.52 s ± 0.01 | 85 % ± 1 | 86 % ± 0 | 0.18 m/s ± 0.00 | 0.18 ± 0.01 | 0.00 ± 0.00 | 0.38 ± 0.01 | 0.92 ± 0.00 | -0.5 mm ± 0.0 | 3.1° ± 0.0 | 3.2° ± 0.0 | 1.5° ± 0.0 |
| official_alpha_walking | 3 | 0 % ± 0 | — | 4 % ± 1 | 0 % ± 0 | 0.15 m/s ± 0.00 | 0.11 ± 0.01 | 0.00 ± 0.00 | 0.02 ± 0.00 | 0.35 ± 0.01 | 0.1 mm ± 0.0 | 0.0° ± 0.0 | 0.1° ± 0.0 | 0.0° ± 0.0 |
| ours_walk_teacher_v1 | 3 | 0 % ± 0 | — | 4 % ± 1 | 1 % ± 0 | 0.19 m/s ± 0.00 | 0.15 ± 0.00 | 0.02 ± 0.00 | 0.57 ± 0.00 | 1.01 ± 0.00 | 0.3 mm ± 0.0 | 0.1° ± 0.0 | -0.1° ± 0.0 | -0.1° ± 0.0 |
| ours_walk_teacher_v2_turn | 3 | 0 % ± 0 | — | 2 % ± 0 | 0 % ± 0 | 0.14 m/s ± 0.00 | 0.08 ± 0.01 | 0.07 ± 0.00 | 0.64 ± 0.00 | 1.09 ± 0.00 | -0.1 mm ± 0.0 | 0.2° ± 0.0 | 0.2° ± 0.0 | 0.0° ± 0.0 |
| community_running_HannesVonEssen | 3 | 49 % ± 2 | 1.41 s ± 0.01 | 38 % ± 4 | 34 % ± 2 | 1.64 m/s ± 0.00 | 0.30 ± 0.01 | -0.16 ± 0.00 | -0.10 ± 0.02 | -0.10 ± 0.01 | -1.0 mm ± 0.2 | -0.1° ± 0.0 | 0.4° ± 0.1 | -0.4° ± 0.1 |
| official_alpha_stand | 3 | 100 % ± 0 | 0.48 s ± 0.01 | — | — | — | — | — | — | — | 0.5 mm ± 0.0 | 9.3° ± 0.0 | 18.5° ± 0.0 | 9.1° ± 0.0 |
| ours_standup_teacher_v3 | 3 | 100 % ± 0 | 0.40 s ± 0.01 | — | — | — | — | — | — | — | -28.2 mm ± 0.0 | 9.5° ± 0.0 | 19.6° ± 0.0 | 9.4° ± 0.0 |

Reading the table:

- `official_*` are the files shipped on the official Hugging Face repository (`velstand.onnx` is the
  deployed walk + get-up policy, v7; `alpha_walking` and `alpha_stand` are the teachers it was
  distilled from). `ours_*` are this repository's policies (release v0.1 and the training log).
  `community_running_HannesVonEssen` is a community running policy trained for 1.65 m/s: it ignores
  the small commands used here (runs at 1.64 m/s whatever is asked), so its tracking cells are not a
  fault of the policy, and it has no get-up training.
- Walk teachers (`alpha_walking`, `ours_walk_teacher_*`) have no get-up by design; their get-up and
  push columns say so. Stand teachers are only run through the get-up and body trials.
- A dash means the trial was not run for that policy.
- Seeds change the domain randomisation and the random spawns, pushes and directions; the
  policies are deterministic. Spreads are 0–2 points on rates and ≤ 0.03 rad/s on yaw rates, so a
  single seed already separates the policies — the three seeds are there to show that.

Known blind spots of this exam, on the to-do list: no rough terrain or slopes, no impact forces per
body part (servo housings, head), no energy / torque measure, and the push is a velocity step rather
than a force over time.
