# Solo12 paper method — isolated MuJoCo port

Code, configurations, reference snapshot, checks, checkpoints, datasets, plots
and GIFs specific to this method live under `paper/`. Only existing robot
mesh/inertia assets are read from the shared workspace `third_party/`.
Tutorial `solo12/` code, saved models and results are untouched.

Read [SOURCES.md](SOURCES.md) for equation-by-equation provenance and explicit
MuJoCo/RaiSim differences. This is a simulation locomotion implementation,
not a numerical reproduction or a hardware deployment package.

## Commands from the host

For another PC, first follow [workspace/README.md](../workspace/README.md).
The prepared container has all required packages. Use the repository
as working directory (no installation and no changes to the workspace image):

```bash
docker exec -it -w /home/gdj/workspaces/solo12_rl_ws/src/rl-tutorials \
  docker_solo12_rl_ws python3 -m paper.train --run article_desktop --num-envs 8 --steps 300000000
```

That is the full reported sample budget, 3e8 transitions / 10000 PPO updates.
It is a substantial CPU run; no training-time estimate or convergence guarantee
is implied. Checkpoints every 100 updates (3e6 transitions), fixed nominal
validation every 500 updates (1.5e7 transitions). Initial policy is also evaluated.
A checkpoint is a paired model + observation normalizer; policy std is learned.

For a preliminary run, use `--run article_pilot --steps 30000000`: 3e7 samples,
1000 updates. At that point kc=3e-1, kn=2e-1 and rough terrain has not started.
**Do not judge the full method from that intermediate stage alone.** The
schedule is based on absolute samples and is not compressed for smaller runs.

After training, or while another process trains, evaluate the latest committed
checkpoint with fixed test commands, frozen normalization and full penalties:

```bash
docker exec -it \
  -w /home/gdj/workspaces/solo12_rl_ws/src/rl-tutorials \
  docker_solo12_rl_ws python3 -m paper.evaluate --run article_desktop --episodes 10 --gif
```

The portable container defaults to OSMesa software rendering. The original
GPU-enabled container can instead use `-e MUJOCO_GL=egl` when needed.

The test suite includes forward commands 2e-1,5e-1,1 m/s, reverse, lateral, yaw,
mixed command and stopping. For each: mean actions, stochastic actions and zero
actions as a baseline; nominal conditions and randomized PD/observations/terrain.
Deterministic nominal tests have one episode (repeating it adds no information);
other cases use the requested count. Metrics report full reward, velocity errors,
slip, power loss, reference changes, falls and duration. Short failed episodes
must not be mistaken for low error over a full surviving episode.

Optional: `--nominal-only`, `--commands x02 x05`, `--duration 5`, or
`--checkpoint 30000000` to evaluate a specific saved transition count. Evaluation
writes only below that checkpoint's `evaluations/` folder and never learns.

To resume the latest complete checkpoint after Ctrl+C/interruption:

```bash
docker exec -it -w /home/gdj/workspaces/solo12_rl_ws/src/rl-tutorials \
  docker_solo12_rl_ws python3 -m paper.train --run article_desktop --resume --steps 300000000
```

A pilot can be extended with `--run article_pilot --resume --steps 300000000`.
Incomplete collection since the last checkpoint is discarded. Resume restores
model, optimizer and normalization, but resets physical environments/RNG streams;
it is statistically a continuation, not a bit-for-bit uninterrupted replay.
Existing runs cannot be overwritten by a new training invocation.

## Outputs to examine

`paper/results/<run>/` contains:
- `config.json`, `source/`: resolved configuration and source snapshot.
- `history.csv`, `training.png`: per-buffer physical/reward diagnostics. Training
  commands/noise change with curriculum: this is not a fixed test distribution.
- `checkpoints/<steps>/`: `policy.zip`, `normalization.pkl`, version/source/asset
  manifest; `latest.json` points only to a fully written pair.
- `evaluations/<steps>/quick/`: fixed nominal commands, mean actions, full costs,
  saved during training; `fixed_evaluation.png` shows their tracking errors.
- `evaluations/<steps>/full/`: per-episode CSV, protocol JSON, nominal/randomized
  comparison plots, recorded trajectories, trajectory plots and optional GIFs
  for forward 2e-1 and 5e-1 m/s, with mean/stochastic actions.

Rebuild all existing plots without simulation:

```bash
docker exec -w /home/gdj/workspaces/solo12_rl_ws/src/rl-tutorials \
  docker_solo12_rl_ws python3 -m paper.plot --run article
```

To assess locomotion, look at speed error + slip + contact patterns + fall rate
and duration, **not reward alone**. Raw reward values are not comparable to the
previous tutorial's rate-times-dt reward. For physical comparisons use nominal
`x02` trajectories and the recorded material-contact-point slip index; the
article's foot-frame slip objective is a different quantity. One trained seed
is a diagnostic, not evidence of robustness across training seeds.

## Checks without training

```bash
docker exec -w /home/gdj/workspaces/solo12_rl_ws/src/rl-tutorials \
  docker_solo12_rl_ws python3 -m paper.validate
```

This runs mechanical/MDP tests, Gymnasium/SB3 checks, two worker processes,
checkpoint/normalizer reload, plotting and rendering with an explicitly
**untrained** network. It never calls `learn`, backpropagation or optimizer.step.
Artifacts are under `paper/checks/`, separate from training results.
