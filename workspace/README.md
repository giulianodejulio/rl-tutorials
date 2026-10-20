# Recreate solo12_rl_ws on another Linux PC

This versioned setup replaces the dependency on the original laptop's personal
`~/scripts/docker` helpers. Docker Engine, Git and Python 3 must already be
installed, and the user must be allowed to run Docker. It does not install Docker,
change the host daemon, prune images, start training or modify other workspaces.

## Clone and prepare

Run on the destination PC as your normal user:

```bash
mkdir -p ~/workspaces/solo12_rl_ws/src
git clone git@github.com:giulianodejulio/rl-tutorials.git \
  ~/workspaces/solo12_rl_ws/src/rl-tutorials
cd ~/workspaces/solo12_rl_ws/src/rl-tutorials
bash workspace/bootstrap.sh
bash workspace/build.sh
bash workspace/run.sh
```

GitHub access is required for cloning; HTTPS can replace SSH if preferred:
`https://github.com/giulianodejulio/rl-tutorials.git`.

The checkout must live at `<workspace>/src/rl-tutorials`, because existing model
paths derive `<workspace>/third_party` from that layout. Username is not fixed
and `/home/gdj` is not used by these setup scripts.

Bootstrap downloads the shared `rkourdis/solo12_mujoco` repository at revision
`846cd588f461050068aaea03511915a8ddd8f06a`. It installs the versioned generated
`solo12_gen.xml`, verifies hashes of **all XML/OBJ/STL assets**, and creates local
runtime files only if absent. This preserves the exact asset digest used by
saved paper checkpoints, including the generated XML's timestamp comment.
It refuses to overwrite a differing existing model or reset another revision.

The canonical Dockerfile is `workspace/Dockerfile` inside this repository.
Build it with the supplied script rather than relying on an untracked Dockerfile
at the workspace root. Build context is only `workspace/`; results and credentials
are not sent into the Docker build.

The image uses Ubuntu 22.04/Python 3.10, Torch 2.2.2 **CPU**, NumPy 1.26.4,
Gymnasium 0.29.1, SB3 2.3.2 and MuJoCo 3.14.0, plus plotting/LaTeX/OpenGL tools.
Other utilities use compatible releases resolved during build. Thus the key RL
versions are pinned, but this is not an immutable apt/PyPI dependency lockfile.

The container runs persistently in the background, with matching host UID/GID
and the entire workspace mounted at its actual absolute path. Only the selected
workspace is mounted; there are no privileged mode, host-network or /dev mounts.
Default names: image `solo12_rl_ws:base`, container `docker_solo12_rl_ws`.
Override with `SOLO12_IMAGE`/`SOLO12_CONTAINER` if those names already exist.
The run script refuses to replace an existing container.

## Check before training (no learning)

```bash
docker exec -it \
  -w "$HOME/workspaces/solo12_rl_ws/src/rl-tutorials" \
  docker_solo12_rl_ws python3 -m paper.validate
```

This creates explicitly untrained test artifacts in `paper/checks/`. The default
container uses CPU software rendering (`MUJOCO_GL=osmesa`) and needs no NVIDIA
container runtime. Training also runs on CPU. A GTX 1080 Ti is not needed for it.

## Benchmark the desktop

Start with four workers, then compare six if desired, using separate run names:

```bash
time docker exec \
  -w "$HOME/workspaces/solo12_rl_ws/src/rl-tutorials" \
  docker_solo12_rl_ws python3 -m paper.train \
  --run desktop_bench4 --num-envs 4 --steps 300000
```

Change `desktop_bench4` to `desktop_bench6` and `--num-envs 4` to `--num-envs 6`
for the second benchmark. Both keep the same 30000-transition PPO batch. A
300000-step run completes ten PPO updates and saves its final checkpoint.

For full training choose a **new run name**:

```bash
docker exec -it \
  -w "$HOME/workspaces/solo12_rl_ws/src/rl-tutorials" \
  docker_solo12_rl_ws python3 -m paper.train \
  --run article_desktop --num-envs 4 --steps 300000000
```

Evaluate and create all plots and GIFs with software rendering:

```bash
docker exec -it \
  -w "$HOME/workspaces/solo12_rl_ws/src/rl-tutorials" \
  docker_solo12_rl_ws python3 -m paper.evaluate \
  --run article_desktop --episodes 10 --gif
```

To resume from the last **complete** checkpoint:

```bash
docker exec -it \
  -w "$HOME/workspaces/solo12_rl_ws/src/rl-tutorials" \
  docker_solo12_rl_ws python3 -m paper.train \
  --run article_desktop --resume --steps 300000000
```

Current checkpoint cadence is every 3e6 transitions and at normal completion.
Ctrl+C does **not** save current weights: progress since the latest checkpoint
is lost. For more frequent saves, copy `paper/configs/article.json` to a new
config and set `checkpoint_every` to 10 (every 3e5 transitions), then pass
`--config PATH` when creating the new run. Keep that custom config under paper/.

## Optional NVIDIA rendering

If the host has NVIDIA Container Toolkit configured, `workspace/run.sh --gpu`
requests GPU access and EGL rendering. It does not move PPO or physics to GPU.
The default CPU Torch build is sufficient for this option. To use a CUDA Torch
build for separate future experiments, rebuild with an explicit index, e.g.
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash workspace/build.sh`.
Do not infer CUDA compatibility from the driver-reported CUDA version alone.

For an interactive X11 viewer additional display authentication/mounts are
needed; the supplied run command intentionally supports headless training and
GIFs. See [paper/README.md](../paper/README.md) and [solo12/README.md](../solo12/README.md)
for the experiment protocols. Container stop/start preserves mounted results.

## What is versioned

The current checkout contains tutorial/Solo12/paper code, configurations and the
shared model setup. Generated datasets, checkpoints, plots, GIFs and logs under
`results/` and `paper/results/` stay local and are ignored by Git. Regenerate them
with the experiment scripts or transfer selected runs separately between PCs.
Previously published outputs remain in Git history.

The interrupted laptop paper run reached 1.08e6 transitions in its local CSV,
while its only saved checkpoint was at zero steps; it provides no trained policy
to resume on the desktop. Cache files, shell history, Docker runtime files and
regenerable validation fixtures are excluded. No Docker image binary is
committed; build it with the commands above.
