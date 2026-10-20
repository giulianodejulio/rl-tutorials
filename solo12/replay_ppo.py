"""Observe the saved PPO policy without learning; optional viewer or saved GIF."""

import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
from stable_baselines3 import PPO
import torch

from solo12.solo12_env import Solo12Env


@contextmanager
def passive_viewer(model, data):
    from mujoco import viewer as viewer_module

    # In the installed Linux viewer, close() requests exit but does not join
    # its daemon thread. Interpreter shutdown can terminate GLFW too early.
    # No other threads are launched by this script during this narrow window.
    existing_threads = set(threading.enumerate())
    viewer = viewer_module.launch_passive(model, data)
    viewer_threads = set(threading.enumerate()) - existing_threads
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in viewer_threads:
            thread.join()


def save_gif(model, qpos, qvel, dt, path, stochastic):
    from PIL import Image, ImageDraw

    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, -1.5e-1]
    camera.distance, camera.azimuth, camera.elevation = 1.1e0, 135, -20
    frames = []
    stride = 4
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        for index in range(0, len(qpos)-1, stride):
            data.qpos[:], data.qvel[:] = qpos[index], qvel[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 480, 24), fill="white")
            draw.text((8, 6), f"t={index*dt:.2e} s | {'Gaussian samples' if stochastic else 'Mean actions'} | No learning",
                      fill="black")
            frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=round(stride*dt*1e3), loop=0)


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="ppo_first", help="Saved training folder name")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="Sample Gaussian actions instead of using means")
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = repository / "results" / "solo12" / "data" / args.run
    model = PPO.load(directory / "policy.zip", device="cpu")
    with np.load(directory / "training.npz") as config:
        env = Solo12Env(
            slip_weight=float(config["slip_weight"]) if "slip_weight" in config else 0.0,
            smoothness_weight=float(config["smoothness_weight"]) if "smoothness_weight" in config else 0.0,
            action_history=bool(config["action_history"]) if "action_history" in config else False)
    obs, _ = env.reset(seed=2026)
    initial_x = env.data.qpos[0]
    rng = np.random.default_rng(2026)
    qpos, qvel = [env.data.qpos.copy()], [env.data.qvel.copy()]
    actions, rewards, velocities = [], [], []
    fallen = False
    with ExitStack() as stack:
        viewer = None
        if args.viewer:
            viewer = stack.enter_context(passive_viewer(env.model, env.data))
        for _ in range(env.num_steps):
            if viewer is not None and not viewer.is_running():
                break
            start = time.monotonic()
            with torch.no_grad():
                distribution = model.policy.get_distribution(
                    torch.tensor(obs[None], dtype=torch.float32)
                ).distribution
                mean = distribution.mean.numpy()[0]
                action = rng.normal(mean, distribution.stddev.numpy()[0]) if args.stochastic else mean
            obs, reward, terminated, truncated, info = env.step(action)
            qpos.append(env.data.qpos.copy())
            qvel.append(env.data.qvel.copy())
            actions.append(action.copy())
            rewards.append(reward)
            velocities.append(info["vx"])
            fallen = info["fallen"]
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0, env.control_dt-(time.monotonic()-start)))
            if terminated or truncated:
                break
    if not rewards:
        env.close()
        return
    mode = "stochastic" if args.stochastic else "mean"
    qpos, qvel = np.array(qpos), np.array(qvel)
    path = directory / f"replay_{mode}.npz"
    np.savez(path, qpos=qpos, qvel=qvel, actions=np.array(actions), rewards=np.array(rewards),
             vx=np.array(velocities), control_dt=env.control_dt, stochastic=args.stochastic,
             fallen=fallen, evaluation_seed=2026)
    print(f"Duration={len(rewards)*env.control_dt:.3e} s; return={np.sum(rewards):.3e}")
    print(f"Mean vx={np.mean(velocities):.3e} m/s; displacement x={env.data.qpos[0]-initial_x:.3e} m")
    print(f"Fallen={fallen}; saved: {path}")
    if args.gif:
        output = repository / "results" / "solo12" / "figures" / f"{args.run}_{mode}.gif"
        output.parent.mkdir(parents=True, exist_ok=True)
        save_gif(env.model, qpos, qvel, env.control_dt, output, args.stochastic)
        print(f"Saved: {output}")
    env.close()


if __name__ == "__main__":
    main()
