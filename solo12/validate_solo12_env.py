"""Validate the Solo12 Gym interface and exercise zero/random offsets; no RL."""

from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

from solo12.solo12_env import Solo12Env


def collect_probe(env, random_offsets):
    observation, _ = env.reset(seed=2026)
    rng = np.random.default_rng(2026)
    records = []
    for _ in range(env.num_steps):
        action = rng.uniform(-0.1, 0.1, 12) if random_offsets else np.zeros(12)
        observation, reward, terminated, truncated, info = env.step(action)
        np.testing.assert_allclose(info["desired_positions"], env.reference + env.action_scale*action)
        records.append(dict(observation=observation, action=action, reward=reward,
                            time=env.data.time, **info))
        if terminated or truncated:
            break
    return {key: np.array([r[key] for r in records]) for key in records[0]}


def main():
    env = Solo12Env()
    check_env(env, warn=True)
    arrays = {}
    for name, random_offsets in (("zero_offsets", False), ("small_random_offsets", True)):
        probe = collect_probe(env, random_offsets)
        arrays.update({f"{name}_{key}": value for key, value in probe.items()})
        print(f"{name}: steps={len(probe['reward'])}, return={probe['reward'].sum():.4f}, "
              f"final height={probe['height'][-1]:.4f} m, fallen={probe['fallen'][-1]}")
    # Explicitly validate the fall terminal, in addition to the horizon terminal.
    env.reset()
    env.data.qpos[2] = env.floor_height + 0.05
    _, _, terminated, truncated, info = env.step(np.zeros(12))
    assert terminated and not truncated and info["fallen"]
    # Validate clipping of vector controls.
    env.reset()
    _, _, _, _, info = env.step(np.full(12, 2.0))
    np.testing.assert_allclose(info["desired_positions"], env.reference + env.action_scale)
    directory = Path(__file__).resolve().parents[1] / "results" / "solo12" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "solo12_env_validation.npz"
    np.savez(path, **arrays, nominal_reference=env.reference, joint_names=np.array(env.joint_names),
             action_scale=env.action_scale, control_dt=env.control_dt,
             frame_skip=env.frame_skip, num_steps=env.num_steps,
             target_velocity=env.target_velocity)
    env.close()
    print("Validated Gym interface, action mapping/clipping and fall termination.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
