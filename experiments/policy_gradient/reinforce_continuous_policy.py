import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D


def gaussian_policy(observation, target_velocity, weights, sigma, rng):
    position, velocity = observation
    features = np.array([1.0, position, velocity, target_velocity])
    mean_action = float(np.dot(weights, features))
    action = float(rng.normal(mean_action, sigma))
    return features, mean_action, action


def collect_rollout(env, weights, sigma, target_velocity, num_steps, rng):
    observation = env.reset()
    trajectory = []

    for _ in range(num_steps):
        features, mean_action, action = gaussian_policy(
            observation, target_velocity, weights, sigma, rng
        )
        next_observation, reward = env.step(action, target_velocity)

        # Store the Gaussian sample before the environment clips it.
        trajectory.append({
            "features": features,
            "mean_action": mean_action,
            "action": action,
            "reward": reward,
        })
        observation = next_observation

    return trajectory


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    env = PointMass1D()
    weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma = 0.3

    trajectory = collect_rollout(
        env, weights, sigma, target_velocity=0.8, num_steps=200, rng=rng
    )

    for t, step in enumerate(trajectory[:5]):
        applied_action = np.clip(step["action"], -1.0, 1.0)
        print(
            f"t={t} features={step['features']} "
            f"mean={step['mean_action']:.3f} "
            f"sampled={step['action']:.3f} applied={applied_action:.3f} "
            f"reward={step['reward']:.3f}"
        )

    total_reward = sum(step["reward"] for step in trajectory)
    print(f"\nStored steps: {len(trajectory)}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Weights (no update): {weights}")
