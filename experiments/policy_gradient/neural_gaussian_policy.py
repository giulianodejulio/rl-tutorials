"""First neural Gaussian policy: forward pass and rollouts, no training."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.reinforce_batch_gradient import returns_to_go


def build_policy(hidden_size):
    # Inputs: position, velocity, target velocity. Biases belong to Linear layers.
    # No output activation: the Gaussian mean may be any real number.
    return nn.Sequential(nn.Linear(3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))


def sample_action(model, observation, target_velocity, sigma, rng):
    inputs = np.array([observation[0], observation[1], target_velocity])
    # Only collect data here; a later step will recompute log probabilities
    # with gradients enabled, treating recorded states and actions as fixed data.
    with torch.no_grad():
        mean_action = model(torch.tensor(inputs, dtype=torch.float32)).item()
    action = float(rng.normal(mean_action, sigma))
    return inputs, mean_action, action


def collect_neural_rollout(env, model, sigma, target_velocity, num_steps, rng):
    observation = env.reset()
    trajectory = []
    for _ in range(num_steps):
        inputs, mean_action, action = sample_action(
            model, observation, target_velocity, sigma, rng
        )
        next_observation, reward = env.step(action, target_velocity)
        trajectory.append(dict(inputs=inputs, mean_action=mean_action,
                               action=action, reward=reward))
        observation = next_observation
    return trajectory


def main():
    hidden_size, initialization_seed, rollout_seed = 16, 123, 42
    sigma, target_velocity, num_steps, num_rollouts = 0.3, 0.8, 200, 100
    torch.manual_seed(initialization_seed)
    model = build_policy(hidden_size)
    rng = np.random.default_rng(rollout_seed)
    env = PointMass1D()
    trajectories = [collect_neural_rollout(
        env, model, sigma, target_velocity, num_steps, rng
    ) for _ in range(num_rollouts)]
    inputs = np.array([[s["inputs"] for s in tau] for tau in trajectories])
    mean_actions = np.array([[s["mean_action"] for s in tau] for tau in trajectories])
    actions = np.array([[s["action"] for s in tau] for tau in trajectories])
    rewards = np.array([[s["reward"] for s in tau] for tau in trajectories])
    returns = np.array([returns_to_go(tau) for tau in trajectories])
    if not all(np.isfinite(a).all() for a in (inputs, mean_actions, actions, rewards, returns)):
        raise FloatingPointError("Non-finite neural policy rollout")
    directory = Path(__file__).resolve().parents[2] / "results" / "policy_gradient" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "neural_gaussian_policy.npz"
    np.savez(path, inputs=inputs, mean_actions=mean_actions, actions=actions,
             rewards=rewards, returns=returns, sigma=sigma,
             target_velocity=target_velocity, num_steps=num_steps,
             num_rollouts=num_rollouts, hidden_size=hidden_size,
             initialization_seed=initialization_seed, rollout_seed=rollout_seed,
             input_names=np.array(["position", "velocity", "target_velocity"]))
    model_path = directory / "neural_gaussian_policy.pt"
    torch.save(dict(state_dict=model.state_dict(), hidden_size=hidden_size,
                    input_names=["position", "velocity", "target_velocity"]), model_path)
    print(f"Mean initial action: {mean_actions[0, 0]:.6f}")
    print(f"Untrained policy return: mean={returns[:, 0].mean():.3f}, "
          f"std={returns[:, 0].std(ddof=1):.3f}")
    print("Randomly initialized network, not the previous hand-initialized linear policy.")
    print("Original Gaussian actions saved; clipping occurs only in env.step.")
    print(f"Saved: {path}\nSaved: {model_path}")


if __name__ == "__main__":
    main()
