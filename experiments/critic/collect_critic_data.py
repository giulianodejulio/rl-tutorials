from pathlib import Path

import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout
from experiments.policy_gradient.reinforce_batch_gradient import returns_to_go


def critic_features(velocity, target_velocity, t, num_steps):
    error = velocity - target_velocity
    remaining_fraction = (num_steps - t) / num_steps
    return np.array([1.0, error, error**2, remaining_fraction])


def collect_critic_data(weights, sigma, target_velocity, num_steps, num_rollouts, rng):
    env = PointMass1D()
    features = np.zeros((num_rollouts, num_steps, 4))
    returns = np.zeros((num_rollouts, num_steps))

    for i in range(num_rollouts):
        trajectory = collect_rollout(
            env, weights, sigma, target_velocity, num_steps, rng
        )
        returns[i] = returns_to_go(trajectory)
        for t, step in enumerate(trajectory):
            # Policy features contain velocity BEFORE action t at index 2.
            velocity = step["features"][2]
            features[i, t] = critic_features(velocity, target_velocity, t, num_steps)

    return features, returns


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma = 0.3
    target_velocity = 0.8
    num_steps = 200
    num_train_rollouts = 100
    num_validation_rollouts = 50

    train_features, train_returns = collect_critic_data(
        weights, sigma, target_velocity, num_steps, num_train_rollouts,
        np.random.default_rng(42),
    )
    # Split entire rollouts: neighboring steps share future rewards.
    validation_features, validation_returns = collect_critic_data(
        weights, sigma, target_velocity, num_steps, num_validation_rollouts,
        np.random.default_rng(2026),
    )

    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "critic_dataset.npz"
    np.savez(
        path, train_features=train_features, train_returns=train_returns,
        validation_features=validation_features, validation_returns=validation_returns,
        policy_weights=weights, sigma=sigma, target_velocity=target_velocity,
        num_steps=num_steps, training_seed=42, validation_seed=2026,
        feature_names=np.array(["bias", "velocity_error", "squared_error", "remaining_fraction"]),
    )
    print(f"Training: features {train_features.shape}, targets {train_returns.shape}")
    print(f"Validation: features {validation_features.shape}, targets {validation_returns.shape}")
    print("\nFirst rollout: (critic features, observed return)")
    for t in range(3):
        print(f"t={t}: {train_features[0, t]}, g_t={train_returns[0, t]:.6f}")
    print(f"\nPolicy weights unchanged: {weights}")
    print("No critic fitted; returns are undiscounted Monte Carlo targets.")
    print(f"Saved: {path}")
