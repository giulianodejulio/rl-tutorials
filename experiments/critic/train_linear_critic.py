from pathlib import Path

import numpy as np


def train_critic(features, returns, learning_rate, num_epochs):
    # Keep each state-time feature vector paired with its observed return.
    x = features.reshape(-1, features.shape[-1])
    y = returns.reshape(-1)
    theta = np.zeros(x.shape[1])
    train_mse = []

    for epoch in range(num_epochs + 1):
        predictions = x @ theta
        errors = y - predictions
        train_mse.append(np.mean(errors**2))
        if epoch % 200 == 0 or epoch == num_epochs:
            print(f"epoch={epoch:4d} train_mse={train_mse[-1]:.6f}", flush=True)
        if epoch == num_epochs:
            break

        # Gradient descent on mean squared error / 2; targets stay fixed.
        theta += learning_rate * (x.T @ errors) / len(y)

    return theta, np.array(train_mse)


def main():
    np.set_printoptions(precision=6, suppress=True)
    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    dataset = directory / "critic_dataset.npz"
    if not dataset.exists():
        raise FileNotFoundError(
            f"Missing {dataset}. Run python3 -m experiments.critic.collect_critic_data first."
        )
    with np.load(dataset, allow_pickle=False) as data:
        train_features = data["train_features"]
        train_returns = data["train_returns"]
        validation_features = data["validation_features"]
        validation_returns = data["validation_returns"]
        policy_weights = data["policy_weights"]
        feature_names = data["feature_names"]
        sigma = data["sigma"]
        target_velocity = data["target_velocity"]
        num_steps = data["num_steps"]

    learning_rate = 0.1
    num_epochs = 1000
    theta, train_mse = train_critic(
        train_features, train_returns, learning_rate, num_epochs
    )
    validation_predictions = validation_features @ theta
    validation_mse = np.mean((validation_returns - validation_predictions)**2)
    initial_validation_mse = np.mean(validation_returns**2)

    # Fit the temporal reference on training rollouts only, not validation.
    time_baseline = train_returns.mean(axis=0)
    baseline_validation_mse = np.mean((validation_returns - time_baseline)**2)
    print(f"\nCritic weights theta: {theta}")
    print(f"Fixed policy weights w: {policy_weights}")
    print(f"Initial validation MSE (zero prediction): {initial_validation_mse:.6f}")
    print(f"Final validation MSE (critic): {validation_mse:.6f}")
    print(f"Validation MSE (time baseline): {baseline_validation_mse:.6f}")
    print("MSE is against sampled returns, not against the unknown exact value function.")

    path = directory / "linear_critic.npz"
    np.savez(
        path, theta=theta, train_mse=train_mse,
        validation_predictions=validation_predictions, validation_mse=validation_mse,
        time_baseline=time_baseline, baseline_validation_mse=baseline_validation_mse,
        learning_rate=learning_rate, num_epochs=num_epochs,
        policy_weights=policy_weights, feature_names=feature_names,
        sigma=sigma, target_velocity=target_velocity, num_steps=num_steps,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
