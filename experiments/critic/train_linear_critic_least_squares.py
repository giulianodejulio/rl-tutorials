"""Fit the same linear critic by least squares, using saved training data."""

from pathlib import Path

import numpy as np


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    with np.load(directory / "critic_dataset.npz", allow_pickle=False) as data:
        metadata = {key: data[key] for key in (
            "policy_weights", "feature_names", "sigma", "target_velocity", "num_steps"
        )}
        train_features = data["train_features"]
        train_returns = data["train_returns"]
        validation_features = data["validation_features"]
        validation_returns = data["validation_returns"]

    # Each row is one observed state-time pair; no validation data enter the fit.
    x = train_features.reshape(-1, train_features.shape[-1])
    y = train_returns.reshape(-1)
    theta, _, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    train_mse = np.mean((y - x @ theta)**2)
    validation_predictions = validation_features @ theta
    validation_mse = np.mean((validation_returns - validation_predictions)**2)

    # Re-evaluate saved gradient-descent weights on this same dataset.
    with np.load(directory / "linear_critic.npz", allow_pickle=False) as previous:
        for key, value in metadata.items():
            if not np.array_equal(value, previous[key]):
                raise ValueError(f"Dataset and saved critic differ in {key}.")
        gd_theta = previous["theta"]
        learning_rate = previous["learning_rate"]
        num_epochs = previous["num_epochs"]
    gd_train_mse = np.mean((y - x @ gd_theta)**2)
    gd_validation_mse = np.mean((validation_returns - validation_features @ gd_theta)**2)
    time_baseline = train_returns.mean(axis=0)
    baseline_train_mse = np.mean((train_returns - time_baseline)**2)
    baseline_validation_mse = np.mean((validation_returns - time_baseline)**2)

    np.set_printoptions(precision=6, suppress=True)
    print(f"Training samples: {len(y)}; features: {x.shape[1]}; rank: {rank}")
    print(f"Least-squares theta: {theta}")
    print(f"Gradient-descent theta: {gd_theta}")
    print(f"Gradient descent: learning_rate={learning_rate}, epochs={num_epochs}")
    print("Method                     Training MSE   Validation MSE")
    for name, training, validation in (
        ("Gradient descent", gd_train_mse, gd_validation_mse),
        ("Least squares", train_mse, validation_mse),
        ("Temporal baseline", baseline_train_mse, baseline_validation_mse),
    ):
        print(f"{name:26s} {training:.6f}       {validation:.6f}")
    print("MSE compares predictions with observed Monte Carlo returns.")

    path = directory / "linear_critic_least_squares.npz"
    np.savez(
        path, theta=theta, train_mse=train_mse,
        validation_predictions=validation_predictions, validation_mse=validation_mse,
        rank=rank, singular_values=singular_values,
        gd_theta=gd_theta, gd_train_mse=gd_train_mse,
        gd_validation_mse=gd_validation_mse,
        learning_rate=learning_rate, num_epochs=num_epochs,
        time_baseline=time_baseline, baseline_train_mse=baseline_train_mse,
        baseline_validation_mse=baseline_validation_mse, **metadata,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
