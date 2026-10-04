"""Compare REINFORCE gradients at fixed weights using three baselines."""

from pathlib import Path

import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.critic.collect_critic_data import critic_features
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout
from experiments.policy_gradient.reinforce_time_baseline import rollout_gradient_with_baseline


def predict_critic_baseline(trajectory, theta, target_velocity):
    num_steps = len(trajectory)
    return np.array([
        critic_features(step["features"][2], target_velocity, t, num_steps) @ theta
        for t, step in enumerate(trajectory)
    ])


def collect_gradient_comparison(weights, sigma, target_velocity, num_steps,
                                theta, time_baseline, num_rollouts, rng):
    env = PointMass1D()
    gradients = np.zeros((num_rollouts, 3, len(weights)))
    for i in range(num_rollouts):
        trajectory = collect_rollout(
            env, weights, sigma, target_velocity, num_steps, rng
        )
        baselines = (
            np.zeros(num_steps), time_baseline,
            predict_critic_baseline(trajectory, theta, target_velocity),
        )
        for method, baseline in enumerate(baselines):
            gradients[i, method] = rollout_gradient_with_baseline(
                trajectory, sigma, baseline
            )
    return gradients


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    with np.load(directory / "linear_critic_least_squares.npz", allow_pickle=False) as model:
        theta = model["theta"]
        weights = model["policy_weights"]
        sigma = float(model["sigma"])
        target_velocity = float(model["target_velocity"])
        num_steps = int(model["num_steps"])
        time_baseline = model["time_baseline"]
        saved_train_mse = float(model["train_mse"])
    with np.load(directory / "critic_dataset.npz", allow_pickle=False) as data:
        for key, value in (("policy_weights", weights), ("sigma", sigma),
                           ("target_velocity", target_velocity), ("num_steps", num_steps)):
            np.testing.assert_array_equal(data[key], value)
        np.testing.assert_allclose(time_baseline, data["train_returns"].mean(axis=0))
        np.testing.assert_allclose(
            np.mean((data["train_returns"] - data["train_features"] @ theta)**2),
            saved_train_mse,
        )
        excluded_seeds = {int(data["training_seed"]), int(data["validation_seed"])}
    comparison_seed = 271828
    if comparison_seed in excluded_seeds:
        raise ValueError("Comparison seed must differ from training and validation seeds.")
    num_rollouts = 1000
    gradients = collect_gradient_comparison(
        weights, sigma, target_velocity, num_steps, theta, time_baseline,
        num_rollouts, np.random.default_rng(comparison_seed),
    )
    names = np.array(["No baseline", "Temporal baseline", "Critic baseline"])
    means = gradients.mean(axis=0)
    stds = gradients.std(axis=0, ddof=1)
    # Variance of a batch mean is rollout variance / batch size.
    batch_size = 100
    standard_errors = stds / np.sqrt(num_rollouts)
    batch_standard_deviations = stds / np.sqrt(batch_size)
    np.set_printoptions(precision=6, suppress=True)
    print("Components: bias, position, velocity, target")
    for method, name in enumerate(names):
        print(f"\n{name}")
        print(f"Mean gradient: {means[method]}")
        print(f"Rollout gradient std: {stds[method]}")
        print(f"Mean standard error ({num_rollouts} rollouts): {standard_errors[method]}")
        print(f"Predicted batch gradient std (N={batch_size}): {batch_standard_deviations[method]}")
        print(f"Sum of component variances: {np.sum(stds[method]**2):.6f}")
    output_directory = directory.parents[1] / "policy_gradient" / "data"
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "reinforce_critic_baseline.npz"
    np.savez(path, gradients=gradients, method_names=names, means=means,
             rollout_stds=stds, mean_standard_errors=standard_errors,
             batch_standard_deviations=batch_standard_deviations,
             batch_size=batch_size, num_rollouts=num_rollouts,
             comparison_seed=comparison_seed, policy_weights=weights,
             theta=theta, sigma=sigma, target_velocity=target_velocity,
             num_steps=num_steps, time_baseline=time_baseline)
    print(f"\nPolicy weights unchanged: {weights}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
