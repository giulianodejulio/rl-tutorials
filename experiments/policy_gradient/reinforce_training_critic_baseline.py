"""REINFORCE training with a separately fitted Monte Carlo critic baseline."""

from pathlib import Path

import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.critic.collect_critic_data import collect_critic_data
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout
from experiments.policy_gradient.reinforce_critic_baseline import predict_critic_baseline
from experiments.policy_gradient.reinforce_time_baseline import rollout_gradient_with_baseline
from experiments.policy_gradient.reinforce_training import evaluate_policy


def fit_current_critic(weights, sigma, target_velocity, num_steps, num_rollouts, rng):
    features, returns = collect_critic_data(
        weights, sigma, target_velocity, num_steps, num_rollouts, rng
    )
    x = features.reshape(-1, features.shape[-1])
    y = returns.reshape(-1)
    theta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    mse = np.mean((y - x @ theta)**2)
    return theta, mse, rank


def train_with_critic(initial_weights, sigma, target_velocity, num_steps,
                      num_rollouts, num_critic_rollouts, num_updates,
                      learning_rate, num_evaluation_rollouts):
    rng = np.random.default_rng(42)
    critic_rng = np.random.default_rng(31415)
    env = PointMass1D()
    weights = initial_weights.copy()
    weight_history = [weights.copy()]
    critic_history, critic_mse, critic_ranks = [], [], []
    gradients, batch_returns = [], []
    evaluation_updates = [0]
    evaluation_returns = [evaluate_policy(
        weights, sigma, target_velocity, num_steps, num_evaluation_rollouts
    )]

    for update in range(num_updates):
        theta, mse, rank = fit_current_critic(
            weights, sigma, target_velocity, num_steps, num_critic_rollouts, critic_rng
        )
        rollout_gradients, returns = [], []
        for _ in range(num_rollouts):
            trajectory = collect_rollout(
                env, weights, sigma, target_velocity, num_steps, rng
            )
            baseline = predict_critic_baseline(trajectory, theta, target_velocity)
            rollout_gradients.append(
                rollout_gradient_with_baseline(trajectory, sigma, baseline)
            )
            returns.append(sum(step["reward"] for step in trajectory))
        gradient = np.mean(rollout_gradients, axis=0)
        weights += learning_rate * gradient
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(theta)):
            raise FloatingPointError("Non-finite policy or critic weights")
        weight_history.append(weights.copy())
        critic_history.append(theta.copy())
        critic_mse.append(mse)
        critic_ranks.append(rank)
        gradients.append(gradient)
        batch_returns.append(returns)
        if update == 0 or (update + 1) % 10 == 0:
            print(f"update={update + 1}: critic_train_mse={mse:.6f}, "
                  f"batch_return_before_update={np.mean(returns):.3f}", flush=True)
        if (update + 1) % 10 == 0 or update + 1 == num_updates:
            evaluation_updates.append(update + 1)
            evaluation_returns.append(evaluate_policy(
                weights, sigma, target_velocity, num_steps, num_evaluation_rollouts
            ))
            print(f"  evaluation_after_update={evaluation_returns[-1].mean():.3f}",
                  flush=True)

    return dict(
        weights=np.array(weight_history), critic_weights=np.array(critic_history),
        critic_train_mse=np.array(critic_mse), critic_ranks=np.array(critic_ranks),
        gradients=np.array(gradients), batch_returns=np.array(batch_returns),
        evaluation_updates=np.array(evaluation_updates),
        evaluation_returns=np.array(evaluation_returns),
    )


def main():
    configuration = dict(
        initial_weights=np.array([0.0, 0.0, -1.0, 1.0]),
        sigma=0.3, target_velocity=0.8, num_steps=200,
        num_rollouts=100, num_critic_rollouts=100, num_updates=50,
        learning_rate=0.00005, num_evaluation_rollouts=200,
    )
    history = train_with_critic(**configuration)
    directory = Path(__file__).resolve().parents[2] / "results" / "policy_gradient" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reinforce_training_critic_baseline.npz"
    np.savez(path, **history, **configuration,
             training_seed=42, critic_seed=31415, evaluation_seed=2026)
    before, after = history["evaluation_returns"][[0, -1]]
    print(f"Initial return: {before.mean():.3f}; final return: {after.mean():.3f}")
    print(f"Mean paired improvement: {(after - before).mean():.3f}")
    print("The last critic was fitted BEFORE the final policy update.")
    print("One training seed; critic MSE is training error, not validation error.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
