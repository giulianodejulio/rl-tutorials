import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout
from experiments.policy_gradient.reinforce_batch_gradient import returns_to_go, rollout_gradient


def estimate_time_baseline(env, weights, sigma, target_velocity,
                           num_steps, num_reference_rollouts, rng):
    reference_returns = []
    for _ in range(num_reference_rollouts):
        trajectory = collect_rollout(
            env, weights, sigma, target_velocity, num_steps, rng
        )
        reference_returns.append(returns_to_go(trajectory))
    # Rows are rollouts; columns are time steps.
    return np.mean(reference_returns, axis=0)


def rollout_gradient_with_baseline(trajectory, sigma, baseline):
    returns = returns_to_go(trajectory)
    gradient = np.zeros_like(trajectory[0]["features"])
    for t, step in enumerate(trajectory):
        score = (
            (step["action"] - step["mean_action"]) / sigma**2
        ) * step["features"]
        gradient += (returns[t] - baseline[t]) * score
    return gradient


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma = 0.3
    target_velocity = 0.8
    num_steps = 200
    num_reference_rollouts = 100
    num_rollouts = 100
    env = PointMass1D()

    baseline = estimate_time_baseline(
        env, weights, sigma, target_velocity, num_steps,
        num_reference_rollouts, np.random.default_rng(2026),
    )
    # Comparison rollouts are separate from the reference data.
    rng = np.random.default_rng(42)
    without_baseline = []
    with_baseline = []
    for _ in range(num_rollouts):
        trajectory = collect_rollout(
            env, weights, sigma, target_velocity, num_steps, rng
        )
        without_baseline.append(rollout_gradient(trajectory, sigma))
        with_baseline.append(
            rollout_gradient_with_baseline(trajectory, sigma, baseline)
        )

    without_baseline = np.array(without_baseline)
    with_baseline = np.array(with_baseline)
    print(f"Reference rollouts: {num_reference_rollouts}")
    print(f"Comparison rollouts: {num_rollouts}")
    print(f"Baseline, first 3 steps: {baseline[:3]}")
    print(f"Baseline, last step: {baseline[-1]:.4f}")
    print("\nGradient components: bias, position, velocity, target")
    print(f"Mean without baseline: {without_baseline.mean(axis=0)}")
    print(f"Mean with baseline:    {with_baseline.mean(axis=0)}")
    std_without = without_baseline.std(axis=0, ddof=1)
    std_with = with_baseline.std(axis=0, ddof=1)
    print(f"Std without baseline:  {std_without}")
    print(f"Std with baseline:     {std_with}")
    print(f"Std ratio (with / without): {std_with / std_without}")
    print(f"\nWeights unchanged: {weights}")
    print("Means need not match on a finite batch. Std is across rollout gradients.")
