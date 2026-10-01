import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout
from experiments.policy_gradient.reinforce_time_baseline import (
    estimate_time_baseline,
    rollout_gradient_with_baseline,
)
from experiments.policy_gradient.reinforce_training import evaluate_policy, train


def train_with_time_baseline(initial_weights, sigma, target_velocity, num_steps,
                             num_rollouts, num_reference_rollouts, num_updates,
                             learning_rate, num_evaluation_rollouts):
    rng = np.random.default_rng(42)
    # Separate from batch noise (42) and evaluation noise (2026).
    reference_rng = np.random.default_rng(31415)
    env = PointMass1D()
    weights = initial_weights.copy()

    for update in range(num_updates):
        # Re-estimate the reference with the current policy before each batch.
        baseline = estimate_time_baseline(
            env, weights, sigma, target_velocity, num_steps,
            num_reference_rollouts, reference_rng,
        )
        gradients = []
        returns = []
        for _ in range(num_rollouts):
            trajectory = collect_rollout(
                env, weights, sigma, target_velocity, num_steps, rng
            )
            gradients.append(
                rollout_gradient_with_baseline(trajectory, sigma, baseline)
            )
            returns.append(sum(step["reward"] for step in trajectory))

        batch_gradient = np.mean(gradients, axis=0)
        weights += learning_rate * batch_gradient
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError("Non-finite policy weights")

        if update == 0 or (update + 1) % 10 == 0:
            print(
                f"update={update + 1:3d} "
                f"baseline_t0={baseline[0]:.3f} "
                f"batch_return_before_update={np.mean(returns):.3f} "
                f"gradient_norm={np.linalg.norm(batch_gradient):.3f}",
                flush=True,
            )
        if (update + 1) % 10 == 0:
            evaluation_returns = evaluate_policy(
                weights, sigma, target_velocity, num_steps, num_evaluation_rollouts
            )
            print(f"  evaluation_after_update={evaluation_returns.mean():.3f}", flush=True)

    return weights


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    initial_weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma = 0.3
    target_velocity = 0.8
    num_steps = 200
    num_rollouts = 100
    num_reference_rollouts = 100
    num_updates = 50
    learning_rate = 0.00005
    num_evaluation_rollouts = 200

    print("Training without baseline:", flush=True)
    weights_without = train(
        initial_weights, sigma, target_velocity, num_steps,
        num_rollouts, num_updates, learning_rate,
    )
    print("\nTraining with time baseline:", flush=True)
    weights_with = train_with_time_baseline(
        initial_weights, sigma, target_velocity, num_steps,
        num_rollouts, num_reference_rollouts, num_updates,
        learning_rate, num_evaluation_rollouts,
    )

    print("\nEvaluation: same 200 noise sequences for all policies")
    for label, weights in [("Initial", initial_weights),
                           ("Without baseline", weights_without),
                           ("With baseline", weights_with)]:
        evaluation_returns = evaluate_policy(
            weights, sigma, target_velocity, num_steps, num_evaluation_rollouts
        )
        print(
            f"{label}: mean={evaluation_returns.mean():.3f}, "
            f"std={evaluation_returns.std(ddof=1):.3f}, weights={weights}"
        )

    print("\nLearning rollout cost (excluding evaluation):")
    print(f"Without baseline: {num_updates * num_rollouts}")
    print(f"With baseline: {num_updates * (num_rollouts + num_reference_rollouts)}")
    print("Same update count, not the same simulation budget. One training seed only.")
