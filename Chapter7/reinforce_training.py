import numpy as np

from continuous_control_rollout import PointMass1D
from reinforce_continuous_policy import collect_rollout
from reinforce_batch_gradient import rollout_gradient


def evaluate_policy(weights, sigma, target_velocity, num_steps, num_rollouts):
    # Reuse evaluation noise across policies, independently of training noise.
    rng = np.random.default_rng(2026)
    env = PointMass1D()
    returns = []
    for _ in range(num_rollouts):
        trajectory = collect_rollout(
            env, weights, sigma, target_velocity, num_steps, rng
        )
        returns.append(sum(step["reward"] for step in trajectory))
    return np.array(returns)


def train(initial_weights, sigma, target_velocity, num_steps,
          num_rollouts, num_updates, learning_rate):
    rng = np.random.default_rng(42)
    env = PointMass1D()
    weights = initial_weights.copy()

    for update in range(num_updates):
        gradients = []
        returns = []
        # Every trajectory in this batch uses the same current weights.
        for _ in range(num_rollouts):
            trajectory = collect_rollout(
                env, weights, sigma, target_velocity, num_steps, rng
            )
            gradients.append(rollout_gradient(trajectory, sigma))
            returns.append(sum(step["reward"] for step in trajectory))

        batch_gradient = np.mean(gradients, axis=0)
        weights += learning_rate * batch_gradient
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError("Non-finite policy weights")

        if update == 0 or (update + 1) % 10 == 0:
            print(
                f"update={update + 1:3d} "
                f"batch_return_before_update={np.mean(returns):.3f} "
                f"gradient_norm={np.linalg.norm(batch_gradient):.3f} "
                f"weights_after_update={weights}",
                flush=True,
            )

    return weights


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    initial_weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma = 0.3
    target_velocity = 0.8
    num_steps = 200
    num_rollouts = 100
    num_updates = 50
    learning_rate = 0.00005
    num_evaluation_rollouts = 200

    before = evaluate_policy(
        initial_weights, sigma, target_velocity, num_steps, num_evaluation_rollouts
    )
    final_weights = train(
        initial_weights, sigma, target_velocity, num_steps,
        num_rollouts, num_updates, learning_rate,
    )
    after = evaluate_policy(
        final_weights, sigma, target_velocity, num_steps, num_evaluation_rollouts
    )

    print(f"\nEvaluation: {num_evaluation_rollouts} rollouts, fixed sigma={sigma}")
    print(f"Initial return: mean={before.mean():.3f}, std={before.std(ddof=1):.3f}")
    print(f"Final return:   mean={after.mean():.3f}, std={after.std(ddof=1):.3f}")
    print(f"Mean paired improvement: {(after - before).mean():.3f}")
    print(f"Initial weights: {initial_weights}")
    print(f"Final weights:   {final_weights}")
