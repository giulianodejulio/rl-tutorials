import numpy as np

from continuous_control_rollout import PointMass1D
from reinforce_continuous_policy import collect_rollout


def returns_to_go(trajectory):
    returns = np.zeros(len(trajectory))
    future_return = 0.0
    for t in reversed(range(len(trajectory))):
        future_return = trajectory[t]["reward"] + future_return
        returns[t] = future_return
    return returns


def rollout_gradient(trajectory, sigma):
    returns = returns_to_go(trajectory)
    gradient = np.zeros_like(trajectory[0]["features"])
    for t, step in enumerate(trajectory):
        # Use the original Gaussian sample, not the clipped acceleration.
        score = (
            (step["action"] - step["mean_action"]) / sigma**2
        ) * step["features"]
        gradient += returns[t] * score
    return gradient


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    rng = np.random.default_rng(42)
    env = PointMass1D()
    weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma = 0.3
    num_rollouts = 100
    num_steps = 200
    learning_rate = 0.001

    gradients = []
    total_returns = []
    for rollout_index in range(num_rollouts):
        # Keep weights fixed across the entire batch; do not reset the RNG.
        trajectory = collect_rollout(
            env, weights, sigma, target_velocity=0.8,
            num_steps=num_steps, rng=rng,
        )
        gradient = rollout_gradient(trajectory, sigma)
        gradients.append(gradient)
        returns = returns_to_go(trajectory)
        total_returns.append(returns[0])

        if rollout_index == 0:
            print("First rollout, first three steps:")
            for t in range(3):
                print(
                    f"t={t} reward={trajectory[t]['reward']:.6f} "
                    f"return_to_go={returns[t]:.6f}"
                )
            print(f"First rollout gradient: {gradient}")

    # Sum over steps inside each rollout, then average over rollouts.
    batch_gradient = np.mean(gradients, axis=0)
    new_weights = weights + learning_rate * batch_gradient

    print(f"\nRollouts: {num_rollouts}, steps per rollout: {num_steps}")
    print(f"Mean total return (before update): {np.mean(total_returns):.6f}")
    print(f"Return standard deviation: {np.std(total_returns, ddof=1):.6f}")
    print(f"Batch gradient: {batch_gradient}")
    print(f"Learning rate: {learning_rate}")
    print(f"Old weights: {weights}")
    print(f"Weight change: {new_weights - weights}")
    print(f"New weights (one update, not evaluated): {new_weights}")
