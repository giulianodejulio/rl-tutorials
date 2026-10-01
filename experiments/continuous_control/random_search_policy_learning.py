import random

import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D, linear_policy


def evaluate_policy(weights, target_velocity, num_steps):
    env = PointMass1D()
    observation = env.reset()
    total_reward = 0.0

    for _ in range(num_steps):
        action = linear_policy(observation, target_velocity, weights)
        observation, reward = env.step(action, target_velocity)
        total_reward += reward

    return total_reward


def learn_with_random_search(initial_weights, target_velocity, num_steps, num_iterations, noise_scale):
    best_weights = np.copy(initial_weights)
    best_return = evaluate_policy(best_weights, target_velocity, num_steps)

    for iteration in range(num_iterations):
        candidate_weights = best_weights + np.random.normal(loc=0.0, scale=noise_scale, size=best_weights.shape)
        candidate_return = evaluate_policy(candidate_weights, target_velocity, num_steps)

        if candidate_return > best_return:
            best_weights = candidate_weights
            best_return = candidate_return
            print(
                f"iteration={iteration:4d} "
                f"best_return={best_return: .3f} "
                f"weights={best_weights}"
            )

    return best_weights, best_return


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)

    target_velocity = 0.8
    num_steps = 200
    num_iterations = 1_000
    noise_scale = 0.2

    initial_weights = np.zeros(4)

    print("Initial policy return:")
    print(evaluate_policy(initial_weights, target_velocity, num_steps))
    print("\nLearning:")

    best_weights, best_return = learn_with_random_search(initial_weights, target_velocity, num_steps, num_iterations, noise_scale)

    print("\nBest policy:")
    print(f"weights={best_weights}")
    print(f"return={best_return:.3f}")
