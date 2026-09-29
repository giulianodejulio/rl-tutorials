import random

import numpy as np


class GridworldEnvironment:
    def mdp_dynamics(self, state, action):
        row, col = state

        if (row, col) == (0, 1):
            return (4, 1), 10
        if (row, col) == (0, 3):
            return (2, 3), 5

        if action == "n":
            next_state = (max(row - 1, 0), col)
            reward = -1 if row == 0 else 0
        elif action == "s":
            next_state = (min(row + 1, 4), col)
            reward = -1 if row == 4 else 0
        elif action == "e":
            next_state = (row, min(col + 1, 4))
            reward = -1 if col == 4 else 0
        elif action == "w":
            next_state = (row, max(col - 1, 0))
            reward = -1 if col == 0 else 0
        else:
            raise ValueError("Invalid action")

        return next_state, reward


def random_state():
    return random.randrange(5), random.randrange(5)


def evaluate_random_policy_with_dp(env, actions, gamma, theta=1e-4):
    value_function = np.zeros((5, 5))
    action_probability = 1.0 / len(actions)

    while True:
        delta = 0.0
        new_value_function = np.copy(value_function)

        for row in range(5):
            for col in range(5):
                expected_value = 0.0

                for action in actions:
                    next_state, reward = env.mdp_dynamics((row, col), action)
                    next_row, next_col = next_state
                    expected_value += action_probability * (
                        reward + gamma * value_function[next_row, next_col]
                    )

                new_value_function[row, col] = expected_value
                delta = max(delta, abs(expected_value - value_function[row, col]))

        value_function = new_value_function

        if delta < theta:
            return value_function


def td0_prediction_with_rollouts(env, actions, alpha, gamma, num_rollouts, rollout_len):
    value_function = np.zeros((5, 5))
    visit_counts = np.zeros((5, 5), dtype=int)

    for _ in range(num_rollouts):
        state = random_state()

        for _ in range(rollout_len):
            action = random.choice(actions)
            next_state, reward = env.mdp_dynamics(state, action)

            row, col = state
            next_row, next_col = next_state
            visit_counts[row, col] += 1

            td_target = reward + gamma * value_function[next_row, next_col]
            td_error = td_target - value_function[row, col]
            value_function[row, col] += alpha * td_error

            state = next_state

    return value_function, visit_counts


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    alpha = 0.05
    num_rollouts = 1_000
    rollout_len = 100

    dp_values = evaluate_random_policy_with_dp(env, actions, gamma)
    td_values, visit_counts = td0_prediction_with_rollouts(
        env, actions, alpha, gamma, num_rollouts, rollout_len
    )

    print("TD(0) value function:")
    print(td_values)
    print("\nDP value function:")
    print(dp_values)
    print("\nVisit counts:")
    print(visit_counts)
    print(f"\nMean absolute error: {np.mean(np.abs(td_values - dp_values)):.3f}")
