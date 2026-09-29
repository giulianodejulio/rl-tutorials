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


def evaluate_random_policy_with_dp(env, actions, gamma, theta=1e-4):
    value_function = np.zeros((5, 5))
    action_probability = 1.0 / len(actions)

    while True:
        delta = 0.0
        new_value_function = np.copy(value_function)

        for row in range(5):
            for col in range(5):
                state = (row, col)
                expected_value = 0.0

                for action in actions:
                    next_state, reward = env.mdp_dynamics(state, action)
                    next_row, next_col = next_state
                    expected_value += action_probability * (
                        reward + gamma * value_function[next_row, next_col]
                    )

                new_value_function[row, col] = expected_value
                delta = max(delta, abs(expected_value - value_function[row, col]))

        value_function = new_value_function

        if delta < theta:
            return value_function


def td0_prediction(env, actions, alpha, gamma, num_steps):
    value_function = np.zeros((5, 5))
    state = (1, 1)

    for _ in range(num_steps):
        action = random.choice(actions)
        next_state, reward = env.mdp_dynamics(state, action)

        row, col = state
        next_row, next_col = next_state

        td_target = reward + gamma * value_function[next_row, next_col]
        td_error = td_target - value_function[row, col]
        value_function[row, col] += alpha * td_error

        state = next_state

    return value_function


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    alpha = 0.05
    num_steps = 100_000

    dp_values = evaluate_random_policy_with_dp(env, actions, gamma)
    td_values = td0_prediction(env, actions, alpha, gamma, num_steps)

    print("DP value function:")
    print(dp_values)
    print("\nTD(0) value function:")
    print(td_values)
    print("\nTD(0) - DP:")
    print(td_values - dp_values)
    print(f"\nMean absolute error: {np.mean(np.abs(td_values - dp_values)):.3f}")
