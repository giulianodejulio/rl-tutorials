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


def td0_with_constant_alpha(env, actions, alpha, gamma, num_rollouts, rollout_len):
    value_function = np.zeros((5, 5))

    for _ in range(num_rollouts):
        state = random_state()

        for _ in range(rollout_len):
            action = random.choice(actions)
            next_state, reward = env.mdp_dynamics(state, action)

            row, col = state
            next_row, next_col = next_state

            td_target = reward + gamma * value_function[next_row, next_col]
            td_error = td_target - value_function[row, col]
            value_function[row, col] += alpha * td_error

            state = next_state

    return value_function


def td0_with_decay_alpha(env, actions, gamma, num_rollouts, rollout_len):
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
            alpha = 1.0 / visit_counts[row, col]

            td_target = reward + gamma * value_function[next_row, next_col]
            td_error = td_target - value_function[row, col]
            value_function[row, col] += alpha * td_error

            state = next_state

    return value_function


if __name__ == "__main__":
    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    num_rollouts = 1_000
    rollout_len = 100

    dp_values = evaluate_random_policy_with_dp(env, actions, gamma)

    random.seed(0)
    constant_values = td0_with_constant_alpha(
        env, actions, alpha=0.01, gamma=gamma, num_rollouts=num_rollouts, rollout_len=rollout_len
    )

    random.seed(0)
    decay_values = td0_with_decay_alpha(env, actions, gamma, num_rollouts, rollout_len)

    constant_error = np.mean(np.abs(constant_values - dp_values))
    decay_error = np.mean(np.abs(decay_values - dp_values))

    print("method         | mean absolute error")
    print("---------------+--------------------")
    print(f"constant alpha | {constant_error:18.3f}")
    print(f"decay alpha    | {decay_error:18.3f}")
