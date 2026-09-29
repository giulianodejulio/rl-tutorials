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


def evaluate_random_policy_q_with_dp(env, actions, gamma, theta=1e-4):
    q_values = np.zeros((5, 5, len(actions)))
    action_probability = 1.0 / len(actions)

    while True:
        delta = 0.0
        new_q_values = np.copy(q_values)

        for row in range(5):
            for col in range(5):
                for action_index, action in enumerate(actions):
                    next_state, reward = env.mdp_dynamics((row, col), action)
                    next_row, next_col = next_state
                    next_value = 0.0

                    for next_action_index in range(len(actions)):
                        next_value += action_probability * q_values[
                            next_row, next_col, next_action_index
                        ]

                    target = reward + gamma * next_value
                    new_q_values[row, col, action_index] = target
                    delta = max(delta, abs(target - q_values[row, col, action_index]))

        q_values = new_q_values

        if delta < theta:
            return q_values


def td0_q_prediction(env, actions, alpha, gamma, num_rollouts, rollout_len):
    q_values = np.zeros((5, 5, len(actions)))

    for _ in range(num_rollouts):
        state = random_state()
        action_index = random.randrange(len(actions))

        for _ in range(rollout_len):
            action = actions[action_index]
            next_state, reward = env.mdp_dynamics(state, action)
            next_action_index = random.randrange(len(actions))

            row, col = state
            next_row, next_col = next_state

            td_target = reward + gamma * q_values[next_row, next_col, next_action_index]
            td_error = td_target - q_values[row, col, action_index]
            q_values[row, col, action_index] += alpha * td_error

            state = next_state
            action_index = next_action_index

    return q_values


if __name__ == "__main__":
    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    alpha = 0.01
    num_rollouts = 1_000
    rollout_len = 100

    dp_q_values = evaluate_random_policy_q_with_dp(env, actions, gamma)

    random.seed(0)
    td_q_values = td0_q_prediction(env, actions, alpha, gamma, num_rollouts, rollout_len)

    print("Actions:", actions)
    print("\nQ values for state (0, 0), estimated with TD:")
    print(td_q_values[0, 0])
    print("\nQ values for state (0, 0), computed with DP:")
    print(dp_q_values[0, 0])
    print(f"\nMean absolute error: {np.mean(np.abs(td_q_values - dp_q_values)):.3f}")
