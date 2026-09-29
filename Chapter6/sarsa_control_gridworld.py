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


def epsilon_greedy_action(q_values, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(q_values.shape[-1])

    row, col = state
    return int(np.argmax(q_values[row, col]))


def sarsa_control(env, actions, alpha, gamma, epsilon, num_rollouts, rollout_len):
    q_values = np.zeros((5, 5, len(actions)))

    for _ in range(num_rollouts):
        state = random_state()
        action_index = epsilon_greedy_action(q_values, state, epsilon)

        for _ in range(rollout_len):
            action = actions[action_index]
            next_state, reward = env.mdp_dynamics(state, action)
            next_action_index = epsilon_greedy_action(q_values, next_state, epsilon)

            row, col = state
            next_row, next_col = next_state

            td_target = reward + gamma * q_values[next_row, next_col, next_action_index]
            td_error = td_target - q_values[row, col, action_index]
            q_values[row, col, action_index] += alpha * td_error

            state = next_state
            action_index = next_action_index

    return q_values


def greedy_policy(q_values, actions):
    symbols = {"n": "^", "s": "v", "e": ">", "w": "<"}
    policy = []

    for row in range(5):
        policy_row = []

        for col in range(5):
            best_action_index = int(np.argmax(q_values[row, col]))
            best_action = actions[best_action_index]
            policy_row.append(symbols[best_action])

        policy.append(policy_row)

    return policy


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1
    num_rollouts = 3_000
    rollout_len = 100

    q_values = sarsa_control(env, actions, alpha, gamma, epsilon, num_rollouts, rollout_len)

    print("Actions:", actions)
    print("\nQ values for state (0, 0):")
    print(q_values[0, 0])
    print("\nGreedy policy learned from Q:")
    for row in greedy_policy(q_values, actions):
        print(" ".join(row))
