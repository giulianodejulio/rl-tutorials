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


def q_learning_control(env, actions, alpha, gamma, num_rollouts, rollout_len):
    q_values = np.zeros((5, 5, len(actions)))

    for rollout in range(num_rollouts):
        epsilon = max(0.01, 1.0 - rollout / (0.8 * num_rollouts))
        state = random_state()

        for _ in range(rollout_len):
            action_index = epsilon_greedy_action(q_values, state, epsilon)
            next_state, reward = env.mdp_dynamics(state, actions[action_index])

            row, col = state
            next_row, next_col = next_state

            td_target = reward + gamma * np.max(q_values[next_row, next_col])
            td_error = td_target - q_values[row, col, action_index]
            q_values[row, col, action_index] += alpha * td_error

            state = next_state

    return q_values


def evaluate_greedy_policy(env, actions, q_values, num_rollouts, rollout_len):
    total_reward = 0.0

    for _ in range(num_rollouts):
        state = random_state()

        for _ in range(rollout_len):
            row, col = state
            action_index = int(np.argmax(q_values[row, col]))
            next_state, reward = env.mdp_dynamics(state, actions[action_index])

            total_reward += reward
            state = next_state

    return total_reward / num_rollouts


def greedy_policy(q_values, actions):
    symbols = {"n": "^", "s": "v", "e": ">", "w": "<"}
    policy = []

    for row in range(5):
        policy_row = []

        for col in range(5):
            best_action_index = int(np.argmax(q_values[row, col]))
            policy_row.append(symbols[actions[best_action_index]])

        policy.append(policy_row)

    return policy


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    alpha = 0.1
    num_rollouts = 500
    rollout_len = 100
    eval_rollouts = 200

    q_values = q_learning_control(env, actions, alpha, gamma, num_rollouts, rollout_len)

    random.seed(1)
    reward = evaluate_greedy_policy(env, actions, q_values, eval_rollouts, rollout_len)

    print(f"Greedy policy reward after epsilon decay: {reward:.2f}")
    print("\nGreedy policy learned from Q:")
    for row in greedy_policy(q_values, actions):
        print(" ".join(row))
