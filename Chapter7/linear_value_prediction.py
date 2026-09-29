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


def state_features(state):
    row, col = state
    return np.array([1.0, row / 4.0, col / 4.0])


def value(weights, state):
    return float(np.dot(weights, state_features(state)))


def linear_td_prediction(env, actions, alpha, gamma, num_rollouts, rollout_len):
    weights = np.zeros(3)

    for _ in range(num_rollouts):
        state = random_state()

        for _ in range(rollout_len):
            action = random.choice(actions)
            next_state, reward = env.mdp_dynamics(state, action)

            prediction = value(weights, state)
            td_target = reward + gamma * value(weights, next_state)
            td_error = td_target - prediction
            weights += alpha * td_error * state_features(state)

            state = next_state

    return weights


def value_grid(weights):
    grid = np.zeros((5, 5))

    for row in range(5):
        for col in range(5):
            grid[row, col] = value(weights, (row, col))

    return grid


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    alpha = 0.01
    num_rollouts = 1_000
    rollout_len = 100

    weights = linear_td_prediction(env, actions, alpha, gamma, num_rollouts, rollout_len)

    print("Learned weights:")
    print(weights)
    print("\nValue function approximated from weights:")
    print(value_grid(weights))
