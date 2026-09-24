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


def random_policy_action(actions):
    return random.choice(actions)


def td0_step(value_function, env, state, actions, alpha, gamma):
    action = random_policy_action(actions)
    next_state, reward = env.mdp_dynamics(state, action)

    row, col = state
    next_row, next_col = next_state

    old_value = value_function[row, col]
    td_target = reward + gamma * value_function[next_row, next_col]
    td_error = td_target - old_value
    value_function[row, col] = old_value + alpha * td_error

    return next_state, {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "old_value": old_value,
        "td_target": td_target,
        "td_error": td_error,
        "new_value": value_function[row, col],
    }


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]
    value_function = np.zeros((5, 5))

    alpha = 0.1
    gamma = 0.9
    state = (1, 1)
    num_steps = 10

    for step in range(num_steps):
        next_state, info = td0_step(value_function, env, state, actions, alpha, gamma)

        print(f"\nstep: {step}")
        for key, value in info.items():
            print(f"{key}: {value}")

        state = next_state

    print("\nValue function after TD(0) updates:")
    print(value_function)
