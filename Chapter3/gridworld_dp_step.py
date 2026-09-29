import numpy as np


class GridworldEnvironment:
    def __init__(self, size=5):
        self.size = size

    def step(self, state, action):
        row, col = state

        if state == (0, 1):
            return (4, 1), 10
        if state == (0, 3):
            return (2, 3), 5

        if action == "n":
            next_state = (max(row - 1, 0), col)
            reward = -1 if row == 0 else 0
        elif action == "s":
            next_state = (min(row + 1, self.size - 1), col)
            reward = -1 if row == self.size - 1 else 0
        elif action == "e":
            next_state = (row, min(col + 1, self.size - 1))
            reward = -1 if col == self.size - 1 else 0
        elif action == "w":
            next_state = (row, max(col - 1, 0))
            reward = -1 if col == 0 else 0
        else:
            raise ValueError(f"Invalid action: {action}")

        return next_state, reward


ACTIONS = ("n", "s", "e", "w")
ACTION_SYMBOLS = {
    "n": "^",
    "s": "v",
    "e": ">",
    "w": "<",
}


def evaluate_random_policy(env, gamma=0.9, tolerance=1e-4):
    values = np.zeros((env.size, env.size))
    action_probability = 1.0 / len(ACTIONS)
    iterations = 0

    while True:
        new_values = np.copy(values)
        delta = 0.0

        for row in range(env.size):
            for col in range(env.size):
                state = (row, col)
                expected_return = 0.0

                for action in ACTIONS:
                    next_state, reward = env.step(state, action)
                    next_row, next_col = next_state
                    expected_return += action_probability * (
                        reward + gamma * values[next_row, next_col]
                    )

                new_values[row, col] = expected_return
                delta = max(delta, abs(new_values[row, col] - values[row, col]))

        values = new_values
        iterations += 1

        if delta < tolerance:
            return values, iterations


def compute_optimal_values(env, gamma=0.9, tolerance=1e-4):
    values = np.zeros((env.size, env.size))
    policy = np.full((env.size, env.size), "", dtype=object)
    iterations = 0

    while True:
        new_values = np.copy(values)
        new_policy = np.copy(policy)
        delta = 0.0

        for row in range(env.size):
            for col in range(env.size):
                state = (row, col)
                action_returns = []

                for action in ACTIONS:
                    next_state, reward = env.step(state, action)
                    next_row, next_col = next_state
                    action_returns.append(
                        (reward + gamma * values[next_row, next_col], action)
                    )

                best_return = max(action_return for action_return, _ in action_returns)
                best_actions = [
                    action
                    for action_return, action in action_returns
                    if np.isclose(action_return, best_return)
                ]

                new_values[row, col] = best_return
                new_policy[row, col] = "".join(ACTION_SYMBOLS[action] for action in best_actions)
                delta = max(delta, abs(new_values[row, col] - values[row, col]))

        values = new_values
        policy = new_policy
        iterations += 1

        if delta < tolerance:
            return values, policy, iterations


def print_table(title, table):
    print(title)
    for row in table:
        print(" ".join(f"{cell:>7}" for cell in row))
    print()


def main():
    env = GridworldEnvironment()

    random_values, random_iterations = evaluate_random_policy(env)
    optimal_values, optimal_policy, optimal_iterations = compute_optimal_values(env)

    print(f"Random-policy evaluation converged in {random_iterations} iterations.")
    print_table("V(s) under random policy:", np.round(random_values, 1))

    print(f"Optimal value iteration converged in {optimal_iterations} iterations.")
    print_table("V*(s):", np.round(optimal_values, 1))
    print_table("Greedy optimal policy:", optimal_policy)


if __name__ == "__main__":
    main()
