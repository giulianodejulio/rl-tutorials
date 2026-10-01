import random

import numpy as np

from experiments.function_approximation.linear_value_prediction import GridworldEnvironment


def state_action_features(state, action, actions):
    row, col = state
    row = row / 4.0
    col = col / 4.0

    features_per_action = 3
    features = np.zeros(features_per_action * len(actions))
    action_index = actions.index(action)
    start = action_index * features_per_action

    features[start : start + features_per_action] = np.array([1.0, row, col])
    return features


def q_value(weights, state, action, actions):
    features = state_action_features(state, action, actions)
    return float(np.dot(weights, features))


def greedy_action(weights, state, actions):
    q_values = [q_value(weights, state, action, actions) for action in actions]
    return actions[int(np.argmax(q_values))]


def epsilon_greedy_action(weights, state, actions, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    return greedy_action(weights, state, actions)


def random_state():
    return random.randrange(5), random.randrange(5)


def linear_q_learning(env, actions, alpha, gamma, epsilon, num_rollouts, rollout_len):
    weights = np.zeros(3 * len(actions))

    for _ in range(num_rollouts):
        state = random_state()

        for _ in range(rollout_len):
            action = epsilon_greedy_action(weights, state, actions, epsilon)
            next_state, reward = env.mdp_dynamics(state, action)

            prediction = q_value(weights, state, action, actions)
            best_next_action = greedy_action(weights, next_state, actions)
            td_target = reward + gamma * q_value(
                weights,
                next_state,
                best_next_action,
                actions,
            )
            td_error = td_target - prediction

            features = state_action_features(state, action, actions)
            weights += alpha * td_error * features

            state = next_state

    return weights


def greedy_policy_grid(weights, actions):
    symbols = {"n": "^", "s": "v", "e": ">", "w": "<"}
    rows = []

    for row in range(5):
        row_symbols = []
        for col in range(5):
            action = greedy_action(weights, (row, col), actions)
            row_symbols.append(symbols[action])
        rows.append(" ".join(row_symbols))

    return "\n".join(rows)


def max_q_value_grid(weights, actions):
    grid = np.zeros((5, 5))

    for row in range(5):
        for col in range(5):
            grid[row, col] = max(
                q_value(weights, (row, col), action, actions)
                for action in actions
            )

    return grid


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    alpha = 0.01
    epsilon = 0.1
    num_rollouts = 5_000
    rollout_len = 50

    weights = linear_q_learning(
        env,
        actions,
        alpha,
        gamma,
        epsilon,
        num_rollouts,
        rollout_len,
    )

    print("Learned weights:")
    print(weights)
    print("\nGreedy policy learned from approximated Q(s, a):")
    print(greedy_policy_grid(weights, actions))
    print("\nmax_a Q(s, a):")
    print(max_q_value_grid(weights, actions))
