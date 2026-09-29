import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from linear_value_prediction import GridworldEnvironment


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


def random_state():
    return random.randrange(5), random.randrange(5)


def linear_features(state):
    row, col = state
    row = row / 4.0
    col = col / 4.0
    return np.array([1.0, row, col])


def quadratic_features(state):
    row, col = state
    row = row / 4.0
    col = col / 4.0
    return np.array([1.0, row, col, row * row, col * col, row * col])


def value(weights, features):
    return float(np.dot(weights, features))


def approximate_value_with_td(env, actions, feature_function, alpha, gamma, num_rollouts, rollout_len):
    weights = np.zeros(len(feature_function((0, 0))))

    for _ in range(num_rollouts):
        state = random_state()

        for _ in range(rollout_len):
            action = random.choice(actions)
            next_state, reward = env.mdp_dynamics(state, action)

            features = feature_function(state)
            next_features = feature_function(next_state)

            prediction = value(weights, features)
            td_target = reward + gamma * value(weights, next_features)
            td_error = td_target - prediction
            weights += alpha * td_error * features

            state = next_state

    return weights


def value_grid(weights, feature_function):
    grid = np.zeros((5, 5))

    for row in range(5):
        for col in range(5):
            grid[row, col] = value(weights, feature_function((row, col)))

    return grid


def plot_surface_and_points(surface_values, point_values, title, output_path):
    cols, rows = np.meshgrid(np.arange(5), np.arange(5))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        cols,
        rows,
        surface_values,
        alpha=0.45,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.scatter(
        cols.ravel(),
        rows.ravel(),
        point_values.ravel(),
        color="tab:red",
        s=45,
        depthshade=True,
        label="DP tabular V(s)",
    )

    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_zlabel("value")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.legend()
    ax.view_init(elev=25, azim=-135)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    alpha = 0.01
    num_rollouts = 1_000
    rollout_len = 100

    dp_values = evaluate_random_policy_with_dp(env, actions, gamma)

    linear_weights = approximate_value_with_td(
        env,
        actions,
        linear_features,
        alpha,
        gamma,
        num_rollouts,
        rollout_len,
    )
    quadratic_weights = approximate_value_with_td(
        env,
        actions,
        quadratic_features,
        alpha,
        gamma,
        num_rollouts,
        rollout_len,
    )

    linear_values = value_grid(linear_weights, linear_features)
    quadratic_values = value_grid(quadratic_weights, quadratic_features)

    plot_surface_and_points(
        linear_values,
        dp_values,
        "Linear approximation vs DP tabular V",
        "linear_features_vs_dp.png",
    )
    plot_surface_and_points(
        quadratic_values,
        dp_values,
        "Quadratic approximation vs DP tabular V",
        "quadratic_features_vs_dp.png",
    )

    print("Linear weights:")
    print(linear_weights)
    print("\nQuadratic weights:")
    print(quadratic_weights)
    print("\nMean absolute difference from DP:")
    print(f"linear features:    {np.mean(np.abs(linear_values - dp_values)):.3f}")
    print(f"quadratic features: {np.mean(np.abs(quadratic_values - dp_values)):.3f}")
    print("\nSaved plots:")
    print("linear_features_vs_dp.png")
    print("quadratic_features_vs_dp.png")
