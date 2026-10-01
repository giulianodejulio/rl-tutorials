from output_paths import figure_path
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from experiments.function_approximation.linear_value_prediction import (
    GridworldEnvironment,
    linear_td_prediction,
    value,
)


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


def td0_prediction_with_rollouts(env, actions, alpha, gamma, num_rollouts, rollout_len):
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


def random_state():
    return random.randrange(5), random.randrange(5)


def linear_value_grid(weights):
    grid = np.zeros((5, 5))

    for row in range(5):
        for col in range(5):
            grid[row, col] = value(weights, (row, col))

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
        label="tabular V(s)",
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
    fig.savefig(figure_path('function_approximation', output_path), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(precision=2, suppress=True)

    env = GridworldEnvironment()
    actions = ["n", "s", "e", "w"]

    gamma = 0.9
    tabular_alpha = 0.05
    linear_alpha = 0.01
    num_rollouts = 1_000
    rollout_len = 100

    dp_values = evaluate_random_policy_with_dp(env, actions, gamma)
    td_values = td0_prediction_with_rollouts(
        env,
        actions,
        tabular_alpha,
        gamma,
        num_rollouts,
        rollout_len,
    )
    weights = linear_td_prediction(
        env,
        actions,
        linear_alpha,
        gamma,
        num_rollouts,
        rollout_len,
    )
    linear_values = linear_value_grid(weights)

    plot_surface_and_points(
        linear_values,
        dp_values,
        "Linear V estimate vs DP tabular V",
        "linear_vs_dp_values.png",
    )
    plot_surface_and_points(
        linear_values,
        td_values,
        "Linear V estimate vs TD tabular V",
        "linear_vs_td_values.png",
    )

    print("Linear weights:")
    print(weights)
    print("\nMean absolute difference from DP:")
    print(f"linear vs DP: {np.mean(np.abs(linear_values - dp_values)):.3f}")
    print(f"TD vs DP:     {np.mean(np.abs(td_values - dp_values)):.3f}")
    print("\nSaved plots:")
    print("linear_vs_dp_values.png")
    print("linear_vs_td_values.png")
