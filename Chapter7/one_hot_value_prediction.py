import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from linear_value_prediction import GridworldEnvironment
from quadratic_value_prediction import evaluate_random_policy_with_dp


def random_state():
    return random.randrange(5), random.randrange(5)


def one_hot_features(state):
    row, col = state
    features = np.zeros(25)
    features[row * 5 + col] = 1.0
    return features


def value(weights, features):
    return float(np.dot(weights, features))


def td_with_one_hot_features(env, actions, alpha, gamma, num_rollouts, rollout_len):
    weights = np.zeros(25)

    for _ in range(num_rollouts):
        state = random_state()

        for _ in range(rollout_len):
            action = random.choice(actions)
            next_state, reward = env.mdp_dynamics(state, action)

            features = one_hot_features(state)
            next_features = one_hot_features(next_state)

            prediction = value(weights, features)
            td_target = reward + gamma * value(weights, next_features)
            td_error = td_target - prediction
            weights += alpha * td_error * features

            state = next_state

    return weights


def value_grid(weights):
    grid = np.zeros((5, 5))

    for row in range(5):
        for col in range(5):
            grid[row, col] = value(weights, one_hot_features((row, col)))

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
    alpha = 0.05
    num_rollouts = 1_000
    rollout_len = 100

    dp_values = evaluate_random_policy_with_dp(env, actions, gamma)
    weights = td_with_one_hot_features(
        env,
        actions,
        alpha,
        gamma,
        num_rollouts,
        rollout_len,
    )
    one_hot_values = value_grid(weights)

    plot_surface_and_points(
        one_hot_values,
        dp_values,
        "One-hot TD approximation vs DP tabular V",
        "one_hot_features_vs_dp.png",
    )

    print("One-hot weights reshaped as a 5x5 value grid:")
    print(one_hot_values)
    print("\nMean absolute difference from DP:")
    print(f"one-hot features: {np.mean(np.abs(one_hot_values - dp_values)):.3f}")
    print("\nSaved plot:")
    print("one_hot_features_vs_dp.png")
