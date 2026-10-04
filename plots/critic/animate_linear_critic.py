from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from output_paths import figure_path


def padded_limits(values):
    lower, upper = float(np.min(values)), float(np.max(values))
    margin = 0.05 * max(upper - lower, 0.01)
    return lower - margin, upper + margin


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    for filename in ("critic_dataset.npz", "linear_critic.npz"):
        if not (directory / filename).exists():
            raise FileNotFoundError(f"Missing {directory / filename}; collect data and train the critic first.")
    with np.load(directory / "critic_dataset.npz", allow_pickle=False) as data:
        features = data["validation_features"]
        returns = data["validation_returns"]
        train_returns = data["train_returns"]
        policy_weights = data["policy_weights"]
    with np.load(directory / "linear_critic.npz", allow_pickle=False) as model:
        theta = model["theta"]
        baseline = model["time_baseline"]
        stored_predictions = model["validation_predictions"]
        np.testing.assert_allclose(model["policy_weights"], policy_weights)
    predictions = features @ theta
    np.testing.assert_allclose(predictions, stored_predictions)
    np.testing.assert_allclose(baseline, train_returns.mean(axis=0))
    if not all(np.isfinite(a).all() for a in (features, returns, theta, baseline)):
        raise ValueError("Non-finite dataset or model values.")

    errors = features[:, :, 1]
    num_rollouts, num_steps = returns.shape
    curves = []
    for t in range(num_steps):
        # Show the fitted function only within the observed range at each step.
        grid = np.linspace(errors[:, t].min(), errors[:, t].max(), 200)
        x = np.column_stack((np.ones_like(grid), grid, grid**2,
                             np.full_like(grid, features[0, t, 3])))
        curves.append((grid, x @ theta))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.suptitle(f"Critic appreso: {num_rollouts} rollout di validazione")
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, num_rollouts))
    scatter = ax.scatter(errors[:, 0], returns[:, 0], c=colors, s=28,
                         alpha=0.65, edgecolors="none", label="Ritorni osservati")
    critic_line, = ax.plot([], [], color="tab:blue", linewidth=2,
                          label=r"Critic $\widehat{v}_\theta(s,t)$")
    critic_point, = ax.plot([], [], "D", color="tab:blue", markersize=7)
    baseline_line = ax.axhline(baseline[0], color="tab:orange", linestyle="--",
                               linewidth=2, label=r"Baseline temporale $b_t$")
    ax.set_xlim(*padded_limits(errors))
    ax.set_ylim(*padded_limits(np.concatenate(
        [returns.ravel(), baseline, *[y for _, y in curves]]
    )))
    ax.set_xlabel(r"Errore di velocita $e_t=v_t-v^{\mathrm{target}}$")
    ax.set_ylabel("Ritorno osservato / previsione")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    title = ax.set_title("")
    fig.supxlabel("Colore stabile per rollout; assi fissi. Il tempo avanza, i pesi del critic restano fissi.",
                  fontsize=9)

    def update(t):
        scatter.set_offsets(np.column_stack((errors[:, t], returns[:, t])))
        grid, values = curves[t]
        if grid[0] == grid[-1]:
            critic_line.set_data([], [])
            critic_point.set_data([grid[0]], [values[0]])
        else:
            critic_line.set_data(grid, values)
            critic_point.set_data([], [])
        baseline_line.set_ydata([baseline[t], baseline[t]])
        title.set_text(f"t = {t}/{num_steps - 1} | passi rimanenti: {num_steps - t}")
        return scatter, critic_line, critic_point, baseline_line, title

    fps = 10
    animation = FuncAnimation(fig, update, frames=range(num_steps),
                              interval=1000 / fps, repeat=True, blit=False)
    path = figure_path("critic", "linear_critic_evolution.gif")
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"{num_steps} frames, {num_rollouts} validation rollouts, {fps} fps.")
    print("No new rollouts or training: dataset and fitted critic loaded from disk.")


if __name__ == "__main__":
    main()
