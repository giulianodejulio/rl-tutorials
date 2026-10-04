from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from output_paths import figure_path


def padded_limits(values):
    minimum, maximum = float(values.min()), float(values.max())
    margin = 0.05 * max(maximum - minimum, 0.01)
    return minimum - margin, maximum + margin


def main():
    dataset = Path(__file__).resolve().parents[2] / "results" / "critic" / "data" / "critic_dataset.npz"
    if not dataset.exists():
        raise FileNotFoundError(
            f"Missing {dataset}. Run python3 -m experiments.critic.collect_critic_data first."
        )
    with np.load(dataset, allow_pickle=False) as data:
        features = data["train_features"]
        returns = data["train_returns"]
        num_steps = int(data["num_steps"])
    if (features.ndim != 3 or features.shape[1:] != (num_steps, 4)
            or returns.shape != features.shape[:2] or len(features) == 0):
        raise ValueError("Expected features (N, T, 4) and returns (N, T).")
    if not np.isfinite(features).all() or not np.isfinite(returns).all():
        raise ValueError("Dataset contains non-finite values.")

    errors = features[:, :, 1]
    num_rollouts = len(features)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    fig.suptitle(f"Dati del critic: {num_rollouts} rollout, policy fissa")
    # Each row is one rollout: keep its color throughout the animation.
    colors = plt.get_cmap("turbo")(np.linspace(0, 1, num_rollouts))
    scatter = ax.scatter(errors[:, 0], returns[:, 0], c=colors, s=30,
                         alpha=0.8, edgecolors="none")
    ax.set_xlim(*padded_limits(errors))
    ax.set_ylim(*padded_limits(returns))
    ax.set_xlabel(r"Errore di velocita $e_t = v_t - v^{\mathrm{target}}$")
    ax.set_ylabel(r"Ritorno futuro osservato $g_t^{(i)}$")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.grid(alpha=0.2)
    title = ax.set_title("")
    fig.supxlabel("Un colore per rollout; assi fissi. Evoluzione dei dati, non training.",
                  fontsize=10)

    def update(t):
        scatter.set_offsets(np.column_stack((errors[:, t], returns[:, t])))
        title.set_text(f"t = {t}/{num_steps - 1} | passi rimanenti: {num_steps - t}")
        return scatter, title

    fps = 10
    animation = FuncAnimation(fig, update, frames=range(num_steps),
                              interval=1000 / fps, blit=False, repeat=True)
    path = figure_path("critic", "critic_data_evolution.gif")
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"{num_steps} frames at {fps} fps ({num_steps / fps:.1f} seconds per loop).")
    print("No new rollouts generated and no model trained.")


if __name__ == "__main__":
    main()
