"""Plot full 2D training from saved data only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def main():
    source = Path(__file__).resolve().parents[2] / "results" / "policy_gradient" / "data" / "reinforce_neural_2d_training.npz"
    with np.load(source, allow_pickle=False) as d:
        evaluations, checkpoints = d["evaluation_returns"], d["evaluation_updates"]
        velocities, target = d["evaluation_velocities"], d["target_velocity"]
        batch_returns, norms = d["batch_returns"], d["gradient_norms"]
        alpha = float(d["learning_rate"])
    updates = np.arange(1, len(norms) + 1)
    means = evaluations.mean(axis=1)
    se = evaluations.std(axis=1, ddof=1) / np.sqrt(evaluations.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("REINFORCE a due azioni: training completo con baseline temporale")
    ax = axes[0, 0]
    ax.plot(updates, batch_returns.mean(axis=1), label="Batch prima dell'update")
    ax.errorbar(checkpoints, means, yerr=se, fmt="o-", capsize=3,
                label="Valutazione: media ±1 SE")
    ax.set(title="Ritorni durante il training", xlabel="Aggiornamenti / indice del batch", ylabel="Ritorno")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(updates, alpha * norms, color="tab:green")
    ax.set(title="Ampiezza dei singoli aggiornamenti", xlabel="Indice del batch",
           ylabel="Norma della variazione dei pesi")
    steps = np.arange(velocities.shape[2])
    for j, ax in enumerate(axes[1]):
        for m, color, label in [(0, "tab:blue", "Iniziale"), (-1, "tab:orange", "Finale")]:
            values = velocities[m, :, :, j]
            mean = values.mean(axis=0)
            err = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            ax.plot(steps, mean, color=color, label=label)
            ax.fill_between(steps, mean-err, mean+err, color=color, alpha=0.2)
        ax.axhline(target[j], color="black", linestyle="--", label="Target")
        ax.set(title=f"Velocità asse {'xy'[j]}: media ±1 SE", xlabel="Passo t", ylabel="Velocità")
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    path = figure_path("policy_gradient", "reinforce_neural_2d_training.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    diff = evaluations[-1] - evaluations[0]
    print(f"Saved: {path}")
    print(f"Initial={means[0]:.3f}; final={means[-1]:.3f}; paired improvement={diff.mean():.3f}")
    print(f"Final mean velocity at last observed state={velocities[-1, :, -1].mean(axis=0)}")
    print("Bands are +/-1 SE; one training seed only.")


if __name__ == "__main__":
    main()
