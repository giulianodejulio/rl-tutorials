"""Plot saved REINFORCE training history, without training or simulation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def main():
    source = (Path(__file__).resolve().parents[2] / "results" / "policy_gradient"
              / "data" / "reinforce_training_critic_baseline.npz")
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. First run python3 -m "
            "experiments.policy_gradient.reinforce_training_critic_baseline"
        )
    with np.load(source, allow_pickle=False) as data:
        weights = data["weights"]
        gradients = data["gradients"]
        batch_returns = data["batch_returns"]
        evaluation_updates = data["evaluation_updates"]
        evaluation_returns = data["evaluation_returns"]
        critic_mse = data["critic_train_mse"]
        alpha = float(data["learning_rate"])
    updates = np.arange(1, len(gradients) + 1)
    means = evaluation_returns.mean(axis=1)
    se = evaluation_returns.std(axis=1, ddof=1) / np.sqrt(evaluation_returns.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("REINFORCE con critic Monte Carlo: training della policy")
    ax = axes[0, 0]
    ax.plot(updates, batch_returns.mean(axis=1), color="tab:blue",
            label="Ritorno medio batch (prima dell'update)")
    ax.errorbar(evaluation_updates, means, yerr=se, fmt="o-", capsize=3,
                color="tab:orange", label="Valutazione dopo update: media ±1 SE")
    ax.set(title="Ritorni: stessi rumori di valutazione tra policy",
           xlabel="Aggiornamenti completati / indice del batch", ylabel="Ritorno totale")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for j, name in enumerate(["Bias", "Posizione", "Velocità", "Target"]):
        ax.plot(np.arange(len(weights)), weights[:, j] - weights[0, j], label=name)
    ax.set(title="Variazione dei pesi della policy",
           xlabel="Aggiornamenti completati", ylabel="Peso corrente − peso iniziale")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    ax.plot(updates, alpha * np.linalg.norm(gradients, axis=1), color="tab:green")
    ax.set(title="Ampiezza degli aggiornamenti della policy",
           xlabel="Indice del batch m", ylabel="Norma della variazione dei pesi")
    ax = axes[1, 1]
    initial, final = evaluation_returns[[0, -1]]
    lower, upper = min(initial.min(), final.min()), max(initial.max(), final.max())
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    bins = np.linspace(lower, upper, 25)
    for values, color, name in [(initial, "tab:blue", "Iniziale"),
                                (final, "tab:orange", "Finale")]:
        ax.hist(values, bins=bins, histtype="step", linewidth=1.6, color=color,
                label=f"{name}: media = {values.mean():.3f}")
        ax.axvline(values.mean(), color=color, linestyle="--")
    ax.set(title="Distribuzione dei ritorni di valutazione",
           xlabel="Ritorno totale", ylabel="Numero di rollout")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    path = figure_path("policy_gradient", "reinforce_training_critic_baseline.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle("Diagnostica durante il training: policy e dati cambiano a ogni batch")
    axes[0].plot(updates, critic_mse, color="tab:green")
    axes[0].set(title="MSE training del critic su nuovi dati a ogni batch",
                xlabel="Indice del batch m", ylabel="MSE training (non validazione)")
    for j, name in enumerate(["Bias", "Posizione", "Velocità", "Target"]):
        axes[1].plot(updates, gradients[:, j], label=name)
    axes[1].set(title="Componenti del gradiente medio della policy",
                xlabel="Indice del batch m", ylabel="Componente del gradiente")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    path = figure_path("policy_gradient", "reinforce_training_critic_diagnostics.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")
    differences = final - initial
    print(f"Paired mean improvement: {differences.mean():.6f}; "
          f"SE: {differences.std(ddof=1) / np.sqrt(len(differences)):.6f}")
    print("Evaluation error bars are +/-1 SE, not confidence intervals. "
          "They do not measure variability across training seeds.")


if __name__ == "__main__":
    main()
