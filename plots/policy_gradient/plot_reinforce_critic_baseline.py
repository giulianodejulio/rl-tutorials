"""Plot saved fixed-policy gradient comparisons; no new simulations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def main():
    source = (Path(__file__).resolve().parents[2] / "results" /
              "policy_gradient" / "data" / "reinforce_critic_baseline.npz")
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Run experiments.policy_gradient.reinforce_critic_baseline first."
        )
    with np.load(source, allow_pickle=False) as data:
        gradients = data["gradients"]
        batch_size = int(data["batch_size"])
    if gradients.ndim != 3 or gradients.shape[1:] != (3, 4):
        raise ValueError("Expected gradients with shape (rollouts, 3 methods, 4 components).")
    if len(gradients) < 2 or not np.all(np.isfinite(gradients)):
        raise ValueError("At least two finite rollout gradients are required.")
    names = ["Senza baseline", "Baseline temporale", "Baseline critic"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    components = ["Bias", "Posizione", "Velocità", "Target"]
    means = gradients.mean(axis=0)
    stds = gradients.std(axis=0, ddof=1)
    standard_errors = stds / np.sqrt(len(gradients))
    positions = np.arange(4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"REINFORCE a policy fissa: stessi {len(gradients)} rollout per i tre metodi")
    for method, (name, color) in enumerate(zip(names, colors)):
        offset = (method - 1) * 0.24
        axes[0, 0].bar(positions + offset, stds[method], width=0.24,
                       color=color, label=name)
        axes[0, 1].errorbar(positions + offset, means[method],
                            yerr=standard_errors[method], fmt="o", capsize=4,
                            color=color, label=name)
        axes[1, 0].bar(positions + offset, stds[method] / np.sqrt(batch_size),
                       width=0.24, color=color, label=name)
    axes[0, 0].set(title=r"$s_{q,j}$: Dispersione dei gradienti dei singoli rollout",
                   ylabel="Deviazione standard")
    axes[0, 1].set(title=r"$\overline{d}_{q,j}\pm s_{q,j}/\sqrt{1000}$: Media dei gradienti ± 1 errore standard",
                   ylabel="Componente del gradiente medio")
    axes[1, 0].set(title=r"$s_{q,j}/\sqrt{100}$: Dispersione stimata della media di un batch (N=100)",
                   ylabel="Deviazione standard stimata: s / √N")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(positions)
        ax.set_xticklabels(components)
        ax.legend(fontsize=9)

    ax = axes[1, 1]
    totals = np.sum(stds**2, axis=1)
    bars = ax.bar(np.arange(3), totals, color=colors)
    for bar, value in zip(bars, totals):
        ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Senza\nbaseline", "Baseline\ntemporale", "Baseline\ncritic"])
    ax.set(title=r"$\sum_{j=0}^3 s_{q,j}^2$: Dispersione complessiva nelle coordinate dei pesi",
           ylabel="Somma delle varianze delle quattro componenti")
    ax.margins(y=0.15)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    print("Error bars: +/-1 SE, not confidence intervals; batch dispersion is estimated.")
    path = figure_path("policy_gradient", "reinforce_critic_baseline_summary.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Distribuzione empirica dei gradienti dei rollout: stessi bin per metodo")
    for component, ax in enumerate(axes.flat):
        values = gradients[:, :, component]
        bins = np.linspace(values.min(), values.max(), 45)
        for method, (name, color) in enumerate(zip(names, colors)):
            ax.hist(values[:, method], bins=bins, histtype="step", linewidth=1.6,
                    color=color, label=name)
            ax.axvline(means[method, component], color=color, linestyle="--", alpha=0.7)
        ax.set(title=components[component], xlabel="Componente del gradiente del rollout",
               ylabel="Numero di rollout")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    print("Histograms: dashed lines are sample means; all samples included; local scales.")
    path = figure_path("policy_gradient", "reinforce_critic_baseline_distributions.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
