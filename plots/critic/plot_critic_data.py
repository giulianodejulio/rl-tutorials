from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts}",
    })
    dataset = Path(__file__).resolve().parents[2] / "results" / "critic" / "data" / "critic_dataset.npz"
    if not dataset.exists():
        raise FileNotFoundError(
            f"Missing {dataset}. Run python3 -m experiments.critic.collect_critic_data first."
        )
    with np.load(dataset, allow_pickle=False) as data:
        train_features = data["train_features"]
        train_returns = data["train_returns"]
        validation_features = data["validation_features"]
        validation_returns = data["validation_returns"]
        num_steps = int(data["num_steps"])

    for features, returns in [(train_features, train_returns),
                              (validation_features, validation_returns)]:
        if (features.ndim != 3 or features.shape[1:] != (num_steps, 4)
                or returns.shape != features.shape[:2] or len(features) < 2):
            raise ValueError("Expected features (N, T, 4) and returns (N, T), with N >= 2.")

    steps = np.arange(num_steps)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Dati del critic: policy fissa, nessun critic addestrato")
    ax = axes[0]
    scatter = ax.scatter(
        np.broadcast_to(steps, train_returns.shape).ravel(),
        train_features[:, :, 1].ravel(), c=train_returns.ravel(),
        s=5, cmap="viridis", rasterized=True,
    )
    fig.colorbar(scatter, ax=ax, label=r"Ritorno osservato $g_t^{(i)}$")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set(title="Stati visitati nei rollout di addestramento", xlabel=r"Passo $t$",
           ylabel=r"Errore $e_t=v_t-v^{\mathrm{target}}$")

    ax = axes[1]
    for values, color, label in [(train_returns, "tab:blue", "Addestramento"),
                                  (validation_returns, "tab:orange", "Verifica")]:
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
    ax.set(title="Ritorni: media e +/- 1 deviazione standard", xlabel=r"Passo $t$",
           ylabel=r"Ritorno futuro $g_t$")
    ax.legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    path = figure_path("critic", "critic_data_overview.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")

    selected_steps = sorted({0, num_steps // 4, num_steps // 2, num_steps - 1})
    fig, axes = plt.subplots(1, len(selected_steps), figsize=(15, 4),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Relazione errore-ritorno a tempo fissato; scale locali per pannello")
    for ax, t in zip(axes.flat, selected_steps):
        for features, returns, color, label in [
            (train_features, train_returns, "tab:blue", "Addestramento"),
            (validation_features, validation_returns, "tab:orange", "Verifica"),
        ]:
            ax.scatter(features[:, t, 1], returns[:, t], s=14, alpha=0.6,
                       color=color, label=label)
        ax.set(title=rf"$t={t}$, $h_t={(num_steps-t)/num_steps:.3f}$",
               xlabel=r"Errore di velocita $e_t$", ylabel=r"Ritorno $g_t^{(i)}$")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    path = figure_path("critic", "critic_data_fixed_steps.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")
    print("Bands show rollout standard deviation, not standard error of the mean.")
    print("At t=0 all rollouts have the same state; future returns can still differ.")


if __name__ == "__main__":
    main()
