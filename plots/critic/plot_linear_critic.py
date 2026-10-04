from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    for filename in ["critic_dataset.npz", "linear_critic.npz"]:
        if not (directory / filename).exists():
            raise FileNotFoundError(f"Missing {directory / filename}; collect data and train the critic first.")
    with np.load(directory / "critic_dataset.npz", allow_pickle=False) as data:
        features = data["validation_features"]
        returns = data["validation_returns"]
        train_returns = data["train_returns"]
        policy_weights = data["policy_weights"]
    with np.load(directory / "linear_critic.npz", allow_pickle=False) as model:
        theta = model["theta"]
        train_mse = model["train_mse"]
        baseline = model["time_baseline"]
        stored_predictions = model["validation_predictions"]
        np.testing.assert_allclose(model["policy_weights"], policy_weights)
    predictions = features @ theta
    # Avoid silently mixing saved outputs from different datasets.
    np.testing.assert_allclose(predictions, stored_predictions)
    np.testing.assert_allclose(baseline, train_returns.mean(axis=0))
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts}",
    })
    steps = np.arange(returns.shape[1])
    critic_mse_by_step = np.mean((returns - predictions)**2, axis=0)
    baseline_mse_by_step = np.mean((returns - baseline)**2, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Critic lineare: policy fissa, target Monte Carlo")
    ax = axes[0, 0]
    ax.plot(np.arange(len(train_mse)), train_mse, color="tab:blue")
    ax.set(title="2 x loss sui dati di addestramento", xlabel="Epoche",
           ylabel=r"$H_\text{GD, train}$")

    ax = axes[0, 1]
    # Mean across validation rollouts, not the exact conditional value function.
    mean_return = returns.mean(axis=0)
    std_return = returns.std(axis=0, ddof=1)
    ax.plot(steps, mean_return, color="black", label=r"Media ritorni di validaz $\overline{g}_{\text{val}}[t]=\frac{1}{N_\text{val}}\sum_{i=0}^{N_\text{val}-1} g_\text{val}[i,t]$")
    ax.fill_between(steps, mean_return - std_return, mean_return + std_return,
                    color="gray", alpha=0.15, label=r"$\overline{g}_{\text{val}}[t]$ +/- 1 dev. standard")
    ax.plot(steps, predictions.mean(axis=0), color="tab:blue", label=r"Media previsioni critic $\frac{1}{N_\text{val}} \sum_{i=0}^{N_\text{val}-1} \hat{v}_\text{val}[i,t]$")
    ax.plot(steps, baseline, color="tab:orange", linestyle="--", label=r"Baseline temporale $b_t$")
    ax.set(title="Medie per passo sui rollout di validazione", xlabel=r"Passo $t$", ylabel="Ritorno")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(steps, returns[0], color="black", label=r"Ritorno osservato $g[0,t]$")
    ax.plot(steps, predictions[0], color="tab:blue", label=r"previsione $\widehat v_\theta(s_t,t)$")
    ax.plot(steps, baseline, color="tab:orange", linestyle="--", label=r"Baseline $b_t$")
    ax.set(title="Primo rollout di validazione", xlabel=r"Passo $t$", ylabel="Ritorno")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(steps, critic_mse_by_step, color="tab:blue",
            label=r"$\mathrm{MSE}_{\text{critic,val}}[t] = \frac{1}{N_{\text{val}}} \sum_{i=0}^{N_{\text{val}}-1}(g_\text{val}[i,t]-\hat{v}_\text{val}[i,t])^2$")
    ax.plot(steps, baseline_mse_by_step, color="tab:orange",
            label=r"$\mathrm{MSE}_{\text{basel,val}}[t] = \frac{1}{N_{\text{val}}} \sum_{i=0}^{N_{\text{val}}-1}(g_\text{val}[i,t]-b_t)^2$")
    ax.set(title="MSE di validazione per passo", xlabel=r"Passo $t$", ylabel="MSE tra rollout")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    path = figure_path("critic", "linear_critic_training.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")

    selected_steps = sorted({0, len(steps) // 4, len(steps) // 2, len(steps) - 1})
    fig, axes = plt.subplots(1, len(selected_steps), figsize=(15, 4),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Critic a tempo fissato: punti di validazione e funzione appresa (scale locali)")
    for ax, t in zip(axes.flat, selected_steps):
        errors = features[:, t, 1]
        ax.scatter(errors, returns[:, t], s=14, alpha=0.5, color="gray", label="Ritorni osservati")
        if np.ptp(errors) > 0:
            grid = np.linspace(errors.min(), errors.max(), 200)
            x = np.column_stack((np.ones_like(grid), grid, grid**2,
                                 np.full_like(grid, features[0, t, 3])))
            ax.plot(grid, x @ theta, color="tab:blue", label=r"$\widehat v_\theta(s,t)$")
        else:
            ax.scatter(errors[0], predictions[0, t], marker="D", color="tab:blue",
                       label=r"$\widehat v_\theta(s,t)$", zorder=3)
        ax.axhline(baseline[t], color="tab:orange", linestyle="--", label=r"$b_t$")
        ax.set(title=rf"$t={t}$", xlabel=r"Errore di velocita $e_t$", ylabel="Ritorno / previsione")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    path = figure_path("critic", "linear_critic_fixed_steps.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
