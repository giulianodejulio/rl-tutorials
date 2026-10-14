"""Plot saved neural critic results; no PyTorch, fitting, or rollouts needed."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    with np.load(directory / "critic_dataset.npz", allow_pickle=False) as data:
        returns = data["validation_returns"]
        train_returns = data["train_returns"]
        features = data["validation_features"]
        metadata = {key: data[key] for key in (
            "policy_weights", "sigma", "target_velocity", "num_steps",
            "training_seed", "validation_seed",
        )}
    with np.load(directory / "neural_critic.npz", allow_pickle=False) as model:
        for key, value in metadata.items():
            np.testing.assert_array_equal(model[key], value)
        train_history = model["train_mse"]
        validation_history = model["validation_mse"]
        neural = model["validation_predictions"]
        linear = model["ls_validation_predictions"]
        baseline = model["time_baseline"]
        np.testing.assert_allclose(baseline, train_returns.mean(axis=0))
        np.testing.assert_allclose(linear, features @ model["theta_ls"])
        np.testing.assert_allclose(np.mean((returns - neural)**2),
                                   validation_history[-1], rtol=1e-5)
        linear_train_mse = float(model["ls_train_mse"])
        hidden_size = int(model["hidden_size"])
        learning_rate = float(model["learning_rate"])
    names = ["Rete neurale", "Critic lineare (lstsq)", "Baseline temporale"]
    colors = ["tab:blue", "tab:green", "tab:orange"]
    predictions = [neural, linear, np.broadcast_to(baseline, returns.shape)]
    validation_mses = [np.mean((returns - p)**2) for p in predictions]
    train_mses = [train_history[-1], linear_train_mse,
                  np.mean((train_returns - baseline)**2)]
    steps = np.arange(returns.shape[1])
    epochs = np.arange(len(train_history))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"Critic neurale: 2 ingressi (e, h), {hidden_size} neuroni ReLU, "
                 f"1 uscita; learning rate = {learning_rate:g}\n"
                 "Policy fissa; MSE rispetto ai ritorni Monte Carlo osservati")
    ax = axes[0, 1]
    ax.plot(epochs, train_history, color="tab:blue", label="Rete: training")
    ax.plot(epochs, validation_history, color="tab:purple", label="Rete: validazione")
    ax.axhline(linear_train_mse, color="tab:green", linestyle="--",
               label="lstsq: training (modello diverso)")
    ax.axhline(validation_mses[1], color="tab:green", linestyle=":",
               label="lstsq: validazione")
    ax.set_yscale("log")
    ax.set(title="Apprendimento della rete su dati fissi",
           xlabel="Epoche", ylabel="MSE (scala logaritmica)")
    ax.legend(fontsize=8)

    ax = axes[0, 0]
    positions = np.arange(3)
    for offset, values, label, alpha in [
        (-0.18, train_mses, "Training", 0.45),
        (0.18, validation_mses, "Validazione", 1.0),
    ]:
        bars = ax.bar(positions + offset, values, width=0.36,
                      color=colors, alpha=alpha, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Rete\nneurale", "Critic lineare\n(lstsq)", "Baseline\ntemporale"])
    ax.set(title="MSE globale finale: media su rollout e passi", ylabel="MSE")
    ax.margins(y=0.18)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for name, color, p, mse in zip(names, colors, predictions, validation_mses):
        ax.plot(steps, np.mean((returns - p)**2, axis=0), color=color,
                label=f"{name}: MSE globale = {mse:.4f}")
    ax.set(title="MSE di validazione per passo: media sui rollout",
           xlabel="Passo t", ylabel="MSE")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(steps, returns[0], color="black", label="Ritorno osservato")
    for name, color, p in zip(names, colors, predictions):
        ax.plot(steps, p[0], color=color, label=name)
    ax.set(title="Primo rollout di validazione (esempio)",
           xlabel="Passo t", ylabel="Ritorno / previsione")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    path = figure_path("critic", "neural_critic_training.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
