"""Compare saved critics and temporal baseline without training or rollouts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "critic" / "data"
    with np.load(directory / "critic_dataset.npz", allow_pickle=False) as data:
        train_features = data["train_features"]
        train_returns = data["train_returns"]
        features = data["validation_features"]
        returns = data["validation_returns"]
        metadata = {key: data[key] for key in (
            "policy_weights", "feature_names", "sigma", "target_velocity", "num_steps"
        )}

    baseline = train_returns.mean(axis=0)
    train_predictions = [np.broadcast_to(baseline, train_returns.shape)]
    predictions = [np.broadcast_to(baseline, returns.shape)]
    for filename in ("linear_critic.npz", "linear_critic_least_squares.npz"):
        with np.load(directory / filename, allow_pickle=False) as model:
            for key, value in metadata.items():
                np.testing.assert_array_equal(model[key], value)
            predicted = features @ model["theta"]
            np.testing.assert_allclose(predicted, model["validation_predictions"])
            np.testing.assert_allclose(baseline, model["time_baseline"])
            train_predictions.append(train_features @ model["theta"])
            predictions.append(predicted)
            if filename == "linear_critic.npz":
                history = model["train_mse"]
                learning_rate = float(model["learning_rate"])

    names = ["Baseline temporale", "Discesa del gradiente", "Minimi quadrati (lstsq)"]
    colors = ["tab:orange", "tab:blue", "tab:green"]
    train_mse = [np.mean((train_returns - p)**2) for p in train_predictions]
    validation_mse = [np.mean((returns - p)**2) for p in predictions]
    steps = np.arange(returns.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Critic lineare: stessi dati e feature, policy fissa\n"
                 "MSE rispetto ai ritorni Monte Carlo osservati")

    ax = axes[0, 0]
    positions = np.arange(3)
    for offset, values, label, alpha in (
        (-0.18, train_mse, "Training", 0.45),
        (0.18, validation_mse, "Validazione", 1.0),
    ):
        bars = ax.bar(positions + offset, values, width=0.36,
                      color=colors, alpha=alpha, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.4f}",
                        (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Baseline\ntemporale", "Discesa del\ngradiente", "Minimi quadrati\n(lstsq)"])
    ax.set(title="MSE globale: media su rollout e passi", ylabel="MSE")
    ax.margins(y=0.18)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(np.arange(len(history)), history, color=colors[1],
            label=f"Discesa del gradiente (learning rate = {learning_rate:g})")
    ax.axhline(train_mse[2], color=colors[2], linestyle="--",
               label="Minimo training con lstsq")
    ax.axhline(train_mse[0], color=colors[0], linestyle=":",
               label="Baseline temporale: MSE training")
    ax.set(title="Ottimizzazione: distanza dal minimo raggiungibile",
           xlabel="Epoche", ylabel="MSE training (due volte la loss)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for name, color, predicted, mse in zip(names, colors, predictions, validation_mse):
        ax.plot(steps, np.mean((returns - predicted)**2, axis=0),
                color=color, label=f"{name}: MSE globale = {mse:.4f}")
    ax.set(title="MSE di validazione per passo: media sui rollout",
           xlabel="Passo t", ylabel="MSE")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(steps, returns[0], color="black", label="Ritorno osservato")
    for name, color, predicted in zip(names, colors, predictions):
        ax.plot(steps, predicted[0], color=color, label=name)
    ax.set(title="Primo rollout di validazione",
           xlabel="Passo t", ylabel="Ritorno")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    path = figure_path("critic", "linear_critic_comparison.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
