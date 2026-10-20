"""Read saved PPO evaluations/logs; no learning, simulation, or SB3 needed."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def read_csv(path):
    with path.open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows])
            for key in rows[0]}


def save(fig, filename):
    for ax in fig.axes:
        ax.grid(alpha=0.2)
    path = figure_path("ppo", filename)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "ppo" / "data"
    with np.load(directory / "ppo_2d_training.npz", allow_pickle=False) as d:
        steps, returns = d["evaluation_steps"], d["evaluation_returns"]
        velocities, target = d["evaluation_velocities"], d["target_velocity"]
        n_steps, epsilon = int(d["n_steps"]), float(d["clip_range"])
    monitor = read_csv(directory / "training.monitor.csv")
    logs = read_csv(directory / "logs" / "progress.csv")
    mean = returns.mean(axis=1)
    se = returns.std(axis=1, ddof=1) / np.sqrt(returns.shape[1])
    differences = returns[-1] - returns[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("PPO in 2D: valutazione stocastica, sigma fisso; un seed di training")
    ax = axes[0, 0]
    ax.errorbar(steps, mean, yerr=se, fmt="o-", capsize=4,
                label="Valutazione: media ±1 SE")
    ax.plot(np.cumsum(monitor["l"]), monitor["r"], alpha=0.3, linewidth=0.7,
            color="gray", label="Ritorno del singolo episodio di training")
    ax.set(title="Ritorni in funzione dei passi nell'ambiente",
           xlabel="Passi di training (valutazione esclusa)", ylabel="Ritorno totale")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    bins = np.linspace(returns[[0, -1]].min(), returns[[0, -1]].max(), 28)
    for index, label, color in [(0, "Iniziale", "tab:blue"), (-1, "Finale", "tab:orange")]:
        ax.hist(returns[index], bins=bins, histtype="step", linewidth=1.7,
                color=color, label=f"{label}: media={mean[index]:.3f}")
    ax.set(title="Distribuzioni dei ritorni di valutazione", xlabel="Ritorno", ylabel="Numero di episodi")
    ax.legend(fontsize=9)
    for j, ax in enumerate(axes[1]):
        for index, label, color in [(0, "Iniziale", "tab:blue"),
                                     (len(steps)//2, "Intermedia", "tab:purple"),
                                     (-1, "Finale", "tab:orange")]:
            values = velocities[index, :, :, j]
            avg = values.mean(axis=0)
            err = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            ax.plot(np.arange(len(avg)), avg, color=color,
                    label=f"{label}: {steps[index]} passi")
            ax.fill_between(np.arange(len(avg)), avg-err, avg+err, color=color, alpha=0.15)
        ax.axhline(target[j], color="black", linestyle="--", label="Target")
        ax.set(title=f"Velocità asse {'xy'[j]}: media ±1 SE", xlabel="Passo t nell'episodio", ylabel="Velocità")
        ax.legend(fontsize=8)
    save(fig, "ppo_2d_training.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle("Confronto appaiato: stessa sequenza di rumore iniziale e finale")
    ax = axes[0]
    ax.scatter(returns[0], returns[-1], s=16, alpha=0.6)
    bounds = [returns[[0, -1]].min(), returns[[0, -1]].max()]
    ax.plot(bounds, bounds, "--", color="gray", label="Ritorni uguali")
    ax.set(title="Sopra la diagonale: miglioramento", xlabel="Ritorno iniziale", ylabel="Ritorno finale")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.hist(differences, bins=25, color="tab:green", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--")
    ax.axvline(differences.mean(), color="black", linestyle=":", label=f"Media={differences.mean():.3f}")
    ax.set(title="Miglioramento del ritorno", xlabel="Finale − iniziale", ylabel="Numero di episodi")
    ax.legend(fontsize=9)
    save(fig, "ppo_2d_paired_evaluation.png")

    # SB3 2.3.2 dumps training metrics at the NEXT rollout's end, before its
    # optimization. Shift the x coordinate back by one buffer for train/*.
    optimization_steps = logs["time/total_timesteps"] - n_steps
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Diagnostica PPO dai log: ottimizzazioni registrate (ultima non presente nel CSV)")
    panels = [
        ("train/approx_kl", "Variazione della policy: KL approssimata", "KL approssimata"),
        ("train/clip_fraction", "Rapporti fuori da [1-epsilon, 1+epsilon]", "Frazione dei campioni"),
        ("train/value_loss", "Loss del critic su target aggiornati a ogni raccolta", "MSE sui target del critic"),
        ("train/explained_variance", "Varianza spiegata dal critic nel buffer raccolto", "Varianza spiegata"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        mask = np.isfinite(logs[key])
        ax.plot(optimization_steps[mask], logs[key][mask])
        ax.set(title=title, xlabel="Passi di training del buffer associato", ylabel=ylabel)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title(f"Rapporti fuori da [{1-epsilon:g}, {1+epsilon:g}]")
    axes[1, 1].axhline(0, color="gray", linestyle=":")
    axes[1, 1].axhline(1, color="gray", linestyle=":")
    save(fig, "ppo_2d_diagnostics.png")
    print(f"Return: {mean[0]:.3f} -> {mean[-1]:.3f}; "
          f"paired improvement={differences.mean():.3f}")
    print(f"Mean final velocity at last observed step: {velocities[-1, :, -1].mean(axis=0)}")
    print("Bands: +/-1 SE, not training-seed variability. Clip fraction is not the fraction of zero gradients.")


if __name__ == "__main__":
    main()
