"""Plot saved neural REINFORCE history; no simulations or PyTorch required."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def probe_means(weights, inputs, hidden_size):
    # Flattened parameter order from Sequential: U, b, a, a0.
    expected = 3 * hidden_size + hidden_size + hidden_size + 1
    if weights.shape[1] != expected:
        raise ValueError("Unexpected saved policy architecture")
    u = weights[:, :3 * hidden_size].reshape(-1, hidden_size, 3)
    b = weights[:, 3 * hidden_size:4 * hidden_size]
    a = weights[:, 4 * hidden_size:5 * hidden_size]
    a0 = weights[:, -1]
    hidden = np.maximum(0, np.einsum("mhj,sj->msh", u, inputs) + b[:, None, :])
    return np.einsum("msh,mh->ms", hidden, a) + a0[:, None]


def save_figure(fig, filename):
    path = figure_path("policy_gradient", filename)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    directory = Path(__file__).resolve().parents[2] / "results" / "policy_gradient" / "data"
    with np.load(directory / "neural_gaussian_policy_training.npz", allow_pickle=False) as d:
        weights = d["weights"]
        norms = d["gradient_norms"]
        batch_returns = d["batch_returns"]
        evaluation_updates = d["evaluation_updates"]
        evaluations = d["evaluation_returns"]
        baselines = d["time_baselines"]
        losses = d["policy_losses"]
        alpha = float(d["learning_rate"])
        sigma = float(d["sigma"])
        target = float(d["target_velocity"])
        hidden_size = int(d["hidden_size"])
        n = int(d["num_rollouts"])
        k = int(d["num_reference_rollouts"])
    if len(weights) != len(norms) + 1:
        raise ValueError("Weight and gradient histories have inconsistent lengths")
    np.testing.assert_allclose(np.linalg.norm(np.diff(weights, axis=0), axis=1),
                               alpha * norms, rtol=1e-3, atol=1e-7)
    updates = np.arange(1, len(norms) + 1)
    means = evaluations.mean(axis=1)
    se = evaluations.std(axis=1, ddof=1) / np.sqrt(evaluations.shape[1])
    initial, final = evaluations[[0, -1]]
    difference = final - initial

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"REINFORCE neurale con baseline temporale: N={n}, K={k}, "
                 f"sigma={sigma:g}, learning rate={alpha:g}")
    ax = axes[0, 0]
    ax.plot(updates, batch_returns.mean(axis=1), label="Batch: prima dell'update")
    ax.errorbar(evaluation_updates, means, yerr=se, fmt="o-", capsize=4,
                color="tab:orange", label="Valutazione: media ±1 SE")
    ax.set(title="Ritorno della policy durante il training",
           xlabel="Aggiornamenti completati / indice del batch", ylabel="Ritorno totale")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    ax.plot(updates, np.linalg.norm(weights[1:] - weights[0], axis=1), color="tab:purple")
    ax.set(title="Spostamento complessivo dei parametri",
           xlabel="Aggiornamenti completati", ylabel="Norma di w corrente − w iniziale")
    ax = axes[1, 0]
    ax.plot(updates, alpha * norms, color="tab:green")
    ax.set(title="Ampiezza dei singoli aggiornamenti",
           xlabel="Indice del batch", ylabel="Norma della variazione dei parametri")
    ax = axes[1, 1]
    bins = np.linspace(min(initial.min(), final.min()), max(initial.max(), final.max()), 26)
    for values, color, name in [(initial, "tab:blue", "Iniziale"), (final, "tab:orange", "Finale")]:
        ax.hist(values, bins=bins, histtype="step", linewidth=1.7, color=color,
                label=f"{name}: media={values.mean():.3f}")
        ax.axvline(values.mean(), color=color, linestyle="--")
    ax.set(title="Ritorni di valutazione iniziali e finali",
           xlabel="Ritorno totale", ylabel="Numero di rollout")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    save_figure(fig, "neural_gaussian_policy_training.png")

    probes = np.array([[0, 0, target], [0, target, target], [2, target, target]])
    mu = probe_means(weights, probes, hidden_size)
    checkpoints = [0, len(norms) // 2, len(norms)]
    colors = ["tab:blue", "tab:purple", "tab:orange"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Policy in stati fissati: media e densità gaussiana prima del clipping")
    for j, (p, v, _) in enumerate(probes):
        ax = axes[0, j]
        ax.plot(np.arange(len(weights)), mu[:, j])
        ax.axhline(target - v, color="gray", linestyle=":", label="Vecchia policy iniziale: target − v")
        ax.set(title=f"p={p:g}, v={v:g}", xlabel="Aggiornamenti completati", ylabel="Media dell'azione")
        ax.legend(fontsize=8)
        ax = axes[1, j]
        action_grid = np.linspace(min(-1.2, mu[checkpoints, j].min() - 4 * sigma),
                                  max(1.2, mu[checkpoints, j].max() + 4 * sigma), 600)
        for m, color in zip(checkpoints, colors):
            density = np.exp(-0.5 * ((action_grid - mu[m, j]) / sigma)**2) / (np.sqrt(2*np.pi)*sigma)
            ax.plot(action_grid, density, color=color, label=f"m={m}, media={mu[m,j]:.3f}")
        ax.axvline(-1, color="gray", linestyle=":")
        ax.axvline(1, color="gray", linestyle=":", label="Limiti del clipping")
        ax.set(xlabel="Azione gaussiana originale", ylabel="Densità")
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    save_figure(fig, "neural_gaussian_policy_evolution.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Diagnostica: dati e policy cambiano a ogni batch")
    ax = axes[0, 0]
    ax.plot(updates, losses)
    ax.set(title="Loss della policy: non è MSE né ritorno",
           xlabel="Indice del batch", ylabel="Loss usata per la backpropagation")
    ax = axes[0, 1]
    steps = np.arange(baselines.shape[1])
    for m, color in zip([0, len(norms)//2, len(norms)-1], colors):
        ax.plot(steps, baselines[m], color=color, label=f"Batch {m+1}")
    ax.set(title="Baseline temporale ristimata con la policy corrente",
           xlabel="Passo t", ylabel="Ritorno medio di riferimento")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    ax.scatter(initial, final, s=12, alpha=0.6)
    limits = [min(initial.min(), final.min()), max(initial.max(), final.max())]
    ax.plot(limits, limits, "--", color="gray", label="Ritorni uguali")
    ax.set(title="Confronto appaiato: stesso rumore per ciascun punto",
           xlabel="Ritorno iniziale", ylabel="Ritorno finale")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    ax.hist(difference, bins=25, color="tab:green", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--")
    ax.axvline(difference.mean(), color="black", linestyle=":",
               label=f"Media={difference.mean():.3f}")
    ax.set(title="Miglioramento appaiato: finale − iniziale",
           xlabel="Differenza di ritorno", ylabel="Numero di rollout")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    save_figure(fig, "neural_gaussian_policy_diagnostics.png")
    print(f"Paired improvement: {difference.mean():.6f}; "
          f"SE={difference.std(ddof=1)/np.sqrt(len(difference)):.6f}")
    print("Evaluation bars: +/-1 SE, not confidence intervals or training-seed variability.")
    print("Fixed probe states may differ from those visited by the policy.")


if __name__ == "__main__":
    main()
