from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from continuous_control_rollout import PointMass1D
from reinforce_continuous_policy import collect_rollout
from reinforce_batch_gradient import returns_to_go, rollout_gradient


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts}",
    })
    weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma = 0.3
    num_rollouts = 100
    num_steps = 200
    learning_rate = 0.001
    rng = np.random.default_rng(42)
    env = PointMass1D()
    gradients = []
    total_returns = []

    for i in range(num_rollouts):
        trajectory = collect_rollout(
            env, weights, sigma, target_velocity=0.8,
            num_steps=num_steps, rng=rng,
        )
        returns = returns_to_go(trajectory)
        gradients.append(rollout_gradient(trajectory, sigma))
        total_returns.append(returns[0])
        if i == 0:
            first_returns = returns
            first_rewards = np.array([step["reward"] for step in trajectory])

    gradients = np.array(gradients)
    total_returns = np.array(total_returns)
    counts = np.arange(1, num_rollouts + 1)
    running_gradient = np.cumsum(gradients, axis=0) / counts[:, None]
    batch_gradient = gradients.mean(axis=0)
    labels = [r"$w_0$ (bias)", r"$w_1$ (posizione)",
              r"$w_2$ (velocita)", r"$w_3$ (target)"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("REINFORCE: 100 rollout, pesi fissi (prima dell'update)")

    ax = axes[0, 0]
    ax.scatter(counts, total_returns, s=13, alpha=0.65,
               label=r"$g_0^{(i)}$", color="tab:blue")
    ax.plot(counts, np.cumsum(total_returns) / counts,
            label="Media progressiva", color="tab:orange")
    ax.axhline(total_returns.mean(), color="black", linestyle="--",
               label=r"$\widehat{J}$ su 100 rollout")
    ax.set(title="Ritorni totali del batch", xlabel="Rollout", ylabel="Ritorno")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    steps = np.arange(num_steps)
    ax.plot(steps, first_returns, color="tab:blue",
            label=r"$g_t=\sum_{k=t}^{T-1}r_{k+1}$")
    ax.plot(steps, np.cumsum(first_rewards), color="tab:orange",
            label=r"$\sum_{k=0}^{t}r_{k+1}$")
    ax.set(title="Primo rollout: futuro e passato", xlabel=r"Passo $t$",
           ylabel="Somma delle ricompense")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    # Deterministic horizontal offsets keep the rollout RNG untouched.
    offsets = np.linspace(-0.15, 0.15, num_rollouts)
    for j, color in enumerate(colors):
        ax.scatter(j + offsets, gradients[:, j], s=9, alpha=0.35, color=color)
    ax.scatter(np.arange(4), batch_gradient, marker="D", color="black",
               s=35, label="Media del batch", zorder=3)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xticks(np.arange(4), labels)
    ax.set(title="Gradienti dei singoli rollout", ylabel="Componente del gradiente")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for j, color in enumerate(colors):
        ax.plot(counts, running_gradient[:, j], color=color, label=labels[j],
                linestyle="--" if j == 3 else "-")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set(title="Media progressiva del gradiente", xlabel="Numero di rollout inclusi",
           ylabel="Componente del gradiente medio")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)

    path = Path(__file__).with_name("reinforce_batch_gradient.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Mean return: {total_returns.mean():.6f}")
    print(f"Return standard deviation: {total_returns.std(ddof=1):.6f}")
    print(f"Batch gradient: {batch_gradient}")
    print(f"Single update: {learning_rate * batch_gradient}")
    print("Progressive means use nested samples, not independent batches.")


if __name__ == "__main__":
    main()
