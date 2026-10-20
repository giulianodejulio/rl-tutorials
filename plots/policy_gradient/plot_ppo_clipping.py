"""Illustrate a single PPO contribution and its derivative in the density ratio.

No rollout, training, or numerical differentiation. The advantage is fixed.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import data_path, figure_path


def clipped_contribution(ratio, advantage, epsilon):
    return np.minimum(ratio * advantage,
                      np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)


def contribution_derivative(ratio, advantage, epsilon):
    # A scalar input: the gradient is just the derivative with respect to rho.
    ratio = np.asarray(ratio, dtype=float)
    if advantage == 0:
        return np.zeros_like(ratio)
    boundary = 1 + epsilon if advantage > 0 else 1 - epsilon
    active = ratio < boundary if advantage > 0 else ratio > boundary
    derivative = np.where(active, advantage, 0.0)
    # The classical derivative is undefined at the corner; do not assign zero.
    return np.where(np.isclose(ratio, boundary, atol=1e-12, rtol=0), np.nan, derivative)


def main():
    epsilon = 0.2
    advantages = np.array([2.0, -2.0])
    ratio = np.linspace(0.1, 1.9, 901)
    contributions, derivatives = [], []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("PPO clipping: contributo di un campione, advantage fissato\n"
                 "Obiettivo da massimizzare; epsilon = 0,2")
    for col, advantage in enumerate(advantages):
        values = clipped_contribution(ratio, advantage, epsilon)
        derivative = contribution_derivative(ratio, advantage, epsilon)
        contributions.append(values)
        derivatives.append(derivative)
        boundary = 1 + epsilon if advantage > 0 else 1 - epsilon
        ax = axes[0, col]
        ax.plot(ratio, ratio * advantage, "--", color="tab:blue", label=r"Senza clipping: $\varrho\widehat A$")
        ax.plot(ratio, values, color="tab:orange", linewidth=2, label="PPO: intero minimo")
        ax.set(title=rf"$\widehat A={advantage:g}$", ylabel="Contributo all'obiettivo")
        ax.legend(fontsize=9)
        ax = axes[1, col]
        ax.axhline(advantage, color="tab:blue", linestyle="--", label="Derivata senza clipping")
        # Draw each smooth branch separately to avoid a spurious vertical line.
        for mask in (ratio < boundary - 1e-12, ratio > boundary + 1e-12):
            ax.plot(ratio[mask], derivative[mask], color="tab:orange", linewidth=2,
                    label="Derivata PPO" if ratio[mask][0] < boundary else None)
        ax.scatter([boundary, boundary], [0, advantage], facecolors="white",
                   edgecolors="tab:orange", s=55, zorder=4)
        ax.set(title="Derivata rispetto al rapporto (non ai pesi)",
               ylabel=r"$\mathrm{d}\ell/\mathrm{d}\varrho$",
               ylim=(min(0, advantage)-0.5, max(0, advantage)+0.5))
        ax.legend(fontsize=9)
        for ax in axes[:, col]:
            ax.axvline(1-epsilon, color="gray", linestyle=":")
            ax.axvline(1+epsilon, color="gray", linestyle=":")
            left, right = (boundary, ratio[-1]) if advantage > 0 else (ratio[0], boundary)
            ax.axvspan(left, right, color="tab:green", alpha=0.1)
            ax.set(xlabel=r"Rapporto delle densità $\varrho$", xlim=(ratio[0], ratio[-1]))
            ax.grid(alpha=0.2)
    path = figure_path("policy_gradient", "ppo_clipping.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    np.savez(data_path(path), ratio=ratio, advantages=advantages, epsilon=epsilon,
             contributions=np.array(contributions), derivatives=np.array(derivatives))
    print(f"Saved: {path}")
    print("Open circles: derivative undefined at the corner. Green: flat contribution.")
    print("This plots an objective to maximize, not the negative loss minimized by PyTorch.")


if __name__ == "__main__":
    main()
