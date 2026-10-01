from output_paths import figure_path, data_path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout
from experiments.policy_gradient.reinforce_batch_gradient import returns_to_go, rollout_gradient
from experiments.policy_gradient.reinforce_time_baseline import (
    estimate_time_baseline,
    rollout_gradient_with_baseline,
)


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts}",
    })
    weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma, target = 0.3, 0.8
    num_steps, num_reference_rollouts, num_rollouts = 200, 100, 100
    env = PointMass1D()
    baseline = estimate_time_baseline(
        env, weights, sigma, target, num_steps, num_reference_rollouts,
        np.random.default_rng(2026),
    )
    rng = np.random.default_rng(42)
    without, with_baseline, returns = [], [], []
    for _ in range(num_rollouts):
        trajectory = collect_rollout(env, weights, sigma, target, num_steps, rng)
        returns.append(returns_to_go(trajectory))
        without.append(rollout_gradient(trajectory, sigma))
        with_baseline.append(rollout_gradient_with_baseline(trajectory, sigma, baseline))
    without = np.array(without)
    with_baseline = np.array(with_baseline)
    returns = np.array(returns)
    std_without = without.std(axis=0, ddof=1)
    std_with = with_baseline.std(axis=0, ddof=1)
    labels = [r"$w_0$: bias", r"$w_1$: posizione", r"$w_2$: velocita", r"$w_3$: target"]
    components = np.arange(4)
    series = [(without, "tab:blue", "Senza baseline", -0.18),
              (with_baseline, "tab:orange", "Con baseline", 0.18)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(r"Baseline temporale: $K=100$ rollout di riferimento, $N=100$ di confronto; pesi fissi")

    ax = axes[0, 0]
    steps = np.arange(num_steps)
    ax.plot(steps, returns[0], color="tab:blue", label=r"Primo rollout: $g_t^{(1)}$")
    ax.plot(steps, baseline, color="black", linestyle="--", label=r"Riferimento separato: $b_t$")
    ax.plot(steps, returns[0] - baseline, color="tab:orange", label=r"Residuo: $g_t^{(1)}-b_t$")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set(title="Il riferimento sottratto al ritorno", xlabel=r"Passo $t$", ylabel="Ritorno / residuo")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    offsets = np.linspace(-0.08, 0.08, num_rollouts)
    for values, color, label, shift in series:
        for j in range(4):
            ax.scatter(j + shift + offsets, values[:, j], s=8, alpha=0.35,
                       color=color, label=label if j == 0 else None)
        ax.scatter(components + shift, values.mean(axis=0), marker="D", s=32,
                   color=color, edgecolors="black", zorder=3)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xticks(components, labels)
    ax.set(title="Gradienti per rollout; rombi = medie", ylabel="Componente del gradiente")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(components - 0.18, std_without, width=0.36, color="tab:blue", label="Senza baseline")
    ax.bar(components + 0.18, std_with, width=0.36, color="tab:orange", label="Con baseline")
    ax.set_xticks(components, labels)
    ax.set(title="Dispersione dei gradienti tra rollout", ylabel="Deviazione standard campionaria")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for values, color, label, shift in series:
        # Standard error of the mean, conditional on this fixed reference baseline.
        ax.errorbar(components + shift, values.mean(axis=0),
                    yerr=values.std(axis=0, ddof=1) / np.sqrt(num_rollouts),
                    fmt="o", capsize=5, color=color, label=label)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xticks(components, labels)
    ax.set(title=r"Media del batch $\pm$ 1 errore standard", ylabel="Componente del gradiente medio")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)

    path = figure_path('policy_gradient', "reinforce_time_baseline.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    np.savez(data_path(path), baseline=baseline, returns=returns,
             gradients_without=without, gradients_with=with_baseline, weights=weights,
             sigma=sigma, target_velocity=target, reference_seed=2026, comparison_seed=42,
             num_reference_rollouts=num_reference_rollouts)
    print(f"Saved: {path}")
    print(f"Mean without: {without.mean(axis=0)}")
    print(f"Mean with:    {with_baseline.mean(axis=0)}")
    print(f"Std without: {std_without}")
    print(f"Std with:    {std_with}")
    print(f"Std reduction (%): {100 * (1 - std_with / std_without)}")
    print("Same comparison rollouts, independent reference data, no weight update.")
    print("Error bars are estimated standard errors, not 95% confidence intervals.")


if __name__ == "__main__":
    main()
