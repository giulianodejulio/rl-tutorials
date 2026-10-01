from output_paths import figure_path, data_path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout
from experiments.policy_gradient.reinforce_batch_gradient import rollout_gradient
from experiments.policy_gradient.reinforce_time_baseline import estimate_time_baseline, rollout_gradient_with_baseline
from experiments.policy_gradient.reinforce_training import evaluate_policy


def training_history(use_baseline, initial_weights, sigma, target_velocity,
                     num_steps, num_rollouts, num_reference_rollouts,
                     num_updates, learning_rate, num_evaluation_rollouts):
    weights = initial_weights.copy()
    env = PointMass1D()
    rng = np.random.default_rng(42)
    reference_rng = np.random.default_rng(31415)
    checkpoints = [0]
    evaluations = [evaluate_policy(
        weights, sigma, target_velocity, num_steps, num_evaluation_rollouts
    )]
    label = "With baseline" if use_baseline else "Without baseline"

    # Same updates as the training scripts, recording evaluation checkpoints.
    for m in range(1, num_updates + 1):
        if use_baseline:
            baseline = estimate_time_baseline(
                env, weights, sigma, target_velocity, num_steps,
                num_reference_rollouts, reference_rng,
            )
        gradients = []
        for _ in range(num_rollouts):
            trajectory = collect_rollout(
                env, weights, sigma, target_velocity, num_steps, rng
            )
            if use_baseline:
                gradient = rollout_gradient_with_baseline(trajectory, sigma, baseline)
            else:
                gradient = rollout_gradient(trajectory, sigma)
            gradients.append(gradient)
        weights += learning_rate * np.mean(gradients, axis=0)
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError("Non-finite policy weights")
        if m % 10 == 0 or m == num_updates:
            checkpoints.append(m)
            evaluations.append(evaluate_policy(
                weights, sigma, target_velocity, num_steps, num_evaluation_rollouts
            ))
            print(f"{label}: update={m}, mean={evaluations[-1].mean():.6f}", flush=True)
    return np.array(checkpoints), np.array(evaluations), weights


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts}",
    })
    initial_weights = np.array([0.0, 0.0, -1.0, 1.0])
    sigma, target_velocity = 0.3, 0.8
    num_steps, num_rollouts, num_reference_rollouts = 200, 100, 100
    num_updates, num_evaluation_rollouts = 50, 200
    learning_rate = 0.00005
    results = []
    for use_baseline in [False, True]:
        results.append(training_history(
            use_baseline, initial_weights, sigma, target_velocity,
            num_steps, num_rollouts, num_reference_rollouts,
            num_updates, learning_rate, num_evaluation_rollouts,
        ))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharey=True)
    fig.suptitle(
        rf"REINFORCE: $M={num_updates}$, $N={num_rollouts}$, $K={num_reference_rollouts}$; un seed di training"
    )
    for result, extra_rollouts, color, label in zip(
        results, [0, num_reference_rollouts], ["tab:blue", "tab:orange"],
        ["Senza baseline", "Con baseline temporale"],
    ):
        checkpoints, evaluation_returns, _ = result
        means = evaluation_returns.mean(axis=1)
        axes[0].plot(checkpoints, means, "o-", color=color, label=label)
        cost = checkpoints * (num_rollouts + extra_rollouts)
        axes[1].plot(cost, means, "o-", color=color, label=label)
    axes[0].set(title="Confronto per update", xlabel=r"Batch / update completato $m$",
                ylabel=r"$\widehat{J}_{\mathrm{eval}}(w^{(m)})$")
    axes[1].set(title="Confronto per esperienza raccolta",
                xlabel=r"Rollout per apprendere: $mN$ oppure $m(N+K)$")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.supxlabel("Valutazione: 200 rollout separati; il costo sull'asse esclude i rollout di valutazione.",
                  fontsize=10)

    path = figure_path('policy_gradient', "reinforce_training_time_baseline.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    np.savez(
        data_path(path), checkpoints=results[0][0],
        evaluation_without=results[0][1], evaluation_with=results[1][1],
        final_weights_without=results[0][2], final_weights_with=results[1][2],
        initial_weights=initial_weights, sigma=sigma, target_velocity=target_velocity,
        num_steps=num_steps, num_rollouts=num_rollouts,
        num_reference_rollouts=num_reference_rollouts, learning_rate=learning_rate,
        training_seed=42, reference_seed=31415, evaluation_seed=2026,
    )
    print(f"Saved: {path}")
    print(f"Learning rollout cost: {num_updates * num_rollouts} without, "
          f"{num_updates * (num_rollouts + num_reference_rollouts)} with baseline")
    print(f"Evaluation rollout cost per method: {len(results[0][0]) * num_evaluation_rollouts}")
    print("Lines join sampled checkpoints; no variability across training seeds is estimated.")


if __name__ == "__main__":
    main()
