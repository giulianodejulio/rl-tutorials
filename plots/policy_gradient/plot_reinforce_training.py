from output_paths import figure_path, data_path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout
from experiments.policy_gradient.reinforce_batch_gradient import rollout_gradient
from experiments.policy_gradient.reinforce_training import evaluate_policy


def plot_policy_evolution(weight_history, sigma, target):
    # Fixed probes isolate weight changes from changes in visited states.
    states = [(0.0, 0.0), (2.0, target), (2.0, target + 0.4)]
    last_update = len(weight_history) - 1
    checkpoints = sorted({0, last_update // 2, last_update})
    colors = ["tab:blue", "tab:orange", "tab:green"]
    styles = ["--", "-.", "-"]
    features = np.array([[1.0, p, v, target] for p, v in states])
    means = weight_history @ features.T
    actions = np.linspace(min(-1.2, means.min() - 4 * sigma),
                          max(1.2, means.max() + 4 * sigma), 1200)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(
        rf"Policy a stati fissati: $\pi_{{w^{{(m)}}}}(a\mid s)$, "
        rf"$v^{{\mathrm{{target}}}}={target}$, $\sigma={sigma}$"
    )
    for j, (position, velocity) in enumerate(states):
        ax = axes[0, j]
        for m, color, style in zip(checkpoints, colors, styles):
            mu = means[m, j]
            density = np.exp(-0.5 * ((actions - mu) / sigma)**2) / (
                np.sqrt(2 * np.pi) * sigma
            )
            ax.plot(actions, density, color=color, linestyle=style,
                    label=rf"$m={m}$, $\mu={mu:.3f}$")
        ax.axvline(-1, color="gray", linestyle=":", label=r"Limiti $a=\pm1$")
        ax.axvline(1, color="gray", linestyle=":")
        ax.set(title=rf"Stato fissato: $p={position:g}$, $v={velocity:g}$",
               xlabel=r"Azione campionata $a$", ylabel=r"Densita $\pi_{w^{(m)}}(a\mid s)$")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax = axes[1, j]
        ax.plot(np.arange(last_update + 1), means[:, j], color="tab:purple")
        ax.axhline(means[0, j], color="tab:blue", linestyle="--", label="Media iniziale")
        ax.set(xlabel=r"Update completati $m$", ylabel=r"$\mu_{w^{(m)}}(s)$",
               title="Spostamento della media nello stesso stato")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.legend(fontsize=8)
        print(f"Fixed state ({position}, {velocity}): "
              f"initial mean={means[0, j]:.6f}, final mean={means[-1, j]:.6f}")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.supxlabel(
        "Densita gaussiane prima del clipping; stati scelti come riferimento. "
        "Pannelli inferiori: scale verticali locali.", fontsize=10,
    )
    path = figure_path('policy_gradient', "reinforce_policy_evolution.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts}",
    })
    initial_weights = np.array([0.0, 0.0, -1.0, 1.0])
    weights = initial_weights.copy()
    sigma, target = 0.3, 0.8
    num_steps, num_rollouts, num_updates = 200, 100, 50
    learning_rate = 0.00005
    rng = np.random.default_rng(42)
    env = PointMass1D()
    weight_history = [weights.copy()]
    gradient_history = []
    batch_returns = []
    evaluation_updates = [0]
    evaluations = [evaluate_policy(weights, sigma, target, num_steps, 200)]

    # Same training loop and RNG as reinforce_training.py, with history recording.
    for update in range(num_updates):
        gradients, returns = [], []
        for _ in range(num_rollouts):
            trajectory = collect_rollout(
                env, weights, sigma, target, num_steps, rng
            )
            gradients.append(rollout_gradient(trajectory, sigma))
            returns.append(sum(step["reward"] for step in trajectory))
        gradient = np.mean(gradients, axis=0)
        batch_returns.append(returns)
        gradient_history.append(gradient)
        weights += learning_rate * gradient
        if not np.all(np.isfinite(weights)):
            raise FloatingPointError("Non-finite policy weights")
        weight_history.append(weights.copy())

        if (update + 1) % 10 == 0:
            evaluation_updates.append(update + 1)
            evaluations.append(evaluate_policy(weights, sigma, target, num_steps, 200))
            print(f"Update {update + 1}: evaluation mean={evaluations[-1].mean():.3f}",
                  flush=True)

    weight_history = np.array(weight_history)
    gradient_history = np.array(gradient_history)
    batch_returns = np.array(batch_returns)
    evaluations = np.array(evaluations)
    evaluation_updates = np.array(evaluation_updates)
    evaluation_means = evaluations.mean(axis=1)
    evaluation_std = evaluations.std(axis=1, ddof=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(rf"REINFORCE: $M={num_updates}$, $N={num_rollouts}$, $T={num_steps}$"
                 r", $\alpha=5\cdot10^{-5}$, $\sigma=0.3$")

    ax = axes[0, 0]
    # Batch m uses w^(m-1); evaluation at m uses w^(m), after its update.
    ax.plot(np.arange(1, num_updates + 1), batch_returns.mean(axis=1),
            color="tab:blue", alpha=0.7, label=r"Batch $m$: $\frac{1}{N}\sum_{i=1}^{N}g_0^{(m,i)}$ (prima dell'update)")
    ax.plot(evaluation_updates, evaluation_means, "o-", color="tab:orange",
            label=r"Dopo l'update: $\widehat{J}_{\mathrm{eval}}(w^{(m)})$; $m=0$: iniziale")
    ax.fill_between(evaluation_updates, evaluation_means - evaluation_std,
                    evaluation_means + evaluation_std, color="tab:orange", alpha=0.15,
                    label="Valutazione: media +/- 1 dev. standard")
    ax.set(title="Prestazione prima e dopo l'update", xlabel=r"Indice del batch $m$ ($0$: inizializzazione)", ylabel="Ritorno totale")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    labels = [r"$w_0^{(m)}$: bias", r"$w_1^{(m)}$: posizione", r"$w_2^{(m)}$: velocita", r"$w_3^{(m)}$: target"]
    for j, label in enumerate(labels):
        ax.plot(np.arange(num_updates + 1), weight_history[:, j] - initial_weights[j],
                label=label)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set(title="Variazione dei pesi rispetto all'inizio", xlabel=r"Update completati $m$",
           ylabel=r"$w_j^{(m)}-w_j^{(0)}$")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(np.arange(1, num_updates + 1),
            learning_rate * np.linalg.norm(gradient_history, axis=1), color="tab:green")
    ax.set(title="Dimensione di ciascun update", xlabel=r"Iterazione $m$",
           ylabel=r"$\|w^{(m)}-w^{(m-1)}\|_2=\alpha\|\widehat{\nabla J}_{\mathrm{batch}}^{(m)}\|_2$")

    ax = axes[1, 1]
    bins = np.linspace(evaluations[[0, -1]].min(), evaluations[[0, -1]].max(), 22)
    for values, color, label in [(evaluations[0], "tab:blue", r"Iniziale: $w^{(0)}=w_{\mathrm{init}}$"),
                                  (evaluations[-1], "tab:orange", r"Finale: $w^{(M)}$")]:
        ax.hist(values, bins=bins, alpha=0.45, color=color, label=label)
        ax.axvline(values.mean(), color=color, linestyle="--")
    ax.set(title="Valutazione iniziale e finale: stessi rumori",
           xlabel=r"Ritorno di valutazione $g_{0,\mathrm{eval}}^{(m,i)}$, $m\in\{0,M\}$", ylabel="Numero di rollout")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)

    path = figure_path('policy_gradient', "reinforce_training.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    np.savez(data_path(path), weights=weight_history,
             gradients=gradient_history, batch_returns=batch_returns,
             evaluation_updates=evaluation_updates, evaluation_returns=evaluations,
             learning_rate=learning_rate, sigma=sigma, target_velocity=target,
             num_steps=num_steps, training_seed=42, evaluation_seed=2026)
    print(f"Saved: {path}")
    print(f"Initial mean: {evaluation_means[0]:.6f}; final mean: {evaluation_means[-1]:.6f}")
    print(f"Final weights: {weights}")
    print("Shading is rollout standard deviation, not uncertainty of the mean.")
    plot_policy_evolution(weight_history, sigma, target)


if __name__ == "__main__":
    main()
