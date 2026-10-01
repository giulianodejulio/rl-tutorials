from output_paths import figure_path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.continuous_control.continuous_control_rollout import PointMass1D
from experiments.policy_gradient.reinforce_continuous_policy import collect_rollout


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amsfonts}",
    })
    weights = np.array([0.0, 0.0, -1.0, 1.0])
    target = 0.8
    sigma = 0.3
    num_steps = 200
    env = PointMass1D()
    trajectory = collect_rollout(
        env, weights, sigma, target, num_steps, np.random.default_rng(42)
    )
    deterministic_env = PointMass1D()
    deterministic = collect_rollout(
        deterministic_env, weights, 0.0, target, num_steps,
        np.random.default_rng(42),
    )

    time = np.arange(num_steps) * env.dt
    velocity = [step["features"][2] for step in trajectory] + [env.velocity]
    reference_velocity = [step["features"][2] for step in deterministic]
    reference_velocity.append(deterministic_env.velocity)
    state_time = np.arange(num_steps + 1) * env.dt
    mean = np.array([step["mean_action"] for step in trajectory])
    actions = np.array([step["action"] for step in trajectory])
    applied = np.clip(actions, -1.0, 1.0)
    rewards = np.array([step["reward"] for step in trajectory])
    reference_rewards = np.array([step["reward"] for step in deterministic])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Policy gaussiana: un rollout, pesi fissi (nessun training)")
    ax = axes[0, 0]
    ax.plot(state_time, velocity, label=r"$\pi_w(\cdot | s)$", color="tab:blue")
    ax.plot(state_time, reference_velocity, label="Policy senza rumore", color="tab:orange")
    ax.axhline(target, color="black", linestyle="--", label=r"$v^{\text{target}}$")
    ax.set(title="Velocita", xlabel="", ylabel="Velocita")
    ax.legend()

    ax = axes[0, 1]
    ax.fill_between(time, mean - sigma, mean + sigma, alpha=0.15,
                    color="tab:blue", label="$\mu\pm\sigma$")
    ax.plot(time, mean, color="tab:blue", label="$\mu$")
    ax.scatter(time, actions, s=9, color="tab:gray", label="Azione campionata")
    clipped = actions != applied
    ax.scatter(time[clipped], applied[clipped], marker="x", color="tab:red",
               label="Azione limitata")
    ax.axhline(1, color="black", linestyle=":")
    ax.axhline(-1, color="black", linestyle=":")
    ax.set(title="Azione e limiti dell'ambiente", xlabel="", ylabel="Accelerazione")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(time + env.dt, rewards, label="Policy gaussiana", color="tab:blue")
    ax.plot(time + env.dt, reference_rewards, label="Policy senza rumore", color="tab:orange")
    ax.set(title="reward " + r"$r$", xlabel=r"$t$", ylabel="")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(time + env.dt, np.cumsum(rewards), label="Policy gaussiana", color="tab:blue")
    ax.plot(time + env.dt, np.cumsum(reference_rewards), label="Policy senza rumore", color="tab:orange")
    ax.set(title="ritorno totale atteso " + r"$\mathbb{E}[G_0]$", xlabel=r"$t$", ylabel="")
    ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)

    path = figure_path('policy_gradient', "gaussian_rollout.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Gaussian return: {rewards.sum():.3f}")
    print(f"Deterministic return: {reference_rewards.sum():.3f}")
    print(f"Clipped actions: {clipped.sum()}/{num_steps}")


if __name__ == "__main__":
    main()
