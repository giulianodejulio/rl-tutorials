"""Plot the saved two-action experiment without new rollouts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from output_paths import figure_path


def main():
    source = Path(__file__).resolve().parents[2] / "results" / "policy_gradient" / "data" / "reinforce_neural_2d.npz"
    with np.load(source, allow_pickle=False) as data:
        before = {key: data[f"evaluation_before_{key}"] for key in ("inputs", "means", "actions", "returns")}
        after = {key: data[f"evaluation_after_{key}"] for key in before}
        target = data["target_velocity"]
    steps = np.arange(before["inputs"].shape[1])
    initial, final = before["returns"][:, 0], after["returns"][:, 0]
    differences = final - initial
    colors = ["tab:blue", "tab:orange"]
    labels = ["Prima", "Dopo un update"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Policy a due azioni: valutazione prima e dopo un aggiornamento REINFORCE")
    for j, ax in enumerate(axes[0]):
        for batch, color, label in zip((before, after), colors, labels):
            velocities = batch["inputs"][:, :, 2 + j]
            mean = velocities.mean(axis=0)
            se = velocities.std(axis=0, ddof=1) / np.sqrt(len(velocities))
            ax.plot(steps, mean, color=color, label=label)
            ax.fill_between(steps, mean-se, mean+se, color=color, alpha=0.2)
        ax.axhline(target[j], color="black", linestyle="--", label="Target")
        ax.set(title=f"Velocità asse {'xy'[j]}: media ±1 SE",
               xlabel="Passo t (stato prima dell'azione)", ylabel="Velocità")
        ax.legend(fontsize=9)
    ax = axes[1, 0]
    ax.scatter(initial, final, s=14, alpha=0.6)
    limits = [min(initial.min(), final.min()), max(initial.max(), final.max())]
    ax.plot(limits, limits, "--", color="gray", label="Nessun cambiamento")
    ax.set(title="Confronto appaiato: sopra la diagonale = miglioramento",
           xlabel="Ritorno prima", ylabel="Ritorno dopo")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    ax.hist(differences, bins=25, color="tab:green", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--")
    ax.axvline(differences.mean(), color="black", linestyle=":",
               label=f"Media = {differences.mean():.3f}")
    ax.set(title="Differenza appaiata dei ritorni", xlabel="Ritorno dopo − ritorno prima",
           ylabel="Numero di rollout")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    path = figure_path("policy_gradient", "reinforce_neural_2d_comparison.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Primo rollout di valutazione: azioni su due assi, prima e dopo (stesso rumore)")
    for row, (batch, label) in enumerate(zip((before, after), labels)):
        for j, ax in enumerate(axes[row]):
            ax.plot(steps, batch["means"][0, :, j], color="black", label="Media gaussiana")
            ax.plot(steps, batch["actions"][0, :, j], color="tab:blue", alpha=0.5,
                    linewidth=0.8, label="Azione originale")
            ax.plot(steps, np.clip(batch["actions"][0, :, j], -1, 1),
                    color="tab:orange", alpha=0.8, linewidth=0.8, label="Accelerazione applicata")
            ax.axhline(-1, color="gray", linestyle=":")
            ax.axhline(1, color="gray", linestyle=":")
            ax.set(title=f"{label}: asse {'xy'[j]}", xlabel="Passo t", ylabel="Azione / media")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    path = figure_path("policy_gradient", "reinforce_neural_2d_actions.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")
    for label, batch in zip(labels, (before, after)):
        velocities = batch["inputs"][:, -1, 2:4]
        print(f"{label}: mean velocity at last observed state = {velocities.mean(axis=0)}")
        print(f"{label}: clipped action fractions by axis = {(np.abs(batch['actions']) > 1).mean(axis=(0,1))}")
    print(f"Mean paired improvement={differences.mean():.6f}; "
          f"SE={differences.std(ddof=1)/np.sqrt(len(differences)):.6f}; "
          f"fraction improved={np.mean(differences>0):.3f}")


if __name__ == "__main__":
    main()
