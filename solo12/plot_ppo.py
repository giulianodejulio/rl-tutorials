"""Plot the first Solo12 PPO experiment, reading saved data only."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from output_paths import figure_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="ppo_first")
    parser.add_argument("--coordinated-reference", action="store_true",
                        help="Compare with the saved open-loop coordinated probe")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "results" / "solo12" / "data" / args.run / "training.npz"
    with np.load(source, allow_pickle=False) as d:
        steps = d["evaluation_steps"]
        target = float(d["target_velocity"])
        dt, horizon = float(d["control_dt"]), int(d["num_steps"])
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        fig.suptitle("PPO Solo12: valutazione a checkpoint di training (un seed)")
        names = dict(stochastic=r"Policy con rumore: $a\sim\pi_w(\cdot\mid o)$",
                     deterministic=r"Policy senza rumore: $a=\mu_w(o)$")
        for label, color in (("stochastic", "tab:blue"), ("deterministic", "tab:orange")):
            name = names[label]
            for ax, key in ((axes[0, 0], "returns"), (axes[0, 1], "mean_vx")):
                values = d[f"{label}_{key}"]
                means = values.mean(axis=1)
                if values.shape[1] > 1:
                    se = values.std(axis=1, ddof=1)/np.sqrt(values.shape[1])
                    ax.errorbar(steps, means, yerr=se, fmt="o-", capsize=3, color=color,
                                label=name + r"; media $\pm\widehat{\mathrm{SE}}$")
                else:
                    ax.plot(steps, means, "o-", color=color, label=name)
            axes[1, 0].plot(steps, d[f"{label}_fallen"].mean(axis=1)*100, "o-",
                            color=color, label=name)
            axes[1, 1].plot(steps, d[f"{label}_lengths"].mean(axis=1)*dt, "o-",
                            color=color, label=name)
        axes[0, 0].axhline(d["baseline_returns"][0], color="gray", linestyle="--",
                           label=r"Offset nulli: $a=\mathbf{0}$")
        axes[0, 1].axhline(d["baseline_mean_vx"][0], color="gray", linestyle="--",
                           label=r"Offset nulli: $a=\mathbf{0}$")
        if args.coordinated_reference:
            reference = source.parent.parent / "coordinated_actions.npz"
            with np.load(reference, allow_pickle=False) as probe:
                for key in ("target_velocity", "control_dt", "action_scale"):
                    if not np.isclose(probe[key], d[key]):
                        raise ValueError(f"Incompatible coordinated reference: {key}")
                if len(probe["zero_time"]) != horizon:
                    raise ValueError("Incompatible coordinated reference: horizon")
                for ax, value in ((axes[0, 0], probe["coordinated_reward"].sum()),
                                  (axes[0, 1], probe["coordinated_vx"].mean())):
                    ax.axhline(value, color="tab:green", linestyle="-.",
                               label="Strategia manuale coordinata (senza rumore)")
        axes[0, 1].axhline(target, color="black", linestyle=":",
                           label=r"Comando: ${}^{\mathcal{I}}v_x^{\mathrm{targ}}$")
        axes[1, 1].axhline(horizon*dt, color="black", linestyle=":",
                           label=r"Orizzonte completo: $T\Delta t_{\mathrm{ctrl}}$")
        for ax, title, ylabel in zip(axes.flat,
                                    [r"Ritorno medio: $\overline{g}_{0,\mathrm{eval}}$",
                                     r"Media delle velocità medie degli episodi: $\langle\bar v_x\rangle_{\mathrm{eval}}$",
                                     r"Episodi terminati per caduta: $100\widehat p_{\mathrm{fall}}$",
                                     r"Durata media degli episodi: $\overline{\tau}_{\mathrm{eval}}$"],
                                    [r"Ritorno $g_0$ (non secondi)",
                                     r"Velocità nel world: ${}^{\mathcal{I}}v_x$ (m/s)",
                                     r"Percentuale di cadute (%)", r"Durata $\tau$ (s)"]):
            ax.set(title=title, xlabel=r"Passi di training raccolti: $n_{\mathrm{env}}$", ylabel=ylabel)
            ax.set_xticks(steps[::max(1, (len(steps)-1)//5)])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2e}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2e}"))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        axes[1, 0].set_ylim(-5, 105)
        axes[1, 1].set_ylim(0, horizon*dt+0.2)
        path = figure_path("solo12", f"{args.run}_training.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved: {path}")
        for label in ("stochastic", "deterministic"):
            print(f"{label}: final return={d[label+'_returns'][-1].mean():.3e}, "
                  f"mean_vx={d[label+'_mean_vx'][-1].mean():.3e}, "
                  f"fall fraction={d[label+'_fallen'][-1].mean():.3e}")


if __name__ == "__main__":
    main()
