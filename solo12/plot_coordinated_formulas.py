"""Plot prescribed formulas only; no MuJoCo simulation or learned policy."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from output_paths import figure_path
from solo12.probe_coordinated_actions import coordinated_action


def scientific_except_zero_exponent(x, _):
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent)
    return mantissa if exponent == 0 else f"{mantissa}e{exponent}"


def save(fig, name):
    for ax in fig.axes:
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_except_zero_exponent))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_except_zero_exponent))
        ax.set_xlabel(r"Tempo $t$ (s)")
    path = figure_path("solo12", name)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    source = Path(__file__).resolve().parents[1] / "results" / "solo12" / "data" / "coordinated_actions.npz"
    with np.load(source, allow_pickle=False) as d:
        names, nominal = list(d["joint_names"]), d["reference"]
        frequency, hip, knee = float(d["frequency"]), float(d["hip_sweep"]), float(d["knee_flexion"])
        scale, ramp_duration = float(d["action_scale"]), float(d["ramp_duration"])
    t = np.linspace(0, 2, 1001)
    alpha = np.clip(t/ramp_duration, 0, 1)
    legs = ["FL", "FR", "HL", "HR"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    offsets, phases, actions = {}, {}, {}
    prescribed = np.array([coordinated_action(s, names, scale, frequency) for s in t])
    for leg in legs:
        phase = 2*np.pi*frequency*t + (np.pi if leg in ("FR", "HL") else 0)
        sign = -1 if leg in ("FL", "FR") else 1
        dq_k = sign*knee*np.maximum(0, np.cos(phase))
        dq_h = -hip*np.sin(phase)-dq_k/2
        phases[leg] = phase
        offsets[leg] = np.column_stack((dq_h, dq_k))
        actions[leg] = alpha[:, None]*offsets[leg]/scale
        for j, joint in enumerate(("HFE", "KFE")):
            np.testing.assert_allclose(actions[leg][:, j], prescribed[:, names.index(leg+"_"+joint)], atol=1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Formule prescritte: fase, rampa e offset prima della rampa (nessuna simulazione)")
    ax = axes[0, 0]
    ax.plot(t, phases["FL"], label=r"$\phi_{\mathrm{FL}}=\phi_{\mathrm{HR}}$")
    ax.plot(t, phases["FR"], label=r"$\phi_{\mathrm{FR}}=\phi_{\mathrm{HL}}$")
    ax.set(title=r"Fasi: $\phi_\ell(t)=2\pi ft+\psi_\ell$", ylabel="Fase (rad)")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    ax.plot(t, alpha, color="black")
    ax.set(title=r"Rampa: $\alpha(t)=\mathrm{clip}(t/t_r,0,1)$", ylabel="Fattore adimensionale")
    for column, joint in enumerate(("HFE", "KFE")):
        ax = axes[1, column]
        for leg, color in zip(legs, colors):
            ax.plot(t, offsets[leg][:, column], color=color, label=leg)
        ax.set(title=r"Offset prima della rampa: $\Delta q_{\ell,\mathrm{" + joint + r"}}(t)$", ylabel="Offset (rad)")
        ax.legend(fontsize=9)
    save(fig, "coordinated_formulas.png")

    fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
    fig.suptitle("Dall'offset al riferimento: zampa prima, giunto dopo\n"
                 "HAA resta nominale e non è mostrato; HFE blu, KFE arancione")
    for row, leg in enumerate(legs):
        effective = alpha[:, None]*offsets[leg]
        for j, (joint, color) in enumerate(zip(("HFE", "KFE"), ("tab:blue", "tab:orange"))):
            ref = nominal[names.index(leg+"_"+joint)]
            label = rf"${leg},\mathrm{{{joint}}}$"
            axes[row, 0].plot(t, effective[:, j], color=color, label=label)
            axes[row, 1].plot(t, actions[leg][:, j], color=color, label=label)
            axes[row, 2].plot(t, ref+effective[:, j], color=color, label=label)
            axes[row, 2].axhline(ref, color=color, linestyle=":", alpha=0.6)
        axes[row, 0].set(title=rf"{leg}: $\alpha(t)\Delta q_{{\ell,j}}(t)$", ylabel="Offset applicato (rad)")
        axes[row, 1].set(title=rf"{leg}: $a_{{\ell,j}}(t)$", ylabel="Azione normalizzata", ylim=(-1.1, 1.1))
        axes[row, 1].axhline(-1, color="gray", linestyle=":")
        axes[row, 1].axhline(1, color="gray", linestyle=":")
        axes[row, 2].set(title=rf"{leg}: $q_{{\mathrm{{ref}},\ell,j}}(t)$", ylabel="Riferimento (rad)")
        for ax in axes[row]:
            ax.legend(fontsize=8)
    save(fig, "coordinated_references.png")
    print("Validated plotted actions against coordinated_action().")


if __name__ == "__main__":
    main()
