"""Controlled comparison: identical observations, with/without action smoothness."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--curriculum", action="store_true", help="Compare fixed penalties with a linear curriculum")
    group.add_argument("--slip", action="store_true", help="Compare smoothness alone with added slip penalty")
    args = parser.parse_args()
    slip_panels = args.slip or args.curriculum
    root = Path(__file__).resolve().parents[1] / 'results' / 'solo12'
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(('Curriculum delle penalità' if args.curriculum else 'Penalità di scivolamento' if args.slip else 'Penalità sulle variazioni dei riferimenti')
                 + ': stessa osservazione e seed (un training per caso)')
    runs = (('ppo_smooth', 'Solo regolarità', 'tab:blue'),
            ('ppo_smooth_slip', 'Regolarità + scivolamento', 'tab:orange')) if args.slip else (
            ('ppo_history', 'Senza penalità', 'tab:blue'), ('ppo_smooth', 'Con penalità', 'tab:orange'))
    if args.curriculum:
        runs = (('ppo_smooth_slip', 'Pesi costanti', 'tab:blue'),
                ('ppo_curriculum', 'Curriculum', 'tab:orange'))
    keys = ('base_returns', 'mean_vx', 'slip_sq_mean', 'contact_count_mean') if slip_panels else (
            'base_returns', 'mean_vx', 'reference_delta_rms', 'fallen')
    for run, label, color in runs:
        with np.load(root/'data'/run/'training.npz') as d:
            x = d['evaluation_steps']
            assert bool(d['action_history'])
            for mode, style, mode_label in (('stochastic', '-', 'con rumore'),
                                             ('deterministic', '--', 'azioni medie')):
                for ax, key in zip(axes.flat, keys):
                    field = f'{mode}_{key}'
                    final_only = field not in d
                    if final_only:
                        with np.load(root/'data'/run/'final_evaluation.npz') as final:
                            values = final[field][None, :]
                    else:
                        values = d[field]
                    plot_x = x[-1:] if final_only else x
                    if key == 'fallen':
                        values = values*100
                    y = values.mean(axis=1)
                    ax.plot(plot_x, y, ('o' if mode == 'stochastic' else 's') if final_only else style, color=color,
                            label=f'{label}, {mode_label}' + (' (solo finale)' if final_only else ''))
                    if values.shape[1] > 1 and key != 'fallen':
                        se = values.std(axis=1, ddof=1)/np.sqrt(values.shape[1])
                        if final_only:
                            ax.errorbar(plot_x, y, yerr=se, fmt='none', color=color, capsize=4)
                        else:
                            ax.fill_between(plot_x, y-se, y+se, color=color, alpha=.12)
                print(f'{run}, {mode}: base return={d[mode+"_base_returns"][-1].mean():.3e}; '
                      f'new return={d[mode+"_returns"][-1].mean():.3e}; '
                      f'vx={d[mode+"_mean_vx"][-1].mean():.3e}; '
                      f'reference delta RMS={d[mode+"_reference_delta_rms"][-1].mean():.3e}; '
                      f'falls={d[mode+"_fallen"][-1].sum()}')
            axes[0, 1].axhline(float(d['target_velocity']), color='gray', ls=':', lw=.8)
    titles = ('Ritorno con reward originaria (misura comune)',
              'Velocità media lungo x', 'Variazione RMS del riferimento per giunto e passo', 'Episodi terminati per caduta')
    ylabels = (r'$g_{0,\mathrm{base}}$', r'$\bar v_x$ (m/s)',
               r'$\sqrt{\frac{1}{12T}\sum_{t,j}(q_{\mathrm{ref},t,j}-q_{\mathrm{ref},t-1,j})^2}$ (rad)', 'Cadute (%)')
    if slip_panels:
        titles = (titles[0], titles[1], 'Scivolamento: somma dei quadrati delle velocità di contatto',
                  'Numero medio di zampe a contatto (non forze di appoggio)')
        ylabels = (ylabels[0], ylabels[1], r'$\overline{S}$ (m²/s²)', r'$\overline{n}_{\mathrm{contatto}}$')
    def number(v, _):
        return f'{v:g}' if v == 0 or 1 <= abs(v) < 10 else f'{v:.1e}'
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set(title=title, xlabel='Passi di training', ylabel=ylabel)
        ax.xaxis.set_major_formatter(FuncFormatter(number))
        ax.yaxis.set_major_formatter(FuncFormatter(number))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[1, 1].set_ylim((0, 4.2) if slip_panels else (-5, 105))
    output = root/'figures'/('ppo_curriculum_comparison.png' if args.curriculum else 'ppo_slip_comparison.png' if args.slip else 'ppo_smoothness_comparison.png')
    fig.savefig(output, dpi=150)
    plt.close(fig)
    if args.curriculum:
        with np.load(root/'data'/'ppo_curriculum'/'training.npz') as d:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            for key, label in (('smoothness_weight', 'Regolarità'), ('slip_weight', 'Scivolamento')):
                ax.step(d['weight_steps'], d['weight_scales']*float(d[key]), where='post', label=label)
            ax.set(xlabel='Passi di training', ylabel='Peso della penalità',
                   title='Pesi di training; in valutazione sono sempre quelli finali')
            ax.xaxis.set_major_formatter(FuncFormatter(number))
            ax.yaxis.set_major_formatter(FuncFormatter(number))
            ax.grid(alpha=.2)
            ax.legend()
            fig.savefig(root/'figures'/'ppo_curriculum_weights.png', dpi=150)
            plt.close(fig)
    print(f'Saved: {output}; bands = +/- one SE over evaluation episodes, not training seeds')


if __name__ == '__main__':
    main()
