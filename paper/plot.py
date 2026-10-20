"""Read paper artifacts only; no simulation or optimization."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from paper.config import run_path


def number(v, _):
    return f'{v:g}' if v == 0 or 1 <= abs(v) < 10 else f'{v:.1e}'


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def decorate(axes):
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.yaxis.set_major_formatter(FuncFormatter(number))
        ax.legend(fontsize=7)


def plot_evaluation(directory):
    rows = read_csv(directory/'episodes.csv')
    for condition in sorted(set(r['condition'] for r in rows)):
        group = [r for r in rows if r['condition'] == condition]
        labels = list(dict.fromkeys(r['command'] for r in group))
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
        fig.suptitle(f'Paper / MuJoCo: {condition}; fixed full reward, frozen observation normalization')
        keys = ('velocity_error', 'material_slip_mean', 'power_mean', 'fallen', 'duration', 'reference_delta_rms')
        titles = ('Mean velocity tracking error (vx, vy, yaw)', 'Material-point slip index (m²/s²)',
                  'Motor power-loss estimate (W)', 'Fall fraction', 'Episode duration (s)', 'Reference-change RMS (rad)')
        for mode, offset, color in (('mean', -.25, 'tab:blue'), ('stochastic', 0., 'tab:orange'), ('zero', .25, 'gray')):
            for ax, key, title in zip(axes.flat, keys, titles):
                means, errors = [], []
                for label in labels:
                    values = np.array([float(r[key]) for r in group if r['mode'] == mode and r['command'] == label])
                    means.append(values.mean())
                    errors.append(values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0.)
                ax.bar(np.arange(len(labels))+offset, means, width=.24, yerr=errors, capsize=2, color=color, label=mode)
                ax.set(title=title, xticks=np.arange(len(labels)), xticklabels=labels)
                ax.tick_params(axis='x', rotation=30)
        decorate(axes)
        axes[1, 0].set_ylim(0, 1.05)
        fig.savefig(directory/f'evaluation_{condition}.png', dpi=150)
        plt.close(fig)
    for path in directory.glob('trace_*.npz'):
        with np.load(path) as d:
            t = np.arange(len(d['velocity']))*float(d['control_dt'])
            fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
            fig.suptitle(path.stem)
            for j, label in enumerate(('vx body', 'vy body', 'wz body')):
                axes[0, 0].plot(t, d['velocity'][:, j], label=label)
                axes[0, 0].axhline(d['command'][j], color=f'C{j}', ls=':')
            axes[0, 0].set(title='Measured velocities and commands', xlabel='Time (s)')
            axes[0, 1].imshow(d['contacts'].T, aspect='auto', origin='lower', interpolation='nearest',
                              extent=(0, max(t[-1], .01), -.5, 3.5), cmap='Greys', vmin=0, vmax=1)
            axes[0, 1].set(title='Leg-floor contacts', yticks=range(4), yticklabels=('FL','FR','HL','HR'))
            axes[1, 0].plot(t, d['reward_full'], label='Full paper reward')
            axes[1, 0].set(title='Per-step full reward', xlabel='Time (s)')
            for j in (0, 1, 2):
                axes[1, 1].plot(t, d['targets'][:, j], label=f'FL target {j}')
                axes[1, 1].plot(t, d['qpos'][1:, 7+j], ls='--', color=f'C{j}', alpha=.7)
            axes[1, 1].set(title='FL targets and actual angles (dashed)', xlabel='Time (s)', ylabel='rad')
            for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
            fig.savefig(path.with_suffix('.png'), dpi=150)
            plt.close(fig)


def plot_training(directory):
    rows = read_csv(directory/'history.csv')
    if not rows:
        return
    data = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    x = data['steps']
    for ax, keys, title in zip(axes.flat,
            (('reward_scale','noise_scale','rough_stage'), ('tracking','full_reward'), ('velocity_error',),
             ('slip','material_slip'), ('power',), ('std_mean','fall_events')),
            ('Curricula', 'Training commands: tracking and full reward', 'Training velocity error',
             'Foot-frame and material-point slip indices', 'Power-loss estimate (W)', 'Exploration std / fall events per buffer')):
        for key in keys:
            ax.plot(x, data[key], label=key)
        ax.set(title=title, xlabel='Environment transitions')
        ax.xaxis.set_major_formatter(FuncFormatter(number))
    decorate(axes)
    fig.savefig(directory/'training.png', dpi=150)
    plt.close(fig)
    quicks = sorted((directory/'evaluations').glob('*/quick/episodes.csv'))
    if quicks:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for label in ('x02', 'x05', 'yaw'):
            values = [(int(p.parents[1].name), np.mean([float(r['velocity_error']) for r in read_csv(p)
                                                      if r['command']==label])) for p in quicks]
            ax.plot([v[0] for v in values], [v[1] for v in values], 'o-', label=label)
        ax.set(title='Fixed nominal evaluation commands (mean actions)', xlabel='Transitions', ylabel='Mean tracking error')
        ax.xaxis.set_major_formatter(FuncFormatter(number))
        ax.legend()
        ax.grid(alpha=.2)
        fig.savefig(directory/'fixed_evaluation.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='article')
    args = parser.parse_args()
    directory = run_path(args.run)
    if (directory/'history.csv').exists():
        plot_training(directory)
    for path in (directory/'evaluations').glob('*/full/episodes.csv'):
        plot_evaluation(path.parent)
    print(f'Plots saved under {directory}')


if __name__ == '__main__':
    main()
