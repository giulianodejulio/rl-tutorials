"""Reconstruct kinematics/contacts from saved states; no training or stepping.

Contact-point speeds use recorded qvel, and are unavailable for the older
manual probe. Contact flags are sampled at control instants, not every physics
step; they do not establish continuous stance or absence of slipping.
"""
import argparse
from pathlib import Path
import numpy as np

LEGS = ('FL', 'FR', 'HL', 'HR')


def reconstruct(env, positions, velocities=None):
    import mujoco
    n = len(positions)
    contacts = np.zeros((n, 4), dtype=bool)
    foot_positions = np.empty((n, 4, 3))
    contact_speed = np.full((n, 4), np.nan)
    nonleg_contact = np.zeros(n, dtype=bool)
    floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    feet = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, leg+'_FOOT') for leg in LEGS]
    for k, qpos in enumerate(positions):
        env.data.qpos[:] = qpos
        env.data.qvel[:] = velocities[k] if velocities is not None else 0
        mujoco.mj_forward(env.model, env.data)
        foot_positions[k] = env.data.xpos[feet]
        speeds = [[] for _ in LEGS]
        for c in env.data.contact:
            if floor not in (c.geom1, c.geom2) or c.dist > 0:
                continue
            geom = c.geom2 if c.geom1 == floor else c.geom1
            body = env.model.geom_bodyid[geom]
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
            leg = next((j for j, prefix in enumerate(LEGS) if name.startswith(prefix+'_')), None)
            if leg is None:
                nonleg_contact[k] = True
                continue
            contacts[k, leg] = True
            if velocities is not None:
                jac = np.zeros((3, env.model.nv))
                mujoco.mj_jac(env.model, env.data, jac, None, c.pos, body)
                # Floor is fixed and horizontal: XY velocity of the material
                # point in contact, not the moving geometric contact location.
                speeds[leg].append(np.linalg.norm((jac @ velocities[k])[:2]))
        for j in range(4):
            if speeds[j]:
                contact_speed[k, j] = max(speeds[j])
    return dict(contacts=contacts, foot_positions=foot_positions,
                contact_speed=contact_speed, nonleg_contact=nonleg_contact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='ppo_long')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'results' / 'solo12'
    output = root / 'data' / args.run / 'locomotion_diagnostics.npz'
    if not args.plot_only:
        from solo12.solo12_env import Solo12Env
        env = Solo12Env()
        arrays = dict(control_dt=env.control_dt, floor_height=env.floor_height,
                      target_velocity=env.target_velocity, reference=env.reference,
                      joint_names=np.array(env.joint_names))
        for mode in ('stochastic', 'mean', 'manual'):
            if mode == 'manual':
                with np.load(root/'data'/'coordinated_actions.npz') as d:
                    qpos = d['coordinated_qpos'].copy()
                    vx = d['coordinated_vx'].copy()
                qvel = None
            else:
                with np.load(root/'data'/args.run/f'replay_{mode}.npz') as d:
                    qpos, qvel, vx = d['qpos'].copy(), d['qvel'].copy(), d['vx'].copy()
                    assert np.isclose(d['control_dt'], env.control_dt)
            result = reconstruct(env, qpos, qvel)
            result.update(qpos=qpos, vx=vx,
                          joint_positions=qpos[:, env.joint_addresses])
            arrays.update({mode+'_'+key: value for key, value in result.items()})
        np.savez(output, **arrays)
        env.close()
        print(f'Saved: {output}')
    plot(output, root/'figures'/f'{args.run}_locomotion.png')


def plot(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    def number(v, _):
        return f'{v:g}' if v == 0 or 1 <= abs(v) < 10 else f'{v:.1e}'
    labels = dict(stochastic='PPO con rumore', mean='PPO senza rumore', manual='Strategia manuale')
    colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')
    with np.load(source) as d:
        dt = float(d['control_dt'])
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
        fig.suptitle('Traiettorie singole: velocità, contatti e movimento dei piedi (nessun nuovo rollout)')
        for row, mode in enumerate(labels):
            positions, contacts = d[mode+'_qpos'], d[mode+'_contacts']
            t = np.arange(len(positions))*dt
            ax = axes[row, 0]
            ax.plot(t[1:], d[mode+'_vx'], lw=1, label='Velocità istantanea')
            ax.axhline(float(d['target_velocity']), color='black', ls=':', label='Comando')
            ax.set(title=labels[mode], ylabel=r'${}^{\mathcal{I}}v_x$ (m/s)')
            ax.legend(fontsize=8)
            ax = axes[row, 1]
            ax.imshow(contacts.T, aspect='auto', origin='lower', interpolation='nearest',
                      extent=(-dt/2, t[-1]+dt/2, -.5, 3.5), cmap='Greys', vmin=0, vmax=1)
            ax.set(yticks=range(4), yticklabels=LEGS, title='Contatto zampa–suolo: nero = contatto')
            ax = axes[row, 2]
            for j, leg in enumerate(LEGS):
                ax.plot(t, d[mode+'_foot_positions'][:, j, 0]-positions[:, 0],
                        color=colors[j], label=leg, lw=1)
            ax.set(title='Posizione del piede rispetto alla base lungo x world',
                   ylabel=r'${}^{\mathcal{I}}p_{\mathrm{piede},x}-{}^{\mathcal{I}}p_{\mathrm{base},x}$ (m)')
            ax.legend(fontsize=8, ncol=4)
            # Exclude the initial settling second, explicitly report window.
            mask = t >= 1
            duty = contacts[mask].mean(axis=0)
            speed = d[mode+'_contact_speed'][mask]
            available = speed[np.isfinite(speed)]
            joint_std = d[mode+'_joint_positions'][mask].std(axis=0)
            diagonal = np.mean(np.all(contacts[mask] == [1, 0, 0, 1], axis=1) |
                               np.all(contacts[mask] == [0, 1, 1, 0], axis=1))
            print(f'{mode}: dx={positions[-1,0]-positions[0,0]:.3e} m; '
                  f'contact fractions (FL FR HL HR)={np.array2string(duty, formatter={"float_kind": lambda x: f"{x:.3e}"})}; '
                  f'all four={np.all(contacts[mask],axis=1).mean():.3e}; '
                  f'exact diagonal pair={diagonal:.3e}; '
                  f'joint SD mean={joint_std.mean():.3e} rad')
            if available.size:
                print(f'  contact-point horizontal speed, t>=1 s: mean={available.mean():.3e}, '
                      f'P90={np.quantile(available,.9):.3e} m/s')
            print(f'  non-leg floor contact={d[mode+"_nonleg_contact"].any()}')
        for ax in axes.flat:
            ax.set_xlabel('Tempo t (s)')
            ax.xaxis.set_major_formatter(FuncFormatter(number))
            if ax not in axes[:, 1]:
                ax.yaxis.set_major_formatter(FuncFormatter(number))
            ax.grid(alpha=.15)
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f'Saved: {output}')


if __name__ == '__main__':
    main()
