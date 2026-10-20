"""Fixed-protocol evaluation and optional GIFs; never updates policy parameters."""
import argparse
import copy
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from paper.artifacts import resolve_checkpoint
from paper.config import run_path
from paper.env import PaperSolo12Env

COMMANDS = {'stop': [0., 0., 0.], 'x02': [.2, 0., 0.], 'x05': [.5, 0., 0.],
            'x10': [1., 0., 0.], 'reverse': [-.5, 0., 0.],
            'lateral': [0., .5, 0.], 'yaw': [0., 0., .5], 'mixed': [.5, .2, .3]}


def _episode(model, normalizer, env, command, mode, seed, trace=False):
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed, options={'command': command, 'hard_reset': True})
    start = env.data.qpos[0]
    records, targets, states, speeds, masks = [], [], [env.data.qpos.copy()], [env.data.qvel.copy()], []
    for _ in range(env.config['episode_steps']):
        if mode == 'zero':
            action = np.zeros(12)
        else:
            action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=mode == 'mean')
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(info)
        if trace:
            states.append(env.data.qpos.copy())
            speeds.append(env.data.qvel.copy())
            targets.append(env.last_target.copy())
            masks.append(env._contacts()[0])
        if terminated or truncated:
            break
    dt, length = env.control_dt, len(records)
    measured = np.array([r['velocity'] for r in records])
    errors = measured-np.asarray(command)
    row = dict(duration=length*dt, fallen=int(records[-1]['fallen']),
               return_full=sum(r['full_reward'] for r in records),
               tracking_mean=float(np.mean([r['tracking'] for r in records])),
               velocity_error=float(np.mean(np.linalg.norm(errors, axis=1))),
               vx_mean=float(measured[:, 0].mean()), vy_mean=float(measured[:, 1].mean()),
               wz_mean=float(measured[:, 2].mean()),
               vx_rmse=float(np.sqrt(np.mean(errors[:, 0]**2))),
               vy_rmse=float(np.sqrt(np.mean(errors[:, 1]**2))),
               wz_rmse=float(np.sqrt(np.mean(errors[:, 2]**2))),
               slip_mean=float(np.mean([r['slip'] for r in records])),
               material_slip_mean=float(np.mean([r['material_slip'] for r in records])),
               power_mean=float(np.mean([r['power'] for r in records])),
               contact_count=float(np.mean([r['contact_count'] for r in records])),
               reference_delta_rms=float(np.sqrt(np.mean([r['cost_smoothness1'] for r in records]) /
                                                   (12*env.config['reward_weights']['smoothness1']))),
               height_mean=float(np.mean([r['height'] for r in records])),
               displacement_x=float(env.data.qpos[0]-start))
    trajectory = None
    if trace:
        trajectory = dict(qpos=np.array(states), qvel=np.array(speeds), targets=np.array(targets),
                          contacts=np.array(masks), velocity=measured, command=np.asarray(command),
                          control_dt=dt, terrain=env.heights.astype(np.float32),
                          reward_full=np.array([r['full_reward'] for r in records]),
                          kp=env.kp, kd=env.kd, seed=seed)
    return row, trajectory


def evaluate_run(directory, checkpoint=None, quick=False, episodes=10, duration=5., gifs=False,
                 selected_commands=None, randomized=True):
    directory = Path(directory)
    if checkpoint is None:
        checkpoint, metadata = resolve_checkpoint(directory)
    else:
        metadata = json.loads((checkpoint/'metadata.json').read_text())
    c = json.loads((directory/'config.json').read_text())
    c['episode_steps'] = round(duration/c['control_dt']) if not quick else 200
    c['continuation_probability'] = 0.
    labels = selected_commands or (['x02', 'x05', 'yaw'] if quick else list(COMMANDS))
    output = directory/'evaluations'/f"{metadata['timesteps']:012d}"/('quick' if quick else 'full')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Loading SB3 checkpoints reseeds global RNGs. Preserve training RNGs during
    # callback evaluation, including NumPy/Python as well as torch CPU RNG.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            model = PPO.load(checkpoint/'policy.zip', device='cpu')
            conditions = ['nominal'] if quick or not randomized else ['nominal', 'randomized']
            for condition in conditions:
                env = PaperSolo12Env(copy.deepcopy(c), randomize=condition == 'randomized',
                                     reward_scale=1., noise_scale=1.,
                                     rough_enabled=condition == 'randomized')
                norm = VecNormalize.load(checkpoint/'normalization.pkl', DummyVecEnv([lambda: env]))
                norm.training, norm.norm_reward = False, False
                try:
                    for label in labels:
                        for mode in (['mean'] if quick else ['mean', 'stochastic', 'zero']):
                            count = 1 if condition == 'nominal' and mode in ('mean', 'zero') else episodes
                            for i in range(count):
                                seed = 20261019+i
                                trace = not quick and condition == 'nominal' and label in ('x02', 'x05') and mode != 'zero' and i == 0
                                row, trajectory = _episode(model, norm, env, COMMANDS[label], mode, seed, trace)
                                row.update(condition=condition, mode=mode, command=label, episode=i,
                                           seed=seed, rough=int(env.rough))
                                rows.append(row)
                                if trajectory is not None:
                                    path = output/f'trace_{condition}_{label}_{mode}.npz'
                                    np.savez_compressed(path, **trajectory)
                                    if gifs:
                                        render_trace(env, trajectory, path.with_suffix('.gif'), mode)
                finally:
                    norm.close()
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    with (output/'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'protocol.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), timesteps=metadata['timesteps'], trained=metadata['trained'],
        config=c, commands={k: COMMANDS[k] for k in labels}, reward_scale=1., noise_scale=1.,
        conditions=conditions, episodes=episodes, duration=c['episode_steps']*c['control_dt'],
        observation_normalization='saved and frozen', policy_updates=False), indent=2)+'\n')
    if not quick:
        from paper.plot import plot_evaluation, plot_training
        plot_evaluation(output)
        if (directory/'history.csv').exists():
            plot_training(directory)
    else:
        print(f"Fixed evaluation at {metadata['timesteps']:.3e} steps: "
              f"velocity error={np.mean([r['velocity_error'] for r in rows]):.3e}", flush=True)
    return rows


def render_trace(env, trajectory, path, mode):
    import mujoco
    from PIL import Image, ImageDraw
    data = mujoco.MjData(env.model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.2, 120, -20
    frames = []
    with mujoco.Renderer(env.model, height=360, width=640) as renderer:
        for k in range(0, len(trajectory['qpos'])-1, 4):
            data.qpos[:], data.qvel[:] = trajectory['qpos'][k], trajectory['qvel'][k]
            mujoco.mj_forward(env.model, data)
            camera.lookat[:] = data.qpos[:3]+[0, 0, -.05]
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 36), fill='white')
            draw.text((5, 3), f'Paper method / MuJoCo | {mode} | command={trajectory["command"]}', fill='black')
            draw.text((5, 19), f't={k*env.control_dt:.2e} s | dx={data.qpos[0]-trajectory["qpos"][0,0]:.2e} m', fill='black')
            frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(4*env.control_dt*1e3), loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='article')
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--nominal-only', action='store_true')
    parser.add_argument('--commands', nargs='+', choices=list(COMMANDS))
    args = parser.parse_args()
    if args.episodes < 1 or args.duration < .01:
        parser.error('Positive episode count and duration >= 1e-2 required')
    torch.set_num_threads(1)
    directory = run_path(args.run)
    checkpoint, metadata = resolve_checkpoint(directory, args.checkpoint)
    if not metadata['trained']:
        print('This checkpoint is UNTRAINED; outputs are interface checks, not learning results.')
    rows = evaluate_run(directory, checkpoint, episodes=args.episodes, duration=args.duration,
                        gifs=args.gif, selected_commands=args.commands, randomized=not args.nominal_only)
    for condition in sorted(set(r['condition'] for r in rows)):
        selected = [r for r in rows if r['condition'] == condition and r['mode'] == 'mean']
        print(f"{condition}, mean actions: mean velocity error={np.mean([r['velocity_error'] for r in selected]):.3e}, "
              f"falls={sum(r['fallen'] for r in selected)}/{len(selected)}")
    output = directory/"evaluations"/format(metadata["timesteps"], "012d")/"full"
    print(f"Saved: {output}")


if __name__ == '__main__':
    main()
