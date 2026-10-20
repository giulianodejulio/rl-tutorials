"""Train the isolated paper-method MuJoCo port. Never imported to start training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from paper.artifacts import prepare_run, resolve_checkpoint, save_checkpoint
from paper.config import curriculum, load_config, run_path
from paper.env import PaperSolo12Env


def make_env(config, rank):
    def create():
        env = PaperSolo12Env(config)
        env.reset(seed=config['seed']+rank)
        return Monitor(env)
    return create


def build_model(env, config):
    c = config
    return PPO('MlpPolicy', env, seed=c['seed'], device='cpu', verbose=0,
               n_steps=c['buffer_size']//c['num_envs'],
               batch_size=c['buffer_size']//c['minibatches'], n_epochs=c['n_epochs'],
               learning_rate=c['learning_rate'], gamma=c['gamma'], gae_lambda=c['gae_lambda'],
               clip_range=c['clip_range'], ent_coef=c['ent_coef'], vf_coef=.5,
               max_grad_norm=c['max_grad_norm'], normalize_advantage=True,
               policy_kwargs=dict(net_arch=dict(pi=c['network'], vf=c['network']),
                                  activation_fn=torch.nn.LeakyReLU,
                                  log_std_init=float(np.log(c['initial_std']))))


class TrainingHistory(BaseCallback):
    def __init__(self, config, directory):
        super().__init__()
        self.config, self.directory = config, directory
        self.rows = []
        self.sums = {}
        self.count = 0
        self.last_recorded = None
        self.keys = ('tracking', 'penalty', 'full_reward', 'velocity_error', 'power',
                     'slip', 'material_slip', 'contact_count', 'height', 'fallen',
                     'cost_clearance', 'cost_slip', 'cost_orientation', 'cost_vertical_velocity',
                     'cost_joint_pose', 'cost_power', 'cost_smoothness1', 'cost_smoothness2')
        self.history_path = directory/'history.csv'
        if self.history_path.exists():
            with self.history_path.open() as stream:
                self.rows = list(csv.DictReader(stream))

    def _flush(self):
        if not self.count:
            return
        c = self.config
        start = self.num_timesteps-self.count
        kc, kn, rough = curriculum(c, start)
        row = dict(steps=self.num_timesteps, reward_scale=kc, noise_scale=kn, rough_stage=int(rough),
                   std_mean=float(self.model.policy.log_std.exp().detach().cpu().mean()))
        row.update({key: value/self.count for key, value in self.sums.items()})
        row['samples'] = self.count
        row['fall_events'] = self.sums.get('fallen', 0)
        write_header = not self.history_path.exists()
        with self.history_path.open('a') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.rows.append(row)
        self.sums, self.count = {}, 0
        if (self.num_timesteps//c['buffer_size']) % 10 == 0:
            print(f"steps={self.num_timesteps:.3e}, kc={kc:.3e}, noise={kn:.3e}, "
                  f"tracking={row['tracking']:.3e}, slip={row['slip']:.3e}, "
                  f"error={row['velocity_error']:.3e}, std={row['std_mean']:.3e}", flush=True)

    def configure_phase(self):
        c = self.config
        kc, kn, rough = curriculum(c, self.num_timesteps)
        # env_method reaches raw Env through Monitor; weights stay fixed per buffer.
        self.training_env.env_method('set_curriculum', kc, kn, rough)
        prefix = 'terrain_' if rough else ''
        self.model.lr_schedule = get_schedule_fn(c[prefix+'learning_rate'])
        self.model.clip_range = get_schedule_fn(c[prefix+'clip_range'])
        self.model.ent_coef = c[prefix+'ent_coef']
        self.model.max_grad_norm = c[prefix+'max_grad_norm']

    def _on_rollout_start(self):
        self._flush()
        self.configure_phase()
        c = self.config
        iteration = self.num_timesteps//c['buffer_size']
        if iteration % c['checkpoint_every'] == 0:
            checkpoint = save_checkpoint(self.model, self.training_env, self.directory,
                                         trained=self.num_timesteps > 0)
            if iteration % c['eval_every'] == 0 and self.last_recorded != self.num_timesteps:
                from paper.evaluate import evaluate_run
                evaluate_run(self.directory, checkpoint, quick=True)
                self.last_recorded = self.num_timesteps

    def _on_step(self):
        for info in self.locals['infos']:
            for key in self.keys:
                self.sums[key] = self.sums.get(key, 0.)+float(info[key])
            self.count += 1
        return True

    def _on_training_end(self):
        self._flush()
        save_checkpoint(self.model, self.training_env, self.directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='article')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--steps', type=int, help='Absolute target number of environment transitions')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--serial', action='store_true', help='DummyVecEnv instead of worker processes')
    args = parser.parse_args()
    directory = run_path(args.run)
    c = json.loads((directory/'config.json').read_text()) if args.resume else load_config(args.config)
    for name, value in (('total_steps', args.steps), ('num_envs', args.num_envs), ('seed', args.seed)):
        if value is not None:
            if args.resume and name != 'total_steps' and value != c[name]:
                parser.error('Do not change seed or worker count on resume')
            c[name] = value
    if c['num_envs'] < 1 or c['buffer_size'] % c['num_envs']:
        parser.error('num-envs must divide buffer_size (30000 by default)')
    if c['total_steps'] <= 0 or c['total_steps'] % c['buffer_size']:
        parser.error('steps must be a positive multiple of buffer_size (30000)')
    torch.set_num_threads(1)
    if not args.resume:
        prepare_run(directory, c)
    else:
        (directory/'resume_target.json').write_text(json.dumps({'total_steps': c['total_steps']})+'\n')
    constructors = [make_env(c, rank) for rank in range(c['num_envs'])]
    raw = DummyVecEnv(constructors) if args.serial or c['num_envs'] == 1 else SubprocVecEnv(constructors, start_method='spawn')
    if args.resume:
        checkpoint, metadata = resolve_checkpoint(directory)
        env = VecNormalize.load(checkpoint/'normalization.pkl', raw)
        env.training, env.norm_reward = True, False
        model = PPO.load(checkpoint/'policy.zip', env=env, device='cpu')
        if model.num_timesteps >= c['total_steps']:
            env.close()
            parser.error('Target already reached; choose a larger --steps')
    else:
        env = VecNormalize(raw, norm_obs=c['normalize_observations'], norm_reward=False,
                           gamma=c['gamma'], clip_obs=10.)
        model = build_model(env, c)
    print(f"Paper-method MuJoCo port; {c['num_envs']} workers, "
          f"buffer={c['buffer_size']}, target={c['total_steps']:.3e}; output={directory}", flush=True)
    print('Full article budget is 3e8 transitions; smaller runs are preliminary.', flush=True)
    # Set the phase before learn() resets environments, especially on resume.
    env.env_method('set_curriculum', *curriculum(c, model.num_timesteps))
    if args.resume and (directory/'history.csv').exists():
        history_path = directory/'history.csv'
        with history_path.open() as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            committed = [row for row in reader if int(row['steps']) <= model.num_timesteps]
        with history_path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(committed)
    callback = TrainingHistory(c, directory)
    try:
        model.learn(total_timesteps=c['total_steps']-model.num_timesteps,
                    reset_num_timesteps=not args.resume, callback=callback)
    except KeyboardInterrupt:
        # Last committed checkpoint is from a completed PPO update; incomplete
        # collection is deliberately discarded rather than mismatching artifacts.
        print('Interrupted. Resume from the last paired checkpoint in latest.json.', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
