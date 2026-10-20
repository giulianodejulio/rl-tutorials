"""First short PPO training: compare reward, velocity and falls with zero offsets."""

import argparse
from pathlib import Path

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from solo12.solo12_env import Solo12Env


def evaluate(env, model=None, num_episodes=10, deterministic=False):
    # Draw a full episode's noise in advance so pairing survives early falls.
    noise = np.random.default_rng(2026).standard_normal((num_episodes, env.num_steps, 12))
    returns, speeds, falls, lengths = [], [], [], []
    base_returns, reference_delta_rms = [], []
    slip_means, contact_means = [], []
    velocities = np.full((num_episodes, env.num_steps), np.nan)
    heights = np.full_like(velocities, np.nan)
    for i in range(num_episodes):
        obs, _ = env.reset(seed=2026+i)
        total, fallen = 0.0, False
        base_total, change_sq = 0.0, 0.0
        slip_sum, contact_sum = 0.0, 0.0
        for t in range(env.num_steps):
            if model is None:
                action = np.zeros(12)
            else:
                with torch.no_grad():
                    distribution = model.policy.get_distribution(
                        torch.tensor(obs[None], dtype=torch.float32)
                    ).distribution
                    mean = distribution.mean.numpy()[0]
                    std = distribution.stddev.numpy()[0]
                action = mean if deterministic else mean + std * noise[i, t]
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            base_total += info["base_reward"]
            change_sq += info["reference_change_sq"]
            slip_sum += info["slip_sq"]
            contact_sum += info["slip_contacts"]
            velocities[i, t], heights[i, t] = info["vx"], info["height"]
            fallen = info["fallen"]
            if terminated or truncated:
                break
        returns.append(total)
        base_returns.append(base_total)
        reference_delta_rms.append(np.sqrt(change_sq / ((t+1)*12)))
        slip_means.append(slip_sum/(t+1))
        contact_means.append(contact_sum/(t+1))
        speeds.append(np.nanmean(velocities[i]))
        falls.append(fallen)
        lengths.append(t+1)
    return dict(slip_sq_mean=np.array(slip_means), contact_count_mean=np.array(contact_means),
                base_returns=np.array(base_returns),
                reference_delta_rms=np.array(reference_delta_rms), returns=np.array(returns), mean_vx=np.array(speeds),
                fallen=np.array(falls), lengths=np.array(lengths),
                velocities=velocities, heights=heights)


def penalty_scale(steps, ramp_steps):
    """Ten percent initially, linear ramp, then hold the target penalties."""
    return 1.0 if ramp_steps == 0 else float(0.1 + 0.9 * np.clip(steps/ramp_steps, 0, 1))


class EvaluationHistory(BaseCallback):
    def __init__(self, env, curriculum_steps=0):
        super().__init__()
        self.env = env
        self.steps, self.stochastic, self.deterministic = [], [], []
        self.curriculum_steps = curriculum_steps
        self.weight_steps, self.weight_scales = [], []

    def apply_curriculum(self):
        # Hold reward weights fixed throughout collection of each rollout buffer.
        scale = penalty_scale(self.num_timesteps, self.curriculum_steps)
        # env_method reaches the underlying Env through Monitor; set_attr would
        # only attach attributes to the wrapper and leave reward weights intact.
        self.training_env.env_method("set_penalty_weights",
                                     self.env.smoothness_weight * scale,
                                     self.env.slip_weight * scale)
        if not self.weight_steps or self.weight_steps[-1] != self.num_timesteps:
            self.weight_steps.append(self.num_timesteps)
            self.weight_scales.append(scale)

    def record(self):
        stochastic = evaluate(self.env, self.model)
        deterministic = evaluate(self.env, self.model, num_episodes=1, deterministic=True)
        self.steps.append(self.num_timesteps)
        self.stochastic.append(stochastic)
        self.deterministic.append(deterministic)
        print(f"steps={self.num_timesteps}: return={stochastic['returns'].mean():.3e}, "
              f"mean_vx={stochastic['mean_vx'].mean():.3e} m/s, "
              f"falls={stochastic['fallen'].sum()}/{len(stochastic['fallen'])}; "
              f"mean-action return={deterministic['returns'][0]:.3e}", flush=True)

    def _on_training_start(self):
        self.apply_curriculum()
        self.record()

    def _on_rollout_start(self):
        self.apply_curriculum()
        if self.num_timesteps > 0 and self.num_timesteps % 5000 == 0:
            self.record()

    def _on_step(self):
        return True

    def _on_training_end(self):
        self.apply_curriculum()
        self.record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--run", default="ppo_first", help="Output folder name")
    parser.add_argument("--action-history", action="store_true")
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--slip-weight", type=float, default=0.0)
    parser.add_argument("--evaluate-only", action="store_true", help="Measure the saved final policy without training")
    parser.add_argument("--curriculum-steps", type=int, default=0,
                        help="Ramp penalties from 10 percent to target over this many steps; 0 disables")
    args = parser.parse_args()
    if args.curriculum_steps < 0 or args.curriculum_steps % 1000:
        parser.error("curriculum-steps must be zero or a positive multiple of 1000")
    if args.curriculum_steps and args.curriculum_steps > args.steps-1000:
        parser.error("Allow at least one training buffer at the final penalty weights")
    if not np.isfinite(args.slip_weight) or args.slip_weight < 0:
        parser.error("slip-weight must be nonnegative and finite")
    if not np.isfinite(args.smoothness_weight) or args.smoothness_weight < 0:
        parser.error("smoothness-weight must be nonnegative and finite")
    args.action_history = args.action_history or args.smoothness_weight > 0
    if args.steps <= 0 or args.steps % 1000:
        parser.error("steps must be a positive multiple of 1000")
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parents[1] / "results" / "solo12" / "data" / args.run
    if args.evaluate_only:
        with np.load(directory / "training.npz") as config:
            env = Solo12Env(
                action_history=bool(config["action_history"]) if "action_history" in config else False,
                smoothness_weight=float(config["smoothness_weight"]) if "smoothness_weight" in config else 0.0,
                slip_weight=float(config["slip_weight"]) if "slip_weight" in config else 0.0)
        model = PPO.load(directory / "policy.zip", device="cpu")
        arrays = {}
        for label, count, deterministic in (("stochastic", 10, False), ("deterministic", 1, True)):
            arrays.update({label+"_"+k: v for k, v in evaluate(
                env, model, num_episodes=count, deterministic=deterministic).items()})
        np.savez(directory / "final_evaluation.npz", **arrays, evaluation_seed=2026,
                 slip_weight=env.slip_weight, smoothness_weight=env.smoothness_weight,
                 action_history=env.action_history, control_dt=env.control_dt,
                 target_velocity=env.target_velocity)
        env.close()
        print(f"Saved: {directory / 'final_evaluation.npz'}")
        return
    if (directory / "training.npz").exists():
        parser.error(f"Results already exist: {directory}; choose another --run")
    directory.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(smoothness_weight=args.smoothness_weight, action_history=args.action_history,
                      slip_weight=args.slip_weight)
    training_env, evaluation_env = Solo12Env(**env_kwargs), Solo12Env(**env_kwargs)
    check_env(training_env, warn=True)
    baseline = evaluate(evaluation_env, num_episodes=1)
    print(f"Zero offsets: return={baseline['returns'][0]:.3e}, "
          f"mean_vx={baseline['mean_vx'][0]:.3e} m/s, fallen={baseline['fallen'][0]}", flush=True)
    model = PPO(
        "MlpPolicy", Monitor(training_env, str(directory / "training.monitor.csv")),
        seed=42, device="cpu", verbose=0,
        n_steps=1000, batch_size=100, n_epochs=10,
        learning_rate=0.0003, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, normalize_advantage=True,
        policy_kwargs=dict(net_arch=dict(pi=[64], vf=[64]), activation_fn=torch.nn.ReLU,
                           log_std_init=float(np.log(0.3))),
    )
    model.policy.log_std.requires_grad_(False)
    model.set_logger(configure(str(directory / "logs"), ["csv"]))
    history = EvaluationHistory(evaluation_env, curriculum_steps=args.curriculum_steps)
    model.learn(total_timesteps=args.steps, callback=history)
    model.save(directory / "policy")
    arrays = {f"baseline_{key}": value for key, value in baseline.items()}
    for label, records in (("stochastic", history.stochastic), ("deterministic", history.deterministic)):
        arrays.update({f"{label}_{key}": np.array([r[key] for r in records])
                       for key in records[0]})
    path = directory / "training.npz"
    np.savez(path, **arrays, evaluation_steps=np.array(history.steps),
             target_velocity=evaluation_env.target_velocity, control_dt=evaluation_env.control_dt,
             num_steps=evaluation_env.num_steps, action_scale=evaluation_env.action_scale,
             curriculum_steps=args.curriculum_steps, curriculum_initial_scale=0.1,
             weight_steps=np.array(history.weight_steps), weight_scales=np.array(history.weight_scales),
             slip_weight=args.slip_weight,
             smoothness_weight=args.smoothness_weight, action_history=args.action_history,
             sigma=0.3, hidden_size=64, n_steps=1000, batch_size=100, n_epochs=10,
             learning_rate=0.0003, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
             training_seed=42, evaluation_seed=2026, total_timesteps=model.num_timesteps,
             sb3_version=stable_baselines3.__version__)
    print(f"Saved: {path}")
    training_env.close()
    evaluation_env.close()


if __name__ == "__main__":
    main()
