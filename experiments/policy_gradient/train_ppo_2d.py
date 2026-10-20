"""First library PPO experiment on the same finite-horizon 2D dynamics."""

from pathlib import Path

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from experiments.continuous_control.point_mass_2d_gym import PointMass2DGym


def evaluate(model, num_rollouts=100):
    env = PointMass2DGym()
    rng = np.random.default_rng(2026)
    returns = np.zeros(num_rollouts)
    velocities = np.zeros((num_rollouts, env.num_steps, 2))
    for i in range(num_rollouts):
        observation, _ = env.reset()
        for t in range(env.num_steps):
            velocities[i, t] = observation[2:4]
            with torch.no_grad():
                inputs = torch.tensor(observation[None], dtype=torch.float32)
                distribution = model.policy.get_distribution(inputs).distribution
                mean = distribution.mean.numpy()[0]
                sigma = distribution.stddev.numpy()[0]
            action = rng.normal(mean, sigma)
            observation, reward, terminated, truncated, _ = env.step(action)
            returns[i] += reward
        assert terminated and not truncated
    env.close()
    return returns, velocities


class EvaluationHistory(BaseCallback):
    def __init__(self, interval=20000):
        super().__init__()
        self.interval = interval
        self.steps, self.returns, self.velocities = [], [], []

    def record(self):
        returns, velocities = evaluate(self.model)
        self.steps.append(self.num_timesteps)
        self.returns.append(returns)
        self.velocities.append(velocities)
        print(f"steps={self.num_timesteps}: evaluation={returns.mean():.3f}", flush=True)

    def _on_training_start(self):
        self.record()

    def _on_rollout_start(self):
        # Here the preceding PPO optimization has completed.
        if self.num_timesteps > 0 and self.num_timesteps % self.interval == 0:
            self.record()

    def _on_step(self):
        return True

    def _on_training_end(self):
        self.record()


def main():
    torch.set_num_threads(1)
    directory = Path(__file__).resolve().parents[2] / "results" / "ppo"
    data_directory = directory / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    env = PointMass2DGym()
    check_env(env, warn=True)
    env = Monitor(env, str(data_directory / "training.monitor.csv"))
    model = PPO(
        "MlpPolicy", env, seed=42, device="cpu", verbose=0,
        n_steps=2000, batch_size=200, n_epochs=10,
        learning_rate=0.0003, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
        normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16]),
                           activation_fn=torch.nn.ReLU, log_std_init=float(np.log(0.3))),
    )
    # Preserve fixed exploration noise from our previous experiments.
    model.policy.log_std.requires_grad_(False)
    model.set_logger(configure(str(data_directory / "logs"), ["csv"]))
    history = EvaluationHistory()
    model.learn(total_timesteps=100000, callback=history)
    model.save(data_directory / "ppo_2d")
    np.savez(data_directory / "ppo_2d_training.npz",
             evaluation_steps=np.array(history.steps),
             evaluation_returns=np.array(history.returns),
             evaluation_velocities=np.array(history.velocities),
             num_steps=200, target_velocity=np.array([0.8, 0.4]),
             sigma=np.array([0.3, 0.3]), training_seed=42, evaluation_seed=2026,
             total_timesteps=model.num_timesteps, n_steps=2000, batch_size=200,
             n_epochs=10, learning_rate=0.0003, gamma=1.0, gae_lambda=0.95,
             clip_range=0.2, hidden_size=16, sb3_version=stable_baselines3.__version__,
             input_names=np.array(["p_x", "p_y", "v_x", "v_y", "target_x", "target_y", "h"]))
    print(f"Initial={history.returns[0].mean():.3f}; final={history.returns[-1].mean():.3f}")
    print(f"Saved model, evaluation data and logs in {data_directory}")
    env.close()


if __name__ == "__main__":
    main()
