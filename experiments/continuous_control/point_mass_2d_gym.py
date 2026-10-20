"""Gymnasium interface for the finite-horizon PointMass2D task."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from experiments.continuous_control.point_mass_2d import PointMass2D


class PointMass2DGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_steps=200, target_velocity=(0.8, 0.4)):
        super().__init__()
        self.physics = PointMass2D()
        self.num_steps = num_steps
        self.target_velocity = np.asarray(target_velocity, dtype=float)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # p_x,p_y,v_x,v_y,target_x,target_y,remaining fraction.
        low = np.array([-np.inf] * 6 + [0.0], dtype=np.float32)
        high = np.array([np.inf] * 6 + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.t = 0

    def observation(self):
        return np.concatenate((self.physics.observation(), self.target_velocity,
                               [(self.num_steps-self.t)/self.num_steps])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.physics.reset()
        self.t = 0
        return self.observation(), {}

    def step(self, action):
        _, reward = self.physics.step(action, self.target_velocity)
        self.t += 1
        # T is the actual end of the return, not a collection timeout.
        terminated = self.t >= self.num_steps
        truncated = False
        return self.observation(), reward, terminated, truncated, {}
