"""Two-axis extension of PointMass1D, with a scalar total reward."""

import numpy as np


class PointMass2D:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.reset()

    def reset(self):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        return self.observation()

    def observation(self):
        return np.concatenate((self.position, self.velocity))

    def step(self, action, target_velocity):
        action = np.asarray(action, dtype=float)
        target_velocity = np.asarray(target_velocity, dtype=float)
        if action.shape != (2,) or target_velocity.shape != (2,):
            raise ValueError("Action and target velocity must each have two components")
        acceleration = np.clip(action, -1.0, 1.0)
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        error = self.velocity - target_velocity
        reward = -np.sum(error**2) - 0.01 * np.sum(acceleration**2)
        return self.observation(), float(reward)
