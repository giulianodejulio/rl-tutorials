import numpy as np


class PointMass1D:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.position = 0.0
        self.velocity = 0.0

    def reset(self):
        self.position = 0.0
        self.velocity = 0.0
        return self.observation()

    def observation(self):
        return np.array([self.position, self.velocity])

    def step(self, action, target_velocity):
        acceleration = np.clip(action, -1.0, 1.0)

        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt

        velocity_error = self.velocity - target_velocity
        action_cost = 0.01 * acceleration * acceleration
        reward = -(velocity_error * velocity_error) - action_cost

        return self.observation(), reward


def linear_policy(observation, target_velocity, weights):
    position, velocity = observation
    features = np.array([1.0, position, velocity, target_velocity])
    action = np.dot(weights, features)
    return float(np.clip(action, -1.0, 1.0))


def rollout(env, weights, target_velocity, num_steps):
    observation = env.reset()
    total_reward = 0.0

    for step in range(num_steps):
        action = linear_policy(observation, target_velocity, weights)
        observation, reward = env.step(action, target_velocity)
        total_reward += reward

        if step % 20 == 0:
            position, velocity = observation
            print(
                f"step={step:3d} "
                f"position={position: .3f} "
                f"velocity={velocity: .3f} "
                f"action={action: .3f} "
                f"reward={reward: .3f}"
            )

    return total_reward


if __name__ == "__main__":
    env = PointMass1D()
    target_velocity = 0.8
    num_steps = 200

    # This hand-written policy accelerates when velocity is below the target.
    weights = np.array([0.0, 0.0, -1.0, 1.0])

    total_reward = rollout(env, weights, target_velocity, num_steps)
    print(f"\nTotal reward: {total_reward:.3f}")
