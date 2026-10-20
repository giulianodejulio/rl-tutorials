"""First finite-horizon Solo12 locomotion environment, with nominal-pose offsets.

An introductory task, not a reproduction of the SoloRL paper's environment.
"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from solo12.hold_solo12_pose import initialize_pose
from solo12.inspect_solo12 import load_model


class Solo12Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, target_velocity=0.2, num_steps=500, frame_skip=5, action_scale=0.25,
                 smoothness_weight=0.0, action_history=False, slip_weight=0.0):
        super().__init__()
        if num_steps < 1 or frame_skip < 1 or action_scale <= 0:
            raise ValueError("Positive horizon, frame skip and action scale required")
        repository = Path(__file__).resolve().parents[1]
        self.scene = repository.parents[1] / "third_party" / "solo12_mujoco" / "scene.xml"
        self.model, self.data = load_model(self.scene)
        if (self.model.nq, self.model.nv, self.model.nu) != (19, 18, 12):
            raise ValueError("Expected floating-base Solo12 with 12 actuators")
        if smoothness_weight < 0 or not np.isfinite(smoothness_weight):
            raise ValueError("Nonnegative finite smoothness weight required")
        if smoothness_weight > 0 and not action_history:
            raise ValueError("Smoothness reward requires previous action in observation")
        if slip_weight < 0 or not np.isfinite(slip_weight):
            raise ValueError("Nonnegative finite slip weight required")
        self.slip_weight = float(slip_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.action_history = bool(action_history)
        self.previous_action = np.zeros(12)
        self.previous_desired = np.zeros(12)
        self.target_velocity = float(target_velocity)
        self.num_steps, self.frame_skip = int(num_steps), int(frame_skip)
        self.action_scale = float(action_scale)
        self.control_dt = self.model.opt.timestep * self.frame_skip
        self.reference, self.joint_addresses, self.joint_names, self.floor_height = initialize_pose(
            self.model, self.data
        )
        self.initial_qpos = self.data.qpos.copy()
        self.nominal_height = self.initial_qpos[2] - self.floor_height
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.lower_legs = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                           leg+"_LOWER_LEG") for leg in ("FL", "FR", "HL", "HR")]
        self.action_space = spaces.Box(-1, 1, shape=(12,), dtype=np.float32)
        # height(1), base quaternion(4), joint angles(12), qvel(18), command(1), h(1).
        # Optional previous applied action (12) keeps the smoothness task Markov.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(49 if self.action_history else 37,), dtype=np.float32)
        self.t, self.episode_done = 0, False

    def set_penalty_weights(self, smoothness_weight, slip_weight):
        """Update the real environment, including when accessed through Monitor."""
        if not np.isfinite([smoothness_weight, slip_weight]).all() or min(smoothness_weight, slip_weight) < 0:
            raise ValueError("Nonnegative finite penalty weights required")
        if smoothness_weight > 0 and not self.action_history:
            raise ValueError("Smoothness reward requires previous action in observation")
        self.smoothness_weight = float(smoothness_weight)
        self.slip_weight = float(slip_weight)

    def observation(self):
        positions = self.data.qpos[2:].copy()
        positions[0] -= self.floor_height
        observation = np.concatenate((positions, self.data.qvel,
                               [self.target_velocity, (self.num_steps-self.t)/self.num_steps])).astype(np.float32)
        if self.action_history:
            observation = np.concatenate((observation, self.previous_action)).astype(np.float32)
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.ctrl[:] = self.reference
        self.t, self.episode_done = 0, False
        self.previous_action[:] = 0
        self.previous_desired[:] = self.reference
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), {}

    def slip_measure(self):
        """Sum per-leg worst horizontal contact-point squared speed, world frame.

        Geometry is sampled at the end of each control step. Using one maximum
        per leg avoids counting mesh-generated duplicate contacts as extra feet.
        The velocity is that of a material point, not the contact location's
        finite difference, and includes both translation and rotation.
        """
        squared_speed = np.zeros(4)
        contact = np.zeros(4, dtype=bool)
        jac = np.zeros((3, self.model.nv))
        for c in self.data.contact:
            if self.floor_geom not in (c.geom1, c.geom2) or c.dist > 0:
                continue
            geom = c.geom2 if c.geom1 == self.floor_geom else c.geom1
            body = int(self.model.geom_bodyid[geom])
            if body not in self.lower_legs:
                continue
            leg = self.lower_legs.index(body)
            contact[leg] = True
            mujoco.mj_jac(self.model, self.data, jac, None, c.pos, body)
            velocity = jac @ self.data.qvel
            squared_speed[leg] = max(squared_speed[leg], float(velocity[:2] @ velocity[:2]))
        return float(squared_speed.sum()), int(contact.sum())

    def step(self, action):
        if self.episode_done:
            raise RuntimeError("Call reset after episode termination")
        action = np.asarray(action, dtype=float)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Action must contain 12 finite components")
        action = np.clip(action, -1, 1)
        # Offsets around the same nominal pose, NOT cumulative angle increments.
        desired = self.reference + self.action_scale * action
        desired = np.clip(desired, self.model.actuator_ctrlrange[:, 0],
                          self.model.actuator_ctrlrange[:, 1])
        reference_change_sq = float(np.sum((desired-self.previous_desired)**2))
        smoothness_cost = self.smoothness_weight * reference_change_sq
        self.previous_action[:] = action
        self.previous_desired[:] = desired
        self.data.ctrl[:] = desired
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.t += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite Solo12 simulation state")
        height = self.data.qpos[2] - self.floor_height
        upright = self.data.xmat[self.base_id].reshape(3, 3)[2, 2]
        vx, vy, vz = self.data.qvel[:3]
        tracking = np.exp(-((vx-self.target_velocity)**2+vy**2)/0.25)
        posture_cost = 10*(height-self.nominal_height)**2 + 0.2*(1-upright)
        motion_cost = 0.2*vz**2 + 0.05*np.sum(self.data.qvel[3:5]**2)
        effort_cost = 0.01*np.mean(action**2) + 0.0001*np.sum(self.data.actuator_force**2)
        fallen = bool(height < 0.12 or upright < 0.5)
        reward = self.control_dt * (tracking-posture_cost-motion_cost-effort_cost)
        if fallen:
            reward -= 1.0
        base_reward = float(reward)
        slip_sq, slip_contacts = self.slip_measure()
        slip_cost = self.slip_weight * slip_sq
        reward -= self.control_dt * (smoothness_cost + slip_cost)
        # Actual finite-horizon objective: zero value beyond T, as in our 2D task.
        terminated = fallen or self.t >= self.num_steps
        self.episode_done = terminated
        info = dict(vx=float(vx), vy=float(vy), height=float(height), upright=float(upright),
                    tracking=float(tracking), posture_cost=float(posture_cost),
                    motion_cost=float(motion_cost), effort_cost=float(effort_cost), fallen=fallen,
                    desired_positions=desired.copy(), base_reward=base_reward,
                    reference_change_sq=reference_change_sq, smoothness_cost=smoothness_cost,
                    slip_sq=slip_sq, slip_contacts=slip_contacts, slip_cost=slip_cost)
        return self.observation(), float(reward), terminated, False, info

    def close(self):
        pass
