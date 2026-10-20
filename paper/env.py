"""Paper locomotion MDP ported to MuJoCo. See SOURCES.md for exact choices.

Primary specification: published equations and tables; official code supplies
nominal pose, torque cap, fall criterion and continuation probability. No old
Solo12Env reward, position servos or time-to-go input are used here.
"""
import copy
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from paper.config import SCENE, load_config

LEGS = ('FL', 'FR', 'HL', 'HR')


def pd_torque(target, q, qd, kp, kd, limit):
    return np.clip(kp*(target-q)-kd*qd, -limit, limit)


def power_loss(torque, velocity):
    friction = 4.77e-2*np.sign(velocity) + 1.35e-4*velocity
    return float(np.sum(friction*velocity + (torque+friction)**2/4.81))


class PaperSolo12Env(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None, randomize=True, reward_scale=0., noise_scale=0.,
                 rough_enabled=False):
        super().__init__()
        self.config = c = copy.deepcopy(config or load_config())
        self.randomize = randomize
        self.reward_scale, self.noise_scale = reward_scale, noise_scale
        self.rough_enabled = rough_enabled
        spec = mujoco.MjSpec.from_file(str(SCENE))
        spec.add_hfield(name='paper_height', nrow=c['terrain_grid'], ncol=c['terrain_grid'],
                        size=[c['terrain_half_size'], c['terrain_half_size'], c['terrain_height'], .1],
                        userdata=np.zeros(c['terrain_grid']**2))
        spec.worldbody.add_geom(name='paper_ground', type=mujoco.mjtGeom.mjGEOM_HFIELD,
                                hfieldname='paper_height', pos=[0, 0, -.4],
                                rgba=[.45, .5, .4, 1], contype=1, conaffinity=1)
        self.model = spec.compile()
        self.model.opt.timestep = c['physics_dt']
        self.data = mujoco.MjData(self.model)
        if (self.model.nq, self.model.nv, self.model.nu) != (19, 18, 12):
            raise ValueError('Expected shared floating-base Solo12 model')
        # Convert the shared position actuators to direct torque motors in memory.
        self.model.actuator_gainprm[:] = 0
        self.model.actuator_gainprm[:, 0] = 1
        self.model.actuator_biasprm[:] = 0
        self.model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        self.model.actuator_ctrllimited[:] = 1
        self.model.actuator_forcelimited[:] = 1
        self.model.actuator_ctrlrange[:] = [-c['torque_limit'], c['torque_limit']]
        self.model.actuator_forcerange[:] = [-c['torque_limit'], c['torque_limit']]
        self.model.dof_damping[6:] = 0
        self.model.dof_frictionloss[6:] = 0
        np.testing.assert_allclose(self.model.actuator_gear[:, 0], 1.)
        joint_ids = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in range(12)]
        self.nominal = np.array([(-.04 if name.startswith(('FL', 'HL')) else .04)
                                if name.endswith('HAA') else
                                (.8 if name.startswith(('FL', 'FR')) else -.8)
                                if name.endswith('HFE') else
                                (-1.6 if name.startswith(('FL', 'FR')) else 1.6) for name in self.names])
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p+'_FOOT') for p in LEGS]
        self.lower = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, p+'_LOWER_LEG') for p in LEGS]
        self.floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.terrain_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'paper_ground')
        self.model.geom_contype[self.floor] = self.model.geom_conaffinity[self.floor] = 0
        self.model.geom_rgba[self.floor, 3] = 0
        self.floor_z = -.4
        self.control_dt = c['control_dt']
        self.substeps = round(self.control_dt/c['physics_dt'])
        if not np.isclose(self.substeps*c['physics_dt'], self.control_dt):
            raise ValueError('Control dt must be an integer multiple of physics dt')
        self.action_space = spaces.Box(-c['action_bound'], c['action_bound'], (12,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (132,), np.float32)
        self.started = False
        self.failed = False
        self.done = True
        self.heights = np.zeros((c['terrain_grid'], c['terrain_grid']))
        self.kp, self.kd = c['kp'], c['kd']
        self.command = np.zeros(3)

    def set_curriculum(self, reward_scale, noise_scale, rough_enabled):
        self.reward_scale = float(np.clip(reward_scale, 0, 1))
        self.noise_scale = float(np.clip(noise_scale, 0, 1))
        self.rough_enabled = bool(rough_enabled)

    def ground_height(self, xy):
        xy = np.asarray(xy)
        size = self.config['terrain_half_size']
        uv = np.clip((xy+size)/(2*size)*(len(self.heights)-1), 0, len(self.heights)-1)
        ij = np.floor(uv).astype(int)
        hi = np.minimum(ij+1, len(self.heights)-1)
        f = uv-ij
        # Grid axes: rows = world y, columns = world x.
        z = ((1-f[..., 0])*(1-f[..., 1])*self.heights[ij[..., 1], ij[..., 0]] +
             f[..., 0]*(1-f[..., 1])*self.heights[ij[..., 1], hi[..., 0]] +
             (1-f[..., 0])*f[..., 1]*self.heights[hi[..., 1], ij[..., 0]] +
             f[..., 0]*f[..., 1]*self.heights[hi[..., 1], hi[..., 0]])
        return self.floor_z + z

    def _terrain(self, force_rough=None):
        c = self.config
        self.rough = (self.rough_enabled and self.np_random.random() < c['terrain_probability']
                      if force_rough is None else bool(force_rough))
        if self.rough:
            n = c['terrain_grid']
            blocks = self.np_random.uniform(1e-3, c['terrain_height'], ((n+1)//2, (n+1)//2))
            self.heights[:] = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)[:n, :n]
        else:
            self.heights[:] = 0
        self.model.hfield_data[:] = (self.heights/c['terrain_height']).ravel()

    def _body_state(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jp, jr, self.base)
        return rotation, rotation.T@(jp@self.data.qvel), rotation.T@(jr@self.data.qvel)

    def _contacts(self):
        contact = np.zeros(4, dtype=bool)
        bad = False
        for point in self.data.contact:
            if self.terrain_geom not in (point.geom1, point.geom2) or point.dist > 0:
                continue
            geom = point.geom2 if point.geom1 == self.terrain_geom else point.geom1
            body = self.model.geom_bodyid[geom]
            if body in self.lower:
                contact[self.lower.index(body)] = True
            else:
                bad = True
        return contact, bad

    def contact_point_slip(self):
        values = np.zeros(4)
        for point in self.data.contact:
            if self.terrain_geom not in (point.geom1, point.geom2) or point.dist > 0:
                continue
            geom = point.geom2 if point.geom1 == self.terrain_geom else point.geom1
            body = self.model.geom_bodyid[geom]
            if body in self.lower:
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jac, None, point.pos, body)
                v = jac@self.data.qvel
                leg = self.lower.index(body)
                values[leg] = max(values[leg], float(v[:2]@v[:2]))
        return float(values.sum())

    def _motor_step(self, target):
        torque = pd_torque(target, self.data.qpos[self.qadr], self.data.qvel[self.vadr],
                           self.kp, self.kd, self.config['torque_limit'])
        loss = power_loss(torque, self.data.qvel[self.vadr])
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)
        return loss

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        c = self.config
        hard = (seed is not None or options.get('hard_reset', False) or not self.started or
                self.failed or self.np_random.random() >= c['continuation_probability'])
        if hard:
            self._terrain(options.get('rough'))
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.qadr] = self.nominal
            mujoco.mj_forward(self.model, self.data)
            feet = self.data.xpos[self.feet]
            shift = np.max(self.ground_height(feet[:, :2])+c['foot_radius']+2e-3-feet[:, 2])
            self.data.qpos[2] += shift
            self.kp, self.kd = c['kp'], c['kd']
            mujoco.mj_forward(self.model, self.data)
            # Bounded settling replaces the upstream unbounded ground-placement loop.
            for _ in range(200):
                self._motor_step(self.nominal)
            self.data.qvel[:] = 0
            self.data.time = 0
            mujoco.mj_forward(self.model, self.data)
            self.last_target = self.nominal.copy()
            self.lastlast_target = self.nominal.copy()
            error = self.last_target-self.data.qpos[self.qadr]
            self.error_history = np.tile(error, (3, 1))
            self.velocity_history = np.zeros((3, 12))
        noise = self.noise_scale if self.randomize else 0.
        self.kp = c['kp'] + noise*self.np_random.uniform(*c['kp_noise'])
        self.kd = c['kd'] + noise*self.np_random.uniform(*c['kd_noise'])
        if 'command' in options:
            self.command = np.asarray(options['command'], dtype=float)
        else:
            self.command = self.noise_scale*self.np_random.uniform(-np.array(c['command_max']), c['command_max'])
        if self.command.shape != (3,) or not np.isfinite(self.command).all():
            raise ValueError('Command must be three finite body-frame velocity components')
        self.t, self.failed, self.done, self.started = 0, False, False, True
        return self.observation(), {'hard_reset': hard, 'command': self.command.copy(), 'rough': self.rough}

    def observation(self):
        rotation, velocity, omega = self._body_state()
        parts = [rotation[2].copy(), velocity, omega, self.command.copy(),
                 self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy(),
                 self.last_target.copy(), self.lastlast_target.copy()]
        if self.randomize:
            for j, scale in zip((0, 1, 2, 4, 5), self.config['observation_noise']):
                parts[j] += self.noise_scale*self.np_random.uniform(-scale, scale, parts[j].shape)
        for j in range(3):
            parts.extend((self.error_history[j], self.velocity_history[j]))
        return np.concatenate(parts).astype(np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset after terminated or truncated')
        action = np.asarray(action, dtype=float)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Expected twelve finite action components')
        target = self.nominal + self.config['action_scale']*np.clip(
            action, -self.config['action_bound'], self.config['action_bound'])
        power = sum(self._motor_step(target) for _ in range(self.substeps))/self.substeps
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Non-finite paper environment state')
        rotation, velocity, omega = self._body_state()
        contact, bad_contact = self._contacts()
        foot_velocity = []
        for foot in self.feet:
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac, None, foot)
            foot_velocity.append(jac@self.data.qvel)
        foot_velocity = np.asarray(foot_velocity)
        foot_height = (self.data.xpos[self.feet, 2] - self.ground_height(self.data.xpos[self.feet, :2])
                       - self.config['foot_radius'])
        speed_xy = np.linalg.norm(foot_velocity[:, :2], axis=1)
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arcsin(np.clip(-rotation[2, 0], -1, 1))
        measured = np.array([velocity[0], velocity[1], omega[2]])
        tracking = self.config['tracking_weight']*np.exp(-np.sum((self.command-measured)**2))
        raw = dict(clearance=float(np.sum((foot_height-self.config['foot_clearance'])**2*np.sqrt(speed_xy))),
                   slip=float(np.sum(contact*speed_xy**2)), orientation=float(roll**2+pitch**2),
                   vertical_velocity=float(velocity[2]**2),
                   joint_pose=float(np.sum((self.data.qpos[self.qadr]-self.nominal)**2)), power=power,
                   smoothness1=float(np.sum((target-self.last_target)**2)),
                   smoothness2=float(np.sum((target-2*self.last_target+self.lastlast_target)**2)))
        weighted = {key: value*self.config['reward_weights'][key] for key, value in raw.items()}
        penalty = sum(weighted.values())
        self.t += 1
        self.failed = bool(bad_contact)
        terminal = self.config['terminal_penalty'] if self.failed else 0.
        reward = float(tracking - self.reward_scale*penalty-terminal)
        self.error_history[1:] = self.error_history[:-1].copy()
        self.velocity_history[1:] = self.velocity_history[:-1].copy()
        self.error_history[0] = target-self.data.qpos[self.qadr]
        self.velocity_history[0] = self.data.qvel[self.vadr]
        if self.randomize:
            self.error_history[0] += self.noise_scale*self.np_random.uniform(-.05, .05, 12)
            self.velocity_history[0] += self.noise_scale*self.np_random.uniform(-.5, .5, 12)
        self.lastlast_target, self.last_target = self.last_target.copy(), target.copy()
        truncated = self.t >= self.config['episode_steps'] and not self.failed
        self.done = self.failed or truncated
        info = dict(tracking=float(tracking), penalty=float(penalty), full_reward=float(tracking-penalty-terminal),
                    velocity=measured, velocity_error=float(np.linalg.norm(measured-self.command)),
                    vx_world=float(self.data.qvel[0]), power=power, slip=raw['slip'],
                    contact_count=int(contact.sum()), material_slip=self.contact_point_slip(), fallen=self.failed,
                    height=float(self.data.qpos[2]-self.ground_height(self.data.qpos[:2])),
                    reward_scale=self.reward_scale, noise_scale=self.noise_scale, rough=self.rough,
                    **{'cost_'+key: value for key, value in weighted.items()})
        return self.observation(), reward, self.failed, truncated, info

    def close(self):
        pass
