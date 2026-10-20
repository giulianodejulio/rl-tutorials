# Solo12: MuJoCo environment and tools

Model source: https://github.com/rkourdis/solo12_mujoco

Inspected upstream revision: `846cd588f461050068aaea03511915a8ddd8f06a`.
The external repository lives in `solo12_rl_ws/third_party/solo12_mujoco`,
outside this tutorial repository. No model assets are copied into the tutorials.

Setup from the workspace root:

```bash
git clone https://github.com/rkourdis/solo12_mujoco.git third_party/solo12_mujoco
cd third_party/solo12_mujoco
git checkout 846cd588f461050068aaea03511915a8ddd8f06a
python3 generate_model.py
```

Generation uses `lxml`. The original requirements pin an older MuJoCo;
we did not install that requirements file or downgrade the workspace.
Generation was run on the host with its available lxml.

From the tutorial repository in the container:

```bash
python3 -m solo12.inspect_solo12
python3 -m solo12.inspect_solo12 --viewer --duration 20
MUJOCO_GL=egl python3 -m solo12.inspect_solo12 --render
```

The viewer needs graphical display access. The headless default only loads the
model, prints its structure and simulates two seconds with zero POSITION
references. The position servos remain active: this is not a zero-torque fall
and is not a standing controller or a learned policy.

The upstream scene has a floor at z=-0.4, a floating base initially at z=0,
12 position actuators with kp=15 and force limits [-10,10]. These are upstream
example settings, not validated hardware parameters or a reproduction of the
paper's controller. Initial loading validates compatibility, not physical fidelity.

Outputs: `results/solo12/data/solo12_inspection.npz` and, with `--render`,
`results/solo12/figures/solo12_inspection.png`.

## Bent-leg pose (second step, no RL)

```bash
python3 -m solo12.hold_solo12_pose --viewer --duration 20
MUJOCO_GL=egl python3 -m solo12.hold_solo12_pose --render
```

The script initializes the robot directly in a bent-leg pose, places its foot
frames 12 mm above the upstream floor, then holds constant position references.
Front legs use HAA/HFE/KFE = [0,0.7,-1.4] rad; hind legs [0,-0.7,1.4] rad.
Actuators are matched by name, not by an assumed leg ordering.
The upstream gains, damping and force limits are unchanged. This tests an
unperturbed stance in this model, not standing up from a fallen pose, recovery,
robust balance, hardware suitability, or locomotion.

Outputs: `results/solo12/data/solo12_pose.npz` (joint tracking, base state,
actuator forces and contacts) and optional `results/solo12/figures/solo12_pose.png`.

Override the position gain in memory without changing XML:

```bash
python3 -m solo12.hold_solo12_pose --kp 3 --viewer --duration 20
```

Both the control gain and position bias coefficient are updated to preserve
`force = kp * (reference - joint_position)` for the unit transmissions.
Damping is unchanged. Overrides save separate outputs, e.g. `solo12_pose_kp_3.npz`.

## Gymnasium environment (third step, no training yet)

```bash
python3 -m solo12.validate_solo12_env
```

`Solo12Env` maps 12 normalized controls to absolute position references:
`q_desired = q_nominal + 0.25 * clip(action, -1, 1)` in radians. Offsets are
not accumulated. It loads a fresh upstream model, so gain overrides from the
separate stance script do not carry over.

One action is held for five 2 ms physics steps (control interval 10 ms).
The finite-horizon episode lasts 500 control steps (5 s), with earlier actual
termination if base height is below 0.12 m or the upright cosine is below 0.5.
Reset is deterministic, initializes the bent-leg pose, and has no randomization.

The 37-component observation is height above the floor, base quaternion (wxyz),
12 joint angles in model qpos order, 18 generalized velocities, forward speed
command, and remaining time fraction. Joint observations and actions have the
same order in this model (FL, FR, HL, HR). World x/y position is omitted on flat,
uniform ground. These are simulator state measurements, not a hardware sensor model.

The first reward tracks forward world-frame velocity (default command 0.2 m/s)
and penalizes lateral velocity, height deviation, tilt, vertical/angular motion,
offset size and actuator effort. The reward rate is multiplied by control_dt,
with an additional -1 on a fall. Exact weights are in `solo12_env.py`; they are
introductory choices, not paper parameters. Standing still has positive tracking
reward and is a baseline to beat, not evidence of locomotion learning.

Validation saves zero-offset and small-random-offset probes in
`results/solo12/data/solo12_env_validation.npz`. It also checks Gym compatibility,
action clipping/mapping and fall termination. No policy is trained in this step.

## First short PPO training

```bash
python3 -m solo12.train_ppo
python3 -m solo12.plot_ppo
```

The first run uses 20,000 control steps, a 1,000-transition buffer, minibatches
of 100 transitions and 10 optimization epochs. Actor and critic each have one
64-neuron ReLU hidden layer. Other principal settings follow the 2D PPO example:
gamma=1, lambda=0.95, clipping=0.2, learning rate=0.0003 and fixed sigma=0.3.
Reward and environment parameters are unchanged. `--steps` accepts a positive
multiple of 1,000. Existing training results are protected; use `--run NAME`
to save a separate experiment in `results/solo12/data/NAME/`.

Evaluation uses 10 stochastic episodes with paired Gaussian noise and one
mean-action episode (reset is deterministic) at each checkpoint. Zero offsets
are an additional baseline. Returns, mean forward speed, fall flags, episode
lengths, velocity/height trajectories, model and CSV logs are saved. These
evaluations do not update the model. The plot distinguishes return from duration.

Replay the saved model without training:

```bash
python3 -m solo12.replay_ppo --viewer
MUJOCO_GL=egl python3 -m solo12.replay_ppo --gif
```

The default applies the Gaussian means. Add `--stochastic` to sample actions.
Replay trajectories are saved under `results/solo12/data/ppo_first/`; optional
GIFs are reconstructed from those recorded states without a second rollout.

## Coordinated-action diagnostic (no RL)

```bash
python3 -m solo12.probe_coordinated_actions
python3 -m solo12.probe_coordinated_actions --viewer
python3 -m solo12.probe_coordinated_actions --plot-only
MUJOCO_GL=egl python3 -m solo12.probe_coordinated_actions --gif
```

A single hand-designed periodic pattern is compared with zero offsets, without
changing gains, reward or action limits. Diagonal pairs FL/HR and FR/HL have
opposite phases. Frequency is 1.5e0 Hz; hip sweep amplitude is 1.2e-1 rad and
extra knee flexion amplitude 2e-1 rad. A 5e-1 s ramp avoids a reference step.
Hip compensation of minus half the knee offset approximately separates leg
shortening from fore/aft swing. This is not inverse kinematics or feedback gait
control and does not guarantee no slipping or stable locomotion.

The saved `results/solo12/data/coordinated_actions.npz` includes references,
actual angles, torques, base states and floor-contact flags for each leg.
`--plot-only` redraws from this file without running MuJoCo. No parameter search
or training is performed; the test establishes feasibility only for this pattern
and unperturbed simulated initial state.

`--gif` renders the saved coordinated qpos trajectory with a fixed side camera,
without another simulation. Output: `results/solo12/figures/coordinated_actions.gif`.

## Longer PPO experiment

Keep the first experiment and repeat the same initialization and settings with
more training steps (not imitation of the coordinated-action diagnostic):

```bash
python3 -m solo12.train_ppo --steps 100000 --run ppo_long
python3 -m solo12.plot_ppo --run ppo_long --coordinated-reference
MUJOCO_GL=egl python3 -m solo12.replay_ppo --run ppo_long --gif
```

Only the training duration changes. Evaluation remains every 5e3 steps with
the same paired noise. The plot and replay accept `--run` and preserve the
first experiment's figures. This is still a single training seed.

At 1e5 steps (seed 42), stochastic evaluation over 10 episodes gives mean
return 4.385 and mean forward speed 1.486e-1 m/s, with no falls. Mean-action
evaluation gives return 4.267 and speed 2.461e-3 m/s. The manual coordinated
probe gives return 4.394 and speed 6.713e-2 m/s. Thus extra experience helps
the stochastic policy, but a reliable mean-action gait has not emerged.
The first 2e4 steps reproduce the original saved evaluation results.

## Saved-state locomotion diagnosis

```bash
python3 -m solo12.analyze_locomotion --run ppo_long
python3 -m solo12.analyze_locomotion --run ppo_long --plot-only
```

The first command needs MuJoCo and reconstructs geometry with mj_forward; it
does not step the simulation or train. The second only reads the saved NPZ
and needs NumPy/Matplotlib. Rows compare one stochastic replay, one mean-action
replay and the saved manual probe; columns show speed, sampled leg-floor
contacts and world-x foot positions relative to the base.

After the initial 1 s, stochastic replay has irregular contacts and horizontal
material-contact-point speeds averaging 3.328e-1 m/s (per leg/sample maximum
over its contacts, then unweighted mean). This indicates sliding at the sampled
contacts, not a measurement of the fraction of forward travel due to sliding.
Mean-action replay keeps all four legs in contact throughout that window.
Contact geometry is reconstructed at 1e-2 s control intervals, so shorter
contact events may be missed; recorded qvel is used for contact-point speeds.
The manual probe has no recorded qvel, so its slip speed is not estimated.
These are single-trajectory diagnoses, not robustness statistics.

The official reference implementation is https://github.com/Gepetto/soloRL
for "Controlling the Solo12 Robot with Deep Reinforcement Learning". Its
Environment.hpp includes foot-slip costs, first/second action-difference costs
and observation histories. These are relevant next comparisons; the current
MuJoCo environment remains an introductory setup, not a reproduction of RaiSim.

## First paper-inspired change: reference smoothness

The optional reward adds `-control_dt * smoothness_weight * sum((q_ref -
previous_q_ref)**2)` using applied (clipped) references. The previous reference
is nominal at reset. Constant references incur no further change penalty.
This is a discrete first-difference cost, not an angular-velocity cost: it is
not divided by control_dt. Weight 5e-1 is an introductory choice, not a copied
or tuned paper coefficient. Fixed Gaussian standard deviation remains 3e-1.

Because this reward depends on the last command, `--action-history` adds the
previous applied normalized action (12 values) to the observation, giving 49
values. Compare two new runs with this same observation and initialization to
isolate the penalty from the extra input. Original defaults remain 37 values
and zero smoothness weight, preserving old models and experiments.

```bash
python3 -m solo12.train_ppo --steps 100000 --run ppo_history --action-history
python3 -m solo12.train_ppo --steps 100000 --run ppo_smooth --action-history --smoothness-weight 0.5
python3 -m solo12.plot_smoothness
MUJOCO_GL=egl python3 -m solo12.replay_ppo --run ppo_smooth --gif --stochastic
```

The saved evaluation records both the modified reward and the original reward
(`base_returns`) as a common performance measure, plus RMS per-joint reference
changes, speed and falls. Plotted bands quantify evaluation noise, not variation
across independent training seeds. Replay reads observation configuration from
the saved training metadata, including for old runs without these new fields.

Completed comparison at 1e5 steps, seed 42 (ten paired stochastic episodes):

| Weight | Mode | Original reward return | Mean vx (m/s) | Per-joint reference-change RMS (rad) |
| --- | --- | --- | --- | --- |
| 0 | stochastic | 4.398 | 1.634e-1 | 1.223e-1 |
| 5e-1 | stochastic | 4.381 | 1.365e-1 | 1.191e-1 |
| 0 | mean actions | 4.284 | 5.457e-3 | 2.820e-3 |
| 5e-1 | mean actions | 4.535 | 6.042e-2 | 1.043e-2 |

No falls in these evaluations. Stochastic reference changes decrease by only
about 3 percent. Mean-action advancement appears late, but the saved replay is
asymmetric: FR/HL remain in contact almost all the time after 1 s, and material
contact points still have horizontal speed (mean 8.090e-2 m/s in that window).
Thus this one-seed result is not evidence of satisfactory or robust locomotion.
Fixed exploration noise remains a limitation; no slip penalty was added here.
The two runs' initial paired trajectories and the reward decomposition were
checked numerically, as were reset, clipping and the zero cost of a constant
reference. Results: ppo_smoothness_comparison.png and ppo_smooth_locomotion.png.

## Foot-slip penalty (optional, backwards compatible)

`--slip-weight` defaults to zero. At each control-step endpoint, the environment
uses mj_forward contact geometry and mj_jac @ qvel for the horizontal world
velocity of the material point on each contacting lower leg. Only floor
contacts with nonpositive separation are counted. For each of four legs use
the maximum squared horizontal speed over its contacts, then sum over legs;
call this S. The added reward is `-control_dt * slip_weight * S`.
Using a maximum per leg avoids multiplying the cost by mesh-contact count.
This is a deliberate MuJoCo adaptation of the official SoloRL foot-slip cost,
not a copy of its coefficient or its contact aggregation. Contact speed includes
both body translation and rotation. A foot moving in the air is not penalized.
The measure samples control endpoints, not all physics substeps.

The introductory slip weight is 5e-1; smoothness stays at 5e-1, observation
stays at 49 values, fixed Gaussian sigma stays at 3e-1, and PPO settings and
seed are unchanged. Slip adds no observation components. Train from scratch
for the same 1e5 steps; original results and checkpoints are preserved.

```bash
python3 -m solo12.train_ppo --run ppo_smooth_slip --steps 100000 --action-history --smoothness-weight 0.5 --slip-weight 0.5
python3 -m solo12.train_ppo --run ppo_smooth --evaluate-only
python3 -m solo12.plot_smoothness --slip
MUJOCO_GL=egl python3 -m solo12.replay_ppo --run ppo_smooth_slip --gif
```

`--evaluate-only` does not train or modify a checkpoint; it saves current metrics
for the final saved policy into final_evaluation.npz. This lets older checkpoints
receive the same slip measurement. In the slip comparison, old slip/contact
metrics are available only at the final checkpoint and are plotted as points.
The two top panels use original reward and forward speed. Bottom panels report
mean S and mean contact count. S is not conditioned on contact: reducing contact
can lower it, so it must be interpreted alongside motion and contacts, not alone.

Completed 1e5-step comparison (seed 42, ten paired stochastic evaluations):
mean slip index S decreases from 2.554e-1 to 2.022e-1 m²/s² (about 21 percent),
while mean forward speed decreases from 1.365e-1 to 1.182e-1 m/s. Mean contact
count changes from 2.073 to 1.996. With mean actions, speed falls from 6.042e-2
to 3.722e-3 m/s; slip drops mainly alongside movement. No falls in the saved
evaluations. Original-reward return for stochastic control slightly increases
from 4.381 to 4.412, illustrating again that return alone is insufficient.
These results do not demonstrate satisfactory locomotion or a reliable benefit
across seeds. Both models have the same initial trajectories, and the exact
reward decomposition was verified numerically. Analytic contact-velocity tests
(stationary, known translation, airborne) and SB3 environment checks passed.
The old saved model's evaluation metrics are reproduced by the updated code.
Figures and GIFs are saved separately with the ppo_smooth_slip prefix.

## Linear penalty curriculum

Optional `--curriculum-steps R` ramps both training penalty weights by
`c(n) = 1e-1 + 9e-1 * clip(n/R, 0, 1)`. With R=8e4 and 1e5 total steps,
the last 2e4 steps train with full target weights (5e-1 each). The weights
change only before collecting a new PPO buffer and remain fixed inside it.
Evaluation always uses full target weights; original-reward return is also
saved. Thus an easier early training reward cannot inflate the evaluation
comparison. This curriculum is a simple teaching experiment, not the official
paper's schedule. `--curriculum-steps 0` preserves constant-weight behavior.
The optional schedule is external to the episode, with no new observation
components. Replay uses the final target weights from metadata.

```bash
python3 -m solo12.train_ppo --run ppo_curriculum --steps 100000 --action-history --smoothness-weight 0.5 --slip-weight 0.5 --curriculum-steps 80000
python3 -m solo12.plot_smoothness --curriculum
MUJOCO_GL=egl python3 -m solo12.replay_ppo --run ppo_curriculum --gif
```

Comparison: ppo_smooth_slip (constant weights), same 49-dimensional inputs,
network initialization, PPO settings, fixed Gaussian sigma and paired evaluation
noise. The saved weight_steps/weight_scales describe the effective staircase
schedule, including the terminal target value. Figures are saved separately.

Completed curriculum result (seed 42, 1e5 steps, same paired evaluation):

| Quantity | Constant penalties | Curriculum |
| --- | --- | --- |
| Stochastic original-reward return | 4.412 | 4.377 |
| Stochastic mean vx (m/s) | 1.182e-1 | 1.033e-1 |
| Stochastic mean slip index (m²/s²) | 2.022e-1 | 2.166e-1 |
| Stochastic mean contact count | 1.996 | 2.324 |
| Mean-action mean vx (m/s) | 3.722e-3 | 5.404e-3 |

No falls in the evaluation episodes. The curriculum does not improve the joint
speed/slip tradeoff in this test; mean-action control still nearly stands still.
This does not establish that curricula generally fail: only one schedule and
one seed were compared. The slip index sums contributions over contacting
legs, so contact count also affects it; it is not a per-contact speed estimate.
The saved schedule, fixed evaluation objective, identical initial policy and
matching training configurations were verified. An interrupted initial preflight
used wrapper-level attributes without updating the raw environment; its logs
are isolated in ppo_curriculum_incomplete and excluded. The corrected path calls
the raw environment's set_penalty_weights through env_method, tested with the
actual Monitor + DummyVecEnv wrapper stack and resulting reward terms.
