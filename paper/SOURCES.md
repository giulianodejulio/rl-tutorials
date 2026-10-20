# Specification and fidelity

This is a **MuJoCo port of the simulation locomotion method**, not an execution
of the authors' RaiSim program and not a claim to reproduce their numbers or
hardware transfer. No tutorial `Solo12Env`, tutorial reward, or tutorial training
checkpoint is used.

Primary specification:
- Aractingi et al., *Controlling the Solo12 quadruped robot with deep
  reinforcement learning*, Scientific Reports 13, 11945 (2023),
  https://doi.org/10.1038/s41598-023-38259-7
- Accessible equations/tables: https://arxiv.org/html/2309.16683v1
- Authors' reference environment: https://github.com/Gepetto/soloRL,
  pinned commit `66554dccb44f4130cc3ddee490b4884b0215a5b5` in `reference/soloRL/`.
  The snapshot retains original author attribution. No upstream license is
  supplied in that repository; its presence here does not assign a new license.
- Shared mesh/inertia model: rkourdis/solo12_mujoco, workspace-pinned commit
  `846cd588f461050068aaea03511915a8ddd8f06a`. Read in place, never modified.

## What is implemented

| Item | Implemented specification |
| --- | --- |
| Policy/value networks | Independent [256,128,32], LeakyReLU; Gaussian diagonal policy with trainable log standard deviations |
| Timing | 1e-3 s physics, 1e-2 s policy; PD recomputed at every physics substep |
| Joint control | Explicit torque PD, kp=3, kd=2e-1, desired velocity zero; torque capped at +/-3 Nm as in official code |
| Actions | Nominal pose + 3e-1 * policy output; no accumulated offsets |
| Pose | Symmetric official pose: HAA +/-4e-2; front HFE/KFE 8e-1/-1.6; hind -8e-1/1.6 |
| Observation | 132 values: projected vertical (3), base linear/angular velocities in body (3+3), command (3), joint q/qd (24), previous two absolute targets (24), three error/velocity history pairs (72). No world position, base height or time-to-go input |
| Histories | Current error = previous applied target minus measured q, followed by two older samples, similarly qd. Noise is sampled once when storing each history entry |
| Rewards | All eight penalty terms and combined 3D velocity tracking, equations and weights in article Table 3 |
| Energy | Article equation Pf + (tau + tau_f)^2/4.81 with tau_f=4.77e-2 sign(qd)+1.35e-4 qd; mean power over physics substeps in W |
| Smoothness | First and second differences of absolute angle targets |
| Clearance | Foot-frame height above local ground, subtract ankle-to-contact offset 1.48e-2 m; target 6e-2 m, weighted by sqrt(horizontal speed) |
| Slip | Article foot-frame horizontal speed squared, gated by contact; one contribution per leg. A separate material-contact-point metric is also recorded for diagnostics |
| Failure | Contact of a body other than the four lower legs with the ground; terminal penalty -10 from official code |
| Episode cuts | 100 steps / 1 s are truncations, not physical terminations. SB3 bootstraps using terminal_observation. At reset, 25% continue the physical state; otherwise nominal reset. New command per chunk |
| Randomization | Article Table 1, on observations and PD gains. Ground truth body velocity is available during policy learning, as in the paper |
| Curricula | Reward kc=min(1,3e-4*i); noise kn=min(1,2e-4*i), i=transitions/30000. All penalties use kc. Commands are sampled from +/-[1.5,1,1]*kn |
| Terrain | Last 1000 of 10000 canonical updates: 80% rough heightmaps, 1e-3 to 5e-2 m, 2x2 grid blocks at 1e-1 m grid spacing, otherwise flat |
| PPO | Article Table 2 flat: lr=5e-3, clip=2e-1, grad clip=5e-1, entropy=1e-2. Rough: 1e-3,5e-2,3e-1,0. Gamma=9.96e-1, GAE=9.5e-1,4 epochs,4 minibatches from official config |

## Deliberate resolutions and remaining differences

The article and the public C++ snapshot are **not identical**. This port gives
priority to published equations/tables instead of silently combining different
reward definitions:
- Published Table 3: tracking 6, clearance 20, slip 7e-2, orientation 3,
  vertical velocity 1.2, joint pose 5e-1, energy 2, first/second smoothness
  2.5/1.5. C++ has separate linear/yaw exponentials, different weights, and an
  extra negative-reward multiplier 5e-1. We use the published combined tracking
  and published weights, with no extra multiplier.
- Published energy is an inverse-K formula with friction-compensated torque.
  C++ accumulates a different 4.81*tau^2 expression and energy per control step.
  We implement published power, not C++ energy; reward is per policy step,
  **not** multiplied by control dt, unlike tutorial experiments.
- Published Table 2 learning rates/entropy differ from public YAML. The port
  uses Table 2, not YAML, and records exact configuration in every run.
- Reward curriculum starts at zero per article; the public code starts at 1e-1.
  Increment magnitudes come from YAML; the precise original runner is not
  published. We define one canonical iteration as 300*100=30000 samples.
- Published command sampling is symmetric and noise-scaled. Public C++ samples
  around forward command 5e-1 with a different range. We use published sampling.
- The published histories have three policy samples; C++ actually indexes a
  six-sample buffer at 0,2,4. We implement the three consecutive measurements.
- The source has unsafe/uninitialized quantities and unbounded reset loops;
  we do not reproduce undefined behavior. Settling is bounded at 200 physics
  steps under nominal PD, then joint/base velocities are reset to zero.
- MuJoCo replaces RaiSim, including contact solver, collision geometry, friction
  and the motor/inertia model. Position actuator gains/biases and inherited
  passive joint damping/friction are disabled in memory for torque control.
  The identified friction appears in the **power objective**, not as an extra
  hidden physical friction force. Assets remain shared and immutable.
- Terrain is a MuJoCo triangular heightfield. Bilinear local height is used for
  clearance (an approximation to the collision surface). Mesh contacts on a
  lower leg are treated as foot contacts, as in the official environment.
- SB3 requires finite action bounds: +/-100 is a numerical guard, not the old
  +/-1 action constraint. Observation normalization (saved VecNormalize), Adam,
  advantage normalization, initial Gaussian std=1 and orthogonal initialization
  follow explicit SB3 choices where the paper does not publish full details.
- CPU default uses 8 workers but preserves a 30000-transition batch by collecting
  3750 steps per worker. This changes parallel temporal sampling relative to
  300 workers *100 steps. `--num-envs 300` recovers that layout if resources allow.
- The separate supervised base-velocity estimator used for real hardware is
  not needed for simulator policy learning/evaluation and is **not included**.
  Real-robot deployment and exact RaiSim reproduction are outside this port.

These differences must be considered before interpreting failure/success as
confirmation or rejection of the original paper. In particular, a short CPU
run is a pilot, not equivalent to its reported 3e8-sample training budget.
