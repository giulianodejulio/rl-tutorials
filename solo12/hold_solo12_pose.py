"""Hold a bent-leg reference using the upstream position servos; no RL."""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

from solo12.inspect_solo12 import load_model, render_snapshot


def initialize_pose(model, data):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
             for a in range(model.nu)]
    reference = []
    for name in names:
        front = name.startswith(("FL_", "FR_"))
        if name.endswith("HAA"):
            reference.append(0.0)
        elif name.endswith("HFE"):
            reference.append(0.7 if front else -0.7)
        elif name.endswith("KFE"):
            reference.append(-1.4 if front else 1.4)
        else:
            raise ValueError(f"Unknown joint convention: {name}")
    reference = np.array(reference)
    joint_ids = model.actuator_trnid[:, 0]
    addresses = model.jnt_qposadr[joint_ids]
    np.testing.assert_allclose(model.actuator_gear[:, 0], 1.0)
    data.qpos[addresses] = reference
    data.ctrl[:] = reference
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_height = data.geom_xpos[floor_id, 2]
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, leg + "_FOOT")
            for leg in ("FL", "FR", "HL", "HR")]
    if min(feet) < 0:
        raise ValueError("Expected four named foot bodies")
    # Initialize directly in the reference pose, with feet just above the floor.
    # This is a reset, not a maneuver that stands the robot up from the ground.
    data.qpos[2] += floor_height + 0.012 - data.xpos[feet, 2].min()
    mujoco.mj_forward(model, data)
    return reference, addresses, names, floor_height


def simulate(model, data, reference, addresses, duration, viewer=None):
    records = []
    for _ in range(int(np.ceil(duration / model.opt.timestep))):
        if viewer is not None and not viewer.is_running():
            break
        start = time.monotonic()
        data.ctrl[:] = reference
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("Non-finite state")
        records.append((data.time, data.qpos.copy(), data.qvel.copy(),
                        data.qpos[addresses].copy(), data.actuator_force.copy(), data.ncon))
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.monotonic()-start)))
    return records


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--kp", type=float, default=None,
                        help="Override all position gains; leave XML unchanged")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")
    if args.kp is not None and (not np.isfinite(args.kp) or args.kp < 0):
        parser.error("kp must be finite and nonnegative")
    scene = repository.parents[1] / "third_party" / "solo12_mujoco" / "scene.xml"
    model, data = load_model(scene)
    if args.kp is not None:
        # Position servo: force = kp*ctrl - kp*length (+ velocity bias).
        # Updating only the control gain would change the reference scaling.
        np.testing.assert_allclose(model.actuator_biasprm[:, 1],
                                   -model.actuator_gainprm[:, 0])
        model.actuator_gainprm[:, 0] = args.kp
        model.actuator_biasprm[:, 1] = -args.kp
    reference, addresses, names, floor_height = initialize_pose(model, data)
    initial_qpos = data.qpos.copy()
    if args.viewer:
        from mujoco import viewer as viewer_module
        with viewer_module.launch_passive(model, data) as viewer:
            records = simulate(model, data, reference, addresses, args.duration, viewer)
    else:
        records = simulate(model, data, reference, addresses, args.duration)
    if not records:
        return
    directory = repository / "results" / "solo12" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    joints = np.array([r[3] for r in records])
    forces = np.array([r[4] for r in records])
    suffix = "" if args.kp is None else f"_kp_{args.kp:g}"
    path = directory / f"solo12_pose{suffix}.npz"
    np.savez(path, time=np.array([r[0] for r in records]),
             qpos=np.array([r[1] for r in records]), qvel=np.array([r[2] for r in records]),
             joint_positions=joints, actuator_forces=forces,
             contacts=np.array([r[5] for r in records]), reference=reference,
             joint_names=np.array(names), initial_qpos=initial_qpos, floor_height=floor_height,
             kp=model.actuator_gainprm[:, 0].copy(),
             timestep=model.opt.timestep, scene_path=str(scene), mujoco_version=mujoco.__version__)
    print("Position gains:", model.actuator_gainprm[:, 0])
    print("References (radians):", dict(zip(names, reference)))
    print(f"Final base height above floor: {data.qpos[2]-floor_height:.4f} m")
    print(f"Final max joint tracking error: {np.max(np.abs(joints[-1]-reference)):.4f} rad")
    print(f"Max absolute actuator torque: {np.max(np.abs(forces)):.4f} Nm")
    print(f"Final base quaternion (w,x,y,z): {data.qpos[3:7]}")
    print(f"Saved: {path}")
    if args.render:
        image = repository / "results" / "solo12" / "figures" / f"solo12_pose{suffix}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        render_snapshot(model, data, image)
        print(f"Saved: {image}")


if __name__ == "__main__":
    main()
