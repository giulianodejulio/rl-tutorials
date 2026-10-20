"""Load and inspect Solo12 in MuJoCo; no policy, reward or learning."""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np


def load_model(scene):
    if not scene.exists():
        raise FileNotFoundError(f"Missing {scene}. See solo12/README.md")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def describe_model(model):
    print(f"MuJoCo: {mujoco.__version__}")
    print(f"nq={model.nq}: position coordinates; nv={model.nv}: velocity coordinates")
    print(f"nu={model.nu}: actuator controls; mass={model.body_mass.sum():.6f} kg")
    print(f"Physics timestep: {model.opt.timestep:g} s")
    print("\nActuator | joint | position gain | control range | force range")
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        print(f"{name} | {joint_name} | {model.actuator_gainprm[a, 0]:g} | "
              f"{model.actuator_ctrlrange[a]} | {model.actuator_forcerange[a]}")
    print("\nZero controls are position references, NOT zero motor torques.")


def simulate(model, data, duration, viewer=None):
    count = int(np.ceil(duration / model.opt.timestep))
    records = []
    for _ in range(count):
        if viewer is not None and not viewer.is_running():
            break
        start = time.monotonic()
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("Non-finite simulation state")
        records.append((data.time, data.qpos.copy(), data.qvel.copy(), data.ncon))
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.monotonic() - start)))
    return records


def render_snapshot(model, data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.qpos[:3] + np.array([0, 0, -0.1])
    camera.distance, camera.azimuth, camera.elevation = 1.1, 135, -20
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        plt.imsave(path, renderer.render())


def main():
    repository = Path(__file__).resolve().parents[1]
    default_scene = repository.parents[1] / "third_party" / "solo12_mujoco" / "scene.xml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=default_scene)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--render", action="store_true", help="Save a snapshot; use MUJOCO_GL=egl headlessly")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")
    model, data = load_model(args.scene)
    describe_model(model)
    initial_qpos = data.qpos.copy()
    if args.viewer:
        from mujoco import viewer as viewer_module
        with viewer_module.launch_passive(model, data) as viewer:
            records = simulate(model, data, args.duration, viewer)
    else:
        records = simulate(model, data, args.duration)
    if not records:
        print("Viewer closed before the first simulation step.")
        return
    directory = repository / "results" / "solo12" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "solo12_inspection.npz"
    np.savez(path, time=np.array([r[0] for r in records]),
             qpos=np.array([r[1] for r in records]), qvel=np.array([r[2] for r in records]),
             contacts=np.array([r[3] for r in records]), initial_qpos=initial_qpos,
             controls=np.zeros(model.nu), scene_path=str(args.scene.resolve()),
             timestep=model.opt.timestep, mujoco_version=mujoco.__version__)
    print(f"\nSimulated {data.time:.3f} s; final base position: {data.qpos[:3]}")
    print(f"Final contacts: {data.ncon}; saved: {path}")
    if args.render:
        image_path = repository / "results" / "solo12" / "figures" / "solo12_inspection.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        render_snapshot(model, data, image_path)
        print(f"Saved: {image_path}")


if __name__ == "__main__":
    main()
