"""Kinematic FR reference animation: fixed base and other joints, no dynamics."""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from solo12.hold_solo12_pose import initialize_pose
from solo12.inspect_solo12 import load_model
from solo12.probe_coordinated_actions import coordinated_action


def main():
    repository = Path(__file__).resolve().parents[1]
    scene = repository.parents[1] / "third_party" / "solo12_mujoco" / "scene.xml"
    model, data = load_model(scene)
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 540
    nominal, addresses, names, _ = initialize_pose(model, data)
    initial = data.qpos.copy()
    initial[:3] = [0, 0, 1.5e-1]
    initial[3:7] = [1, 0, 0, 0]
    selected = [names.index("FR_HFE"), names.index("FR_KFE")]
    movable = addresses[selected]
    fixed = np.array([j for j in range(model.nq) if j not in movable])
    dt, duration, scale = 4e-2, 5e0, 2.5e-1
    times = np.arange(0, duration, dt)
    frames, positions, foot_positions = [], [], []
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_FOOT")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [1e-1, -9e-2, 2e-2]
    camera.distance, camera.azimuth, camera.elevation = 8.5e-1, -100, -10
    directory = repository / "results" / "solo12" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=540, width=800) as renderer:
        for t in times:
            action = coordinated_action(t, names, scale)
            data.qpos[:] = initial
            data.qpos[movable] = nominal[selected] + scale*action[selected]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(data.qpos[fixed], initial[fixed])
            np.testing.assert_allclose(data.qpos[addresses],
                                       nominal + np.where(np.isin(np.arange(model.nu), selected), scale*action, 0))
            positions.append(data.qpos.copy())
            foot_positions.append(data.xpos[foot].copy())
            renderer.update_scene(data, camera=camera)
            # Highlight the FR chain while retaining the complete robot as context.
            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid < 0:
                    continue
                body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                         model.geom_bodyid[geom.objid]) or ""
                if body.startswith("FR_"):
                    geom.rgba[:] = [1e0, 4e-1, 5e-2, 1e0]
                elif body.startswith(("FL_", "HL_", "HR_")):
                    geom.rgba[:] = [6e-1, 6e-1, 6e-1, 2e-1]
                elif body == "base_link":
                    geom.rgba[3] = 5e-1
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 800, 56), fill="white")
            draw.text((10, 6), "FR (orange): prescribed HFE/KFE references | Fixed base and other joints", fill="black")
            draw.text((10, 23), "KINEMATIC ANIMATION: no physics, no RL | Other legs translucent", fill="black")
            draw.text((10, 40), f"t={t:.2e} s | HFE={data.qpos[movable[0]]:.2e} rad | KFE={data.qpos[movable[1]]:.2e} rad",
                      fill="black")
            frames.append(frame)
    output = directory / "fr_reference_motion.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=round(dt*1e3), loop=0)
    frames[len(frames)//3].save(directory / "fr_reference_motion_preview.png")
    data_directory = repository / "results" / "solo12" / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    np.savez(data_directory / "fr_reference_motion.npz", time=times, qpos=np.array(positions),
             foot_positions=np.array(foot_positions), nominal=nominal,
             joint_names=np.array(names), moving_joints=np.array(["FR_HFE", "FR_KFE"]),
             frame_dt=dt, action_scale=scale)
    print(f"Saved: {output}")
    print("Validated: only FR_HFE/FR_KFE change; base and all other joints fixed.")


if __name__ == "__main__":
    main()
