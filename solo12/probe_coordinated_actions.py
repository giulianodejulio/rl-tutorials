"""One hand-designed periodic-action diagnostic; no learning or gait search.

Use --plot-only to redraw from saved data without another simulation.
"""

import argparse
from pathlib import Path
import time

import numpy as np


def coordinated_action(t, names, scale, frequency=1.5e0):
    action = np.zeros(len(names))
    ramp = np.clip(t / 5e-1, 0, 1)
    for j, name in enumerate(names):
        phase = 2 * np.pi * frequency * t
        if name[:2] in ("FR", "HL"):
            phase += np.pi
        if name.endswith("HAA"):
            continue
        # Increase knee flexion during the intended swing half-cycle.
        knee_sign = -1 if name.startswith(("FL", "FR")) else 1
        knee_offset = knee_sign * 2e-1 * max(0, np.cos(phase))
        hip_offset = -1.2e-1 * np.sin(phase) - knee_offset / 2
        offset = hip_offset if name.endswith("HFE") else knee_offset
        action[j] = ramp * offset / scale
    if np.max(np.abs(action)) > 1 + 1e-12:
        raise ValueError("Diagnostic exceeded the policy's normalized action limits")
    return action


def run_probe(env, coordinated, viewer_enabled):
    import mujoco
    from contextlib import ExitStack

    env.reset()
    start_x = env.data.qpos[0]
    qpos = [env.data.qpos.copy()]
    records = []
    with ExitStack() as stack:
        viewer = None
        if viewer_enabled:
            from solo12.replay_ppo import passive_viewer
            viewer = stack.enter_context(passive_viewer(env.model, env.data))
        for t in range(env.num_steps):
            if viewer is not None and not viewer.is_running():
                break
            start = time.monotonic()
            action = (coordinated_action(t*env.control_dt, env.joint_names, env.action_scale)
                      if coordinated else np.zeros(env.model.nu))
            _, reward, terminated, truncated, info = env.step(action)
            contacts = np.zeros(4, dtype=bool)
            floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            for c in env.data.contact:
                if c.geom1 != floor and c.geom2 != floor:
                    continue
                geom = c.geom2 if c.geom1 == floor else c.geom1
                body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY,
                                             env.model.geom_bodyid[geom]) or ""
                for leg, prefix in enumerate(("FL", "FR", "HL", "HR")):
                    contacts[leg] |= body_name.startswith(prefix+"_")
            qpos.append(env.data.qpos.copy())
            records.append(dict(time=env.data.time, action=action, reward=reward,
                                vx=info["vx"], height=info["height"], upright=info["upright"],
                                fallen=info["fallen"], contacts=contacts,
                                joint_positions=env.data.qpos[env.joint_addresses].copy(),
                                desired_positions=info["desired_positions"],
                                torque=env.data.actuator_force.copy()))
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0, env.control_dt-(time.monotonic()-start)))
            if terminated or truncated:
                break
    if not records:
        raise RuntimeError("No samples collected")
    result = {key: np.array([r[key] for r in records]) for key in records[0]}
    result["qpos"] = np.array(qpos)
    result["displacement_x"] = env.data.qpos[0] - start_x
    return result


def plot_results(source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from output_paths import figure_path

    with np.load(source, allow_pickle=False) as d:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        fig.suptitle("Diagnostica senza RL: offset nulli e un solo pattern coordinato")
        for prefix, name, color in (("zero", "Offset nulli", "tab:blue"),
                                     ("coordinated", "Pattern coordinato", "tab:orange")):
            t, rewards = d[prefix+"_time"], d[prefix+"_reward"]
            axes[0, 0].plot(t, d[prefix+"_vx"], color=color, label=name)
            axes[0, 1].plot(t, d[prefix+"_height"], color=color, label=name)
            axes[1, 0].plot(t, np.cumsum(rewards), color=color, label=name)
            print(f"{name}: duration={t[-1]:.3e} s, return={rewards.sum():.3e}, "
                  f"mean vx={d[prefix+'_vx'].mean():.3e} m/s, "
                  f"displacement={float(d[prefix+'_displacement_x']):.3e} m, "
                  f"fallen={d[prefix+'_fallen'][-1]}")
        axes[0, 0].axhline(float(d["target_velocity"]), color="black", linestyle="--", label="Comando")
        axes[0, 1].axhline(1.2e-1, color="gray", linestyle="--", label="Soglia caduta per altezza")
        for ax, title, ylabel in zip(axes.flat[:3],
                                    [r"Velocità world ${}^{\mathcal{I}}v_x$", "Altezza della base", "Reward accumulate dall'inizio"],
                                    ["Velocità (m/s)", "Altezza sul pavimento (m)", "Somma delle reward ricevute"]):
            ax.set(title=title, xlabel="Tempo fisico (s)", ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        ax = axes[1, 1]
        t = d["coordinated_time"]
        names = list(d["joint_names"])
        for joint in ("FL_HFE", "FL_KFE", "FR_HFE", "FR_KFE"):
            j = names.index(joint)
            ax.plot(t, d["coordinated_desired_positions"][:, j]-d["reference"][j], label=joint)
        ax.set(title="Offset comandati alle zampe anteriori", xlabel="Tempo fisico (s)", ylabel="Offset angolare (rad)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
        path = figure_path("solo12", "coordinated_actions.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved: {path}")


def save_animation(source):
    import mujoco
    from PIL import Image, ImageDraw
    from solo12.inspect_solo12 import load_model

    repository = Path(__file__).resolve().parents[1]
    scene = repository.parents[1] / "third_party" / "solo12_mujoco" / "scene.xml"
    model, data = load_model(scene)
    with np.load(source, allow_pickle=False) as saved:
        positions = saved["coordinated_qpos"]
        dt = float(saved["control_dt"])
    camera = mujoco.MjvCamera()
    # Fixed side view makes net forward movement visible against the floor.
    camera.lookat[:] = (positions[0, :3]+positions[-1, :3])/2
    camera.lookat[2] -= 1e-1
    camera.distance, camera.azimuth, camera.elevation = 1.2e0, 90, -15
    frames, stride = [], 4
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for index in range(0, len(positions)-1, stride):
            data.qpos[:] = positions[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 38), fill="white")
            draw.text((8, 5), "Coordinated joint references | No RL | Fixed camera", fill="black")
            draw.text((8, 21), f"t={index*dt:.2e} s | x-x_initial={positions[index,0]-positions[0,0]:.2e} m",
                      fill="black")
            frames.append(frame)
    output = repository / "results" / "solo12" / "figures" / "coordinated_actions.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=round(stride*dt*1e3), loop=0)
    print(f"Saved: {output} (recorded states, no new simulation)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="Show the coordinated probe")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--gif", action="store_true", help="Render saved coordinated states; do not simulate again")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "results" / "solo12" / "data" / "coordinated_actions.npz"
    if args.gif:
        save_animation(source)
        if args.plot_only:
            plot_results(source)
        return
    if not args.plot_only:
        from solo12.solo12_env import Solo12Env
        env = Solo12Env()
        arrays = {}
        for prefix, coordinated in (("zero", False), ("coordinated", True)):
            result = run_probe(env, coordinated, args.viewer and coordinated)
            arrays.update({prefix+"_"+key: value for key, value in result.items()})
        source.parent.mkdir(parents=True, exist_ok=True)
        np.savez(source, **arrays, reference=env.reference, joint_names=np.array(env.joint_names),
                 target_velocity=env.target_velocity, action_scale=env.action_scale,
                 control_dt=env.control_dt, frequency=1.5e0, hip_sweep=1.2e-1,
                 knee_flexion=2e-1, ramp_duration=5e-1)
        env.close()
        print(f"Saved: {source}")
    plot_results(source)


if __name__ == "__main__":
    main()
