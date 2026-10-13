"""Record scripted-command demo videos of a trained policy.

  --scene flat:   stand, spin in place, circle of --radius, straight then left turn
  --scene stairs: climb a staircase of --step_height, turn around, descend (needs a height-scan policy)

Wheels get a red radial stripe (demo-only URDF copy) and the overlay shows wheel speeds, so rolling vs
locked wheels are visible. Run inside the container (see scripts/env.sh), or via scripts/slurm/demo.sbatch:
    isaaclab_exec -p $REPO_ROOT/scripts/demo.py --task Bot-Flat-go2w-Play-v0 \
        --policy <run>/exported/policy.pt --out <dir> --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Bot-Flat-go2w-Play-v0")
parser.add_argument("--robot", default="go2w")
parser.add_argument("--policy", required=True, help="exported TorchScript policy (exported/policy.pt)")
parser.add_argument("--out", required=True, help="output directory for the mp4s")
parser.add_argument("--scene", choices=("flat", "stairs"), default="flat")
parser.add_argument("--radius", type=float, default=1.0, help="circle radius [m]")
parser.add_argument("--step_height", type=float, default=0.10, help="stair step height [m]")
parser.add_argument("--no_wheel_markers", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import bot.tasks  # noqa: E402, F401  (registers Bot-* tasks)
from bot.robots import get_robot  # noqa: E402

FOLLOW = ("follow", (2.0, 2.0, 1.2))  # camera offset from the robot
SIDE = ("follow", (0.0, 3.0, 1.0))
OVERHEAD = ("overhead", None)  # fixed camera looking down on the circle

STAND = (2.0, (0.0, 0.0, 0.0), "stand")
v_circle = 0.5
FLAT_SCENARIOS = {
    "1_stand": ([(6.0, (0.0, 0.0, 0.0), "stand still")], FOLLOW),
    "2_spin_in_place": ([STAND, (10.0, (0.0, 0.0, 1.0), "spin in place: wz = 1.0 rad/s")], FOLLOW),
    "3_circle": (
        [STAND,
         (2 * math.pi * args.radius / v_circle + 2.0, (v_circle, 0.0, v_circle / args.radius),
          f"circle r = {args.radius:.1f} m: vx = {v_circle} m/s, wz = {v_circle / args.radius:.2f} rad/s")],
        OVERHEAD,
    ),
    "4_straight_then_left": (
        [STAND, (5.0, (1.0, 0.0, 0.0), "straight: vx = 1.0 m/s"),
         (5.0, (0.6, 0.0, 0.8), "turn left: vx = 0.6 m/s, wz = 0.8 rad/s"), (3.0, (0.0, 0.0, 0.0), "stop")],
        FOLLOW,
    ),
}
STAIRS_SCENARIOS = {
    f"5_stairs_{round(args.step_height * 100)}cm": (
        [STAND, (10.0, (0.5, 0.0, 0.0), f"climb {args.step_height * 100:.0f} cm steps: vx = 0.5 m/s"),
         (1.0, (0.0, 0.0, 0.0), "stop"), (math.pi, (0.0, 0.0, 1.0), "turn around"),
         (10.0, (0.5, 0.0, 0.0), "descend: vx = 0.5 m/s"), (2.0, (0.0, 0.0, 0.0), "stop")],
        SIDE,
    ),
}


def marked_urdf(urdf_path: str, wheel_regex: str, out_dir: str) -> str:
    """Copy of the URDF with a red radial stripe on the outer face of each wheel."""
    tree = ET.parse(urdf_path)
    src_dir = os.path.dirname(os.path.abspath(urdf_path))
    for mesh in tree.iter("mesh"):  # the copy lives elsewhere: make mesh paths absolute
        fn = mesh.get("filename")
        if fn and "://" not in fn and not os.path.isabs(fn):
            mesh.set("filename", os.path.join(src_dir, fn))
    for link in tree.getroot().iter("link"):
        cyl = link.find("collision/geometry/cylinder")
        if not re.fullmatch(wheel_regex, link.get("name")) or cyl is None:
            continue
        r, length = float(cyl.get("radius")), float(cyl.get("length"))
        y = float(link.find("collision/origin").get("xyz").split()[1])  # wheel axis is the link's y axis
        face = y + math.copysign(length / 2 + 0.003, y if y != 0 else 1.0)
        vis = ET.SubElement(link, "visual")
        ET.SubElement(vis, "origin", xyz=f"{r * 0.45} {face} 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(vis, "geometry"), "box", size=f"{r * 0.9} 0.006 0.025")
        ET.SubElement(ET.SubElement(vis, "material", name="wheel_marker"), "color", rgba="1 0 0 1")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "marked_" + os.path.basename(urdf_path))
    tree.write(path)
    return path


def stairs_terrain(step_height: float) -> terrain_gen.TerrainGeneratorCfg:
    """One 12x12 m tile: robot starts in a pit, stairs lead up on all four sides."""
    return terrain_gen.TerrainGeneratorCfg(
        size=(12.0, 12.0), border_width=5.0, num_rows=1, num_cols=1, curriculum=False,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
        sub_terrains={
            "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=1.0, step_height_range=(step_height, step_height), step_width=0.3,
                platform_width=3.0, border_width=2.0,
            )
        },
    )


def label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    draw.rectangle((10, 10, 760, 20 + 36 * len(lines)), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((20, 15 + 36 * i), line, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main():
    spec = get_robot(args.robot)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.episode_length_s = 1e4  # no time-out resets mid-video
    cmd_cfg = cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1e9, 1e9)  # we set the command ourselves
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    has_terrain_gen = cfg.scene.terrain.terrain_generator is not None
    if args.scene == "stairs":
        assert cfg.scene.height_scanner is not None, "stairs demo needs a height-scan (rough) task"
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = stairs_terrain(args.step_height)
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.curriculum.terrain_levels = None
    elif has_terrain_gen:
        cfg.scene.terrain.terrain_type = "plane"  # flat demos; the height scan still works on a plane
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None
    if not args.no_wheel_markers:
        cfg.scene.robot.spawn.asset_path = marked_urdf(cfg.scene.robot.spawn.asset_path, spec.wheel_bodies, args.out)

    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
    base = env.unwrapped
    policy = torch.jit.load(args.policy, map_location=base.device).eval()
    command = base.command_manager.get_term("base_velocity")
    robot = base.scene["robot"]
    wheel_ids, wheel_names = robot.find_joints(list(spec.wheel_joints), preserve_order=True)
    cam = base.viewport_camera_controller
    dt = base.step_dt
    os.makedirs(args.out, exist_ok=True)

    for name, (segments, (cam_mode, eye)) in (STAIRS_SCENARIOS if args.scene == "stairs" else FLAT_SCENARIOS).items():
        obs, _ = env.reset()
        pos0 = robot.data.root_pos_w[0].cpu().numpy()
        if cam_mode == "follow":
            cam.update_view_to_asset_root("robot")
            cam.update_view_location(eye=eye, lookat=(0.0, 0.0, 0.0))
        else:  # circle centre is radius to the robot's left (it starts facing +x)
            cx, cy = pos0[0], pos0[1] + args.radius
            cam.update_view_to_world()
            cam.update_view_location(eye=(cx - 0.01, cy - 3.5 * args.radius - 1.5, 3.0 * args.radius + 2.5), lookat=(cx, cy, 0.0))
        frames, resets = [], 0
        for seconds, cmd, text in segments:
            for _ in range(int(seconds / dt)):
                command.vel_command_b[:] = torch.tensor(cmd, device=base.device)
                obs["policy"][:, 9:12] = command.vel_command_b  # obs already built with the old command
                with torch.inference_mode():
                    actions = policy(obs["policy"])
                obs, _, terminated, truncated, _ = env.step(actions)
                resets += int(terminated[0] or truncated[0])
                wheels = robot.data.joint_vel[0, wheel_ids].tolist()
                lines = [
                    text,
                    f"cmd    vx {cmd[0]:+.2f}  wz {cmd[2]:+.2f}",
                    f"actual vx {robot.data.root_lin_vel_b[0, 0].item():+.2f}  wz {robot.data.root_ang_vel_b[0, 2].item():+.2f}",
                    "wheels rad/s " + "  ".join(f"{n.split('_')[0]} {w:+5.1f}" for n, w in zip(wheel_names, wheels)),
                ]
                if resets:
                    lines.append(f"FELL / reset x{resets}")
                frames.append(label(env.render(), lines))
        path = os.path.join(args.out, f"{name}.mp4")
        imageio.mimsave(path, frames, fps=round(1 / dt), macro_block_size=8)
        print(f"DEMO saved {path} ({len(frames)} frames, {resets} resets)", flush=True)


if __name__ == "__main__":
    main()
    app.close()
