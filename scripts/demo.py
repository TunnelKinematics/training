"""Record scripted-command demo videos of a trained policy: stand, spin, circle, straight + left turn.

Run inside the container (see scripts/env.sh), or via scripts/slurm/demo.sbatch:
    isaaclab_exec -p $REPO_ROOT/scripts/demo.py --task Bot-Flat-go2w-Play-v0 \
        --policy <run>/exported/policy.pt --out <dir> --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Bot-Flat-go2w-Play-v0")
parser.add_argument("--policy", required=True, help="exported TorchScript policy (exported/policy.pt)")
parser.add_argument("--out", required=True, help="output directory for the mp4s")
parser.add_argument("--radius", type=float, default=1.0, help="circle radius [m]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import bot.tasks  # noqa: E402, F401  (registers Bot-* tasks)

FOLLOW = ("follow", (2.0, 2.0, 1.2))  # camera offset from the robot
OVERHEAD = ("overhead", None)  # fixed camera looking down on the circle

# name -> list of (seconds, (vx, vy, wz), label); camera
v_circle = 0.5
SCENARIOS = {
    "1_stand": ([(6.0, (0.0, 0.0, 0.0), "stand still")], FOLLOW),
    "2_spin_in_place": ([(2.0, (0.0, 0.0, 0.0), "stand"), (10.0, (0.0, 0.0, 1.0), "spin in place: wz = 1.0 rad/s")], FOLLOW),
    "3_circle": (
        [(2.0, (0.0, 0.0, 0.0), "stand"),
         (2 * math.pi * args.radius / v_circle + 2.0, (v_circle, 0.0, v_circle / args.radius),
          f"circle r = {args.radius:.1f} m: vx = {v_circle} m/s, wz = {v_circle / args.radius:.2f} rad/s")],
        OVERHEAD,
    ),
    "4_straight_then_left": (
        [(2.0, (0.0, 0.0, 0.0), "stand"), (5.0, (1.0, 0.0, 0.0), "straight: vx = 1.0 m/s"),
         (5.0, (0.6, 0.0, 0.8), "turn left: vx = 0.6 m/s, wz = 0.8 rad/s"), (3.0, (0.0, 0.0, 0.0), "stop")],
        FOLLOW,
    ),
}


def label(frame: np.ndarray, text: str, cmd, vel) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    lines = [text, f"cmd    vx {cmd[0]:+.2f}  wz {cmd[2]:+.2f}", f"actual vx {vel[0]:+.2f}  wz {vel[1]:+.2f}"]
    draw.rectangle((10, 10, 640, 125), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((20, 15 + 36 * i), line, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.episode_length_s = 1e4  # no time-out resets mid-video
    cmd_cfg = cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1e9, 1e9)  # we set the command ourselves
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    if getattr(cfg.scene, "terrain", None) is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_type = "plane"  # demos run on flat ground; the height scan still works
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None

    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
    base = env.unwrapped
    policy = torch.jit.load(args.policy, map_location=base.device).eval()
    command = base.command_manager.get_term("base_velocity")
    robot = base.scene["robot"]
    cam = base.viewport_camera_controller
    dt = base.step_dt
    os.makedirs(args.out, exist_ok=True)

    for name, (segments, (cam_mode, eye)) in SCENARIOS.items():
        obs, _ = env.reset()
        pos0 = robot.data.root_pos_w[0].cpu().numpy()
        if cam_mode == "follow":
            cam.update_view_to_asset_root("robot")
            cam.update_view_location(eye=eye, lookat=(0.0, 0.0, 0.0))
        else:  # circle centre is radius to the robot's left (it starts facing +x)
            cx, cy = pos0[0], pos0[1] + args.radius
            cam.update_view_to_world()
            cam.update_view_location(eye=(cx - 0.01, cy - 3.5 * args.radius - 1.5, 3.0 * args.radius + 2.5), lookat=(cx, cy, 0.0))
        frames = []
        for seconds, cmd, text in segments:
            for _ in range(int(seconds / dt)):
                command.vel_command_b[:] = torch.tensor(cmd, device=base.device)
                obs["policy"][:, 9:12] = command.vel_command_b  # obs already built with the old command
                with torch.inference_mode():
                    actions = policy(obs["policy"])
                obs, _, terminated, truncated, _ = env.step(actions)
                vel = (robot.data.root_lin_vel_b[0, 0].item(), robot.data.root_ang_vel_b[0, 2].item())
                frames.append(label(env.render(), text, cmd, vel))
        path = os.path.join(args.out, f"{name}.mp4")
        imageio.mimsave(path, frames, fps=round(1 / dt), macro_block_size=8)
        print(f"DEMO saved {path} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
    app.close()
