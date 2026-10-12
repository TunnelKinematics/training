"""Sanity-check a robot asset before training: it loads, stands, and rolls on its wheels.

Run inside the container (see scripts/env.sh):
    isaaclab_exec -p $REPO_ROOT/scripts/check_robot.py --robot go2w --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", default="go2w")
parser.add_argument("--wheel_speed", type=float, default=10.0, help="wheel speed for the roll test [rad/s]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from bot.robots import get_robot  # noqa: E402

DT = 0.005


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"CHECK {name}: {'PASS' if ok else 'FAIL'}  ({detail})", flush=True)
    return ok


def main() -> bool:
    spec = get_robot(args.robot)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(spec.cfg.replace(prim_path="/World/Robot"))
    sim.reset()

    legs, leg_names = robot.find_joints(list(spec.leg_joints), preserve_order=True)
    wheels, wheel_names = robot.find_joints(list(spec.wheel_joints), preserve_order=True)
    wheel_bodies, _ = robot.find_bodies(spec.wheel_bodies)
    mass = robot.root_physx_view.get_masses().sum().item()
    print("joints:", robot.joint_names)
    print("bodies:", robot.body_names)
    print(f"total mass: {mass:.2f} kg")

    ok = True
    ok &= check("joints", len(legs) == 12 and len(wheels) == 4, f"{len(legs)} legs, {len(wheels)} wheels")

    def run(seconds: float, wheel_vel: float):
        pos_target = robot.data.default_joint_pos.clone()
        vel_target = torch.zeros_like(robot.data.default_joint_vel)
        vel_target[:, wheels] = wheel_vel
        for _ in range(int(seconds / DT)):
            robot.set_joint_position_target(pos_target)
            robot.set_joint_velocity_target(vel_target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(DT)

    # 1) Stand: hold the default pose and let it settle on its wheels.
    run(2.0, 0.0)
    height = robot.data.root_pos_w[0, 2].item()
    up = -robot.data.projected_gravity_b[0, 2].item()
    wheel_z = robot.data.body_pos_w[0, wheel_bodies, 2]
    ok &= check("upright", up > 0.95, f"up-alignment {up:.3f}")
    ok &= check("base height", 0.2 < height < 0.6, f"{height:.3f} m")
    ok &= check(
        "wheels on ground",
        bool(torch.all((wheel_z - spec.wheel_radius).abs() < 0.02)),
        f"wheel centers {[round(z, 3) for z in wheel_z.tolist()]} vs radius {spec.wheel_radius}",
    )

    # 2) Roll: spin all wheels forward, measure forward speed after it settles.
    run(2.0, args.wheel_speed)
    x0 = robot.data.root_pos_w[0, 0].item()
    run(1.0, args.wheel_speed)
    speed = robot.data.root_pos_w[0, 0].item() - x0
    expected = args.wheel_speed * spec.wheel_radius
    ok &= check("roll forward", abs(speed - expected) < 0.3 * expected, f"{speed:.3f} m/s, expected ~{expected:.3f}")
    up = -robot.data.projected_gravity_b[0, 2].item()
    ok &= check("upright while rolling", up > 0.95, f"up-alignment {up:.3f}")
    return ok


if __name__ == "__main__":
    passed = main()
    print(f"CHECK robot overall: {'PASS' if passed else 'FAIL'}", flush=True)
    app.close()
