"""Reward terms and default weights for all wheeled-quadruped tasks.

Tune weights here for every robot, or per robot via RobotSpec.reward_weights.
Joint/body selections are filled in from the RobotSpec by env_cfg.make_env_cfg().
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bot import mdp

_CMD = {"command_name": "base_velocity", "std": math.sqrt(0.25)}


@configclass
class WheeledRewardsCfg:
    # -- task: follow the commanded body velocity
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5, params=dict(_CMD))
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.75, params=dict(_CMD))

    # -- stability
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    # -- smoothness / effort (legs only; wheel speed is the task)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- gait preference: roll by default, step only when the terrain demands it
    wheels_off_ground = RewTerm(
        func=mdp.wheels_off_ground, weight=-0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces")}
    )
    leg_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
    )


# Terrain-specific weight changes applied on top of the defaults.
ROUGH_REWARD_WEIGHTS = {
    "flat_orientation_l2": -0.5,  # the body must pitch on stairs and slopes
    "leg_deviation": -0.05,  # legs need room to step
}
