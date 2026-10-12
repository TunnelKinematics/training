"""Unitree Go2-W: 12 leg joints (position control) + 4 wheel joints (velocity control)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from bot import ASSETS_DIR

from .spec import RobotSpec

LEG_JOINTS = (".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")
WHEEL_JOINTS = (".*_foot_joint",)

GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSETS_DIR / "robots/go2w/go2w.urdf"),
        fix_base=False,
        merge_fixed_joints=True,
        activate_contact_sensors=True,
        # Gains here are overwritten by the actuator models below.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=list(LEG_JOINTS),
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        # Velocity control: stiffness 0, damping acts on wheel-speed error.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=list(WHEEL_JOINTS),
            effort_limit_sim=23.7,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.5,
        ),
    },
)

GO2W = RobotSpec(
    cfg=GO2W_CFG,
    leg_joints=LEG_JOINTS,
    wheel_joints=WHEEL_JOINTS,
    base_body="base",
    wheel_bodies=".*_foot",
    wheel_radius=0.086,
)
