"""Flat-ground velocity tracking for a wheeled quadruped, with rolling as the preferred gait.

Robot-agnostic: everything robot-specific comes from a RobotSpec. To train a new robot,
subclass WheeledFlatEnvCfg, override robot(), and register a gym id in __init__.py.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from bot import mdp
from bot.robots import RobotSpec
from bot.robots.go2w import GO2W


@configclass
class WheeledFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def robot(self) -> RobotSpec:
        """Override in subclasses. A method (not a field) so it stays out of the saved config."""
        raise NotImplementedError

    def __post_init__(self):
        super().__post_init__()
        spec = self.robot()

        def legs() -> SceneEntityCfg:
            # fresh object per term: Isaac Lab resolves SceneEntityCfg in place, so terms must not share one
            return SceneEntityCfg("robot", joint_names=list(spec.leg_joints))

        # -- scene: flat plane, no height scan
        self.scene.robot = spec.cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # -- actions: leg position offsets + wheel velocity targets
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=list(spec.leg_joints), scale=0.25, use_default_offset=True
        )
        self.actions.wheel_vel = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=list(spec.wheel_joints), scale=10.0, use_default_offset=True
        )

        # -- observations: wheel angles are unbounded, so only leg positions are observed
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": legs()}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )

        # -- commands: forward/backward + turning; no sideways rolling
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

        # -- events
        self.events.push_robot = None
        self.events.base_com = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = spec.base_body
        self.events.base_external_force_torque.params["asset_cfg"].body_names = spec.base_body
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }

        # -- rewards
        r = self.rewards
        r.track_lin_vel_xy_exp.weight = 1.5
        r.track_ang_vel_z_exp.weight = 0.75
        r.flat_orientation_l2.weight = -2.5
        r.dof_torques_l2.weight = -2.0e-4
        r.dof_torques_l2.params["asset_cfg"] = legs()
        r.dof_acc_l2.params["asset_cfg"] = legs()
        r.feet_air_time = None
        r.undesired_contacts = None
        # rolling preference: keep wheels on the ground and legs near the default stance
        r.wheels_off_ground = RewTerm(
            func=mdp.wheels_off_ground,
            weight=-0.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=spec.wheel_bodies)},
        )
        r.leg_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": legs()})

        # -- terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = spec.base_body


@configclass
class Go2WFlatEnvCfg(WheeledFlatEnvCfg):
    def robot(self) -> RobotSpec:
        return GO2W


@configclass
class Go2WFlatEnvCfg_PLAY(Go2WFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
