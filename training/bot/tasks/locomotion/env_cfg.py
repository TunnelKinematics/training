"""Velocity-tracking locomotion for any wheeled quadruped described by a RobotSpec.

make_env_cfg(robot, terrain, play) builds the config; nothing in here is robot-specific.
  terrain="flat":  plane, blind policy (proprioception only)
  terrain="rough": slopes / rough / boxes / stairs with a curriculum, policy sees a height scan
"""

from isaaclab.envs import ViewerCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from bot import mdp
from bot.robots import get_robot

from .rewards_cfg import ROUGH_REWARD_WEIGHTS, WheeledRewardsCfg
from .terrains import rough_terrains


@configclass
class WheeledLocomotionEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: WheeledRewardsCfg = WheeledRewardsCfg()


def make_env_cfg(robot: str, terrain: str, play: bool = False) -> WheeledLocomotionEnvCfg:
    assert terrain in ("flat", "rough"), terrain
    spec = get_robot(robot)
    cfg = WheeledLocomotionEnvCfg()

    def legs() -> SceneEntityCfg:
        # fresh object per term: Isaac Lab resolves SceneEntityCfg in place, so terms must not share one
        return SceneEntityCfg("robot", joint_names=list(spec.leg_joints))

    # -- scene
    cfg.scene.robot = spec.cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + spec.base_body
    if terrain == "flat":
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.height_scanner = None
        cfg.observations.policy.height_scan = None
        cfg.curriculum.terrain_levels = None
    else:
        cfg.scene.terrain.terrain_generator = rough_terrains(spec.max_step_height)
        cfg.scene.terrain.terrain_generator.curriculum = True

    # -- actions: leg position offsets + wheel velocity targets
    cfg.actions.joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(spec.leg_joints), scale=0.25, use_default_offset=True
    )
    cfg.actions.wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=list(spec.wheel_joints), scale=10.0, use_default_offset=True
    )

    # -- observations: wheel angles are unbounded, so only leg positions are observed
    cfg.observations.policy.joint_pos = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": legs()}, noise=Unoise(n_min=-0.01, n_max=0.01)
    )

    # -- commands: forward/backward + turning; no sideways rolling
    cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

    # -- events (domain randomization)
    ev = cfg.events
    ev.push_robot = None
    ev.base_com = None
    ev.physics_material.params["static_friction_range"] = (0.4, 1.2)
    ev.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
    ev.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    ev.add_base_mass.params["asset_cfg"].body_names = spec.base_body
    ev.base_external_force_torque.params["asset_cfg"].body_names = spec.base_body
    ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}

    # -- rewards: fill in robot joints/bodies, then apply terrain and robot weight overrides
    r = cfg.rewards
    for name in ("dof_torques_l2", "dof_acc_l2", "dof_pos_limits", "leg_deviation"):
        getattr(r, name).params["asset_cfg"] = legs()
    r.wheels_off_ground.params["sensor_cfg"].body_names = spec.wheel_bodies
    if terrain == "rough":
        r.wheels_off_ground.params["scanner_cfg"] = SceneEntityCfg("height_scanner")
    if spec.undesired_contact_bodies:
        r.undesired_contacts.params["sensor_cfg"].body_names = spec.undesired_contact_bodies
    else:
        r.undesired_contacts = None
    weights = {**(ROUGH_REWARD_WEIGHTS if terrain == "rough" else {}), **spec.reward_weights}
    for name, weight in weights.items():
        getattr(r, name).weight = weight

    # -- terminations
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = spec.base_body

    if play:
        cfg.scene.num_envs = 16
        cfg.observations.policy.enable_corruption = False
        cfg.events.base_external_force_torque = None
        cfg.viewer = ViewerCfg(eye=(2.5, 2.5, 1.5), origin_type="asset_root", asset_name="robot", env_index=0)
        if terrain == "rough":
            gen = cfg.scene.terrain.terrain_generator
            gen.num_rows, gen.num_cols, gen.curriculum = 5, 5, False
            cfg.scene.terrain.max_init_terrain_level = None
    return cfg
