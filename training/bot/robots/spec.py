from dataclasses import dataclass, field

from isaaclab.assets import ArticulationCfg


@dataclass(frozen=True)
class RobotSpec:
    """Everything robot-specific that tasks need. A new robot = a new module in bot/robots/ defining SPEC."""

    cfg: ArticulationCfg
    leg_joints: tuple[str, ...]  # regexes for position-controlled leg joints
    wheel_joints: tuple[str, ...]  # regexes for velocity-controlled wheel joints
    base_body: str
    wheel_bodies: str  # regex matching the wheel links
    wheel_radius: float  # m
    max_step_height: float  # tallest stair step in the terrain curriculum [m]
    undesired_contact_bodies: str | None = None  # regex of links that should never touch the ground
    reward_weights: dict[str, float] = field(default_factory=dict)  # per-robot overrides of rewards_cfg.py
