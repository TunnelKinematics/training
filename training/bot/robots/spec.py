from dataclasses import dataclass

from isaaclab.assets import ArticulationCfg


@dataclass(frozen=True)
class RobotSpec:
    """Everything robot-specific that tasks and rewards need. Swapping robots = writing a new spec."""

    cfg: ArticulationCfg
    leg_joints: tuple[str, ...]  # regexes for position-controlled leg joints
    wheel_joints: tuple[str, ...]  # regexes for velocity-controlled wheel joints
    base_body: str
    wheel_bodies: str  # regex matching the wheel links
    wheel_radius: float  # meters
