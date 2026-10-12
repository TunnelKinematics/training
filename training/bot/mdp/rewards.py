from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["terrain_roughness", "wheels_off_ground"]


def terrain_roughness(env: ManagerBasedRLEnv, scanner_cfg: SceneEntityCfg) -> torch.Tensor:
    """Std-dev of terrain height around the robot [m]: ~0 on flat ground, ~step height on stairs."""
    scanner: RayCaster = env.scene.sensors[scanner_cfg.name]
    heights = scanner.data.ray_hits_w[..., 2]
    heights = torch.where(torch.isfinite(heights), heights, scanner.data.pos_w[:, 2:3] - 1.0)
    heights = heights - heights.median(dim=1, keepdim=True).values
    return heights.clamp(-1.0, 1.0).std(dim=1)


def wheels_off_ground(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    scanner_cfg: SceneEntityCfg | None = None,
    roughness_scale: float = 0.03,
) -> torch.Tensor:
    """Number of wheels not touching the ground. Penalizing it makes rolling the default gait.

    With a height scanner, the penalty fades out on rough terrain (stairs, boxes) so the policy may step there.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0]
    off = (forces < threshold).float().sum(dim=1)
    if scanner_cfg is not None:
        off = off * torch.exp(-terrain_roughness(env, scanner_cfg) / roughness_scale)
    return off
