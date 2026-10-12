"""Registers Bot-{Flat,Rough}-<robot>-v0 (and -Play-v0) for every robot in bot/robots/."""

import gymnasium as gym

from bot.robots import ROBOT_NAMES


def _env_entry(robot: str, terrain: str, play: bool):
    def env_cfg():
        from .env_cfg import make_env_cfg  # deferred: needs a running Isaac Sim app

        return make_env_cfg(robot, terrain, play)

    return env_cfg


def _ppo_entry(robot: str, terrain: str):
    def ppo_cfg():
        from bot.agents.rsl_rl_ppo_cfg import make_ppo_cfg

        return make_ppo_cfg(robot, terrain)

    return ppo_cfg


for _robot in ROBOT_NAMES:
    for _terrain in ("flat", "rough"):
        for _play in (False, True):
            gym.register(
                id=f"Bot-{_terrain.capitalize()}-{_robot}{'-Play' if _play else ''}-v0",
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": _env_entry(_robot, _terrain, _play),
                    "rsl_rl_cfg_entry_point": _ppo_entry(_robot, _terrain),
                },
            )
