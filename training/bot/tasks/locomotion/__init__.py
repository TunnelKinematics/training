import gymnasium as gym

_CFG = "bot.tasks.locomotion.flat_env_cfg"
_PPO = "bot.agents.rsl_rl_ppo_cfg:FlatPPORunnerCfg"

gym.register(
    id="Bot-Flat-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_CFG}:Go2WFlatEnvCfg", "rsl_rl_cfg_entry_point": _PPO},
)
gym.register(
    id="Bot-Flat-Go2W-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_CFG}:Go2WFlatEnvCfg_PLAY", "rsl_rl_cfg_entry_point": _PPO},
)
