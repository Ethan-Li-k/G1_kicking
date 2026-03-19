"""Pure kick skill AMP: one main task (g1-kick-skill-amp) with curriculum A/B/C. StageB/StageC tasks are optional finetune only."""

import gymnasium as gym


gym.register(
    id="g1-kick-skill",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "kick_task.kick_skill_env_cfg:KickSkillEnvCfg",
        "play_env_cfg_entry_point": "kick_task.kick_skill_env_cfg:KickSkillPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "kick_task.rsl_rl_ppo_cfg:KickSkillPPORunnerCfg",
    },
)


gym.register(
    id="g1-kick-skill-amp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "kick_task.kick_skill_amp_env_cfg:KickSkillAmpEnvCfg",
        "play_env_cfg_entry_point": "kick_task.kick_skill_amp_env_cfg:KickSkillAmpPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "kick_task.rsl_rl_amp_cfg:KickSkillAMPRunnerCfg",
    },
)


gym.register(
    id="g1-kick-skill-bootstrap-amp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "kick_task.kick_skill_amp_env_cfg:KickSkillBootstrapAmpEnvCfg",
        "play_env_cfg_entry_point": "kick_task.kick_skill_amp_env_cfg:KickSkillBootstrapAmpPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "kick_task.rsl_rl_amp_cfg:KickSkillBootstrapAMPRunnerCfg",
    },
)


# Optional finetune: fixed Stage B/C runner (no curriculum); use only when resuming from checkpoint.
gym.register(
    id="g1-kick-skill-stageB-amp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "kick_task.kick_skill_amp_env_cfg:KickSkillAmpEnvCfg",
        "play_env_cfg_entry_point": "kick_task.kick_skill_amp_env_cfg:KickSkillAmpPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "kick_task.rsl_rl_amp_cfg:KickSkillStageBAMPRunnerCfg",
    },
)
gym.register(
    id="g1-kick-skill-stageC-amp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "kick_task.kick_skill_amp_env_cfg:KickSkillAmpEnvCfg",
        "play_env_cfg_entry_point": "kick_task.kick_skill_amp_env_cfg:KickSkillAmpPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "kick_task.rsl_rl_amp_cfg:KickSkillStageCAMPRunnerCfg",
    },
)
