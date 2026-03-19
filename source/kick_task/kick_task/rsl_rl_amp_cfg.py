from isaaclab.utils import configclass

from beyondAMP.isaaclab.rsl_rl.configs.amp_cfg import AMPPPOAlgorithmCfg
from beyondAMP.isaaclab.rsl_rl.configs.amp_cfg import AMPRunnerCfg
from beyondAMP.isaaclab.rsl_rl.configs.amp_cfg import MotionDatasetCfg
from beyondAMP.isaaclab.rsl_rl.configs.rl_cfg import RslRlPpoActorCriticCfg
from beyondAMP.obs_groups import AMPObsBaiscTerms

from . import kick_amp_data_cfg as data_cfg


@configclass
class _BaseKickSkillAMPRunnerCfg(AMPRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 120000
    save_interval = 100
    experiment_name = "g1_kick_skill"
    run_name = "amp"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = AMPPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    amp_discr_hidden_dims = [256, 256]

    amp_data = MotionDatasetCfg(
        motion_files=data_cfg.kick_motion_files,
        motion_data_weights=data_cfg.kick_motion_data_weights,
        body_names=data_cfg.g1_key_body_names,
        anchor_name=data_cfg.g1_anchor_name,
        amp_obs_terms=AMPObsBaiscTerms,
    )


@configclass
class KickSkillAMPRunnerCfg(_BaseKickSkillAMPRunnerCfg):
    run_name = "stageA"
    amp_reward_coef = 0.5
    amp_task_reward_lerp = 0.85


@configclass
class KickSkillStageBAMPRunnerCfg(_BaseKickSkillAMPRunnerCfg):
    run_name = "stageB"
    amp_reward_coef = 0.45
    amp_task_reward_lerp = 0.85


@configclass
class KickSkillStageCAMPRunnerCfg(_BaseKickSkillAMPRunnerCfg):
    run_name = "stageC"
    amp_reward_coef = 0.35
    amp_task_reward_lerp = 0.75


@configclass
class KickSkillBootstrapAMPRunnerCfg(_BaseKickSkillAMPRunnerCfg):
    # Bootstrap: weaker AMP prior, more room for task reward to drive from-scratch learning.
    run_name = "bootstrap"
    amp_reward_coef = 0.7
    amp_task_reward_lerp = 0.55
