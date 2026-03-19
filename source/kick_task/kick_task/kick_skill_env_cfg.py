from __future__ import annotations

import math
import os

from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .ball_env_cfg import SoccerBallSceneCfg
from . import mdp


def _resolve_kick_motion_default():
    default_rel = "data/datasets/g1_kick_skill/wo_cf_shoot_74_06.npz"
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for prefix in ("", _repo_root):
        path = os.path.join(prefix, default_rel) if prefix else default_rel
        if os.path.isfile(path):
            return path
    return default_rel


_KICK_MOTION_FILE = os.getenv("KICK_SKILL_MOTION_FILE", _resolve_kick_motion_default())


@configclass
class CommandsCfg:
    ball_target_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="ball",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.4, 0.8),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.35, 1.2),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(0.0, 0.0),
        ),
        goal_heading_range_deg=(-8.0, 8.0),
        goal_speed_range=(0.45, 0.95),
        goal_direction_xy=(1.0, 0.0),
    )

    kick_motion = mdp.KickMotionCommandCfg(
        asset_name="robot",
        motion_file=_KICK_MOTION_FILE,
        anchor_body_name="torso_link",
        body_names=[
            "pelvis",
            "torso_link",
            "right_hip_roll_link",
            "right_hip_pitch_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_hip_roll_link",
            "left_hip_pitch_link",
            "left_knee_link",
            "left_ankle_roll_link",
        ],
        critical_body_names=[
            "pelvis",
            "torso_link",
            "right_hip_roll_link",
            "right_hip_pitch_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.005, 0.005),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.08, 0.08),
        },
        velocity_range={
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.05, 0.05),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.12, 0.12),
        },
        joint_position_range=(-0.08, 0.08),
        strike_phase_window=(0.35, 0.62),
        pre_kick_phase_end=0.34,
        recover_phase_start=0.72,
        kicking_foot_body_name="right_ankle_roll_link",
        supporting_foot_body_name="left_ankle_roll_link",
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=0.8)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.15, n_max=0.15))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.8, n_max=0.8))
        last_action = ObsTerm(func=mdp.last_action)

        ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel)
        ball_vel_rel = ObsTerm(func=mdp.ball_vel_rel)
        goal_dir_rel = ObsTerm(func=mdp.goal_dir_rel, params={"command_name": "ball_target_velocity"})
        ball_to_goal_dir_rel = ObsTerm(func=mdp.ball_to_goal_dir_rel, params={"command_name": "ball_target_velocity"})

        kick_phase = ObsTerm(func=mdp.kick_phase, params={"command_name": "kick_motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "kick_motion"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel)
        ball_vel_rel = ObsTerm(func=mdp.ball_vel_rel)
        goal_dir_rel = ObsTerm(func=mdp.goal_dir_rel, params={"command_name": "ball_target_velocity"})
        ball_to_goal_dir_rel = ObsTerm(func=mdp.ball_to_goal_dir_rel, params={"command_name": "ball_target_velocity"})

        kick_phase = ObsTerm(func=mdp.kick_phase, params={"command_name": "kick_motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "kick_motion"})

        def __post_init__(self):
            self.history_length = 4

    critic: CriticCfg = CriticCfg()


# Minimal tracking bodies: pelvis, torso, kicking leg only (no support leg / upper body as main).
_TRACKING_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
]
_TRACKING_BODY_NAMES_NO_RIGHT_ANKLE = [
    "pelvis",
    "torso_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
]


@configclass
class RewardsCfg:
    # Stage A initial weights; curriculum overrides for B/C.
    reward_approach_ball = RewTerm(
        func=mdp.reward_approach_ball,
        weight=0.6,
        params={
            "target_distance": 0.20,
            "kick_zone_distance": 0.16,
        },
    )
    reward_kick_leg_swing = RewTerm(
        func=mdp.reward_kick_leg_swing,
        weight=12.0,
        params={"target_mode": "ball"},
    )
    reward_kick_foot_contact_ball = RewTerm(
        func=mdp.reward_kick_foot_contact_ball,
        weight=11.0,
        params={"min_approach_speed": 0.10, "contact_distance": 0.14},
    )
    reward_first_clean_contact_bonus = RewTerm(
        func=mdp.reward_first_clean_contact_bonus,
        weight=8.0,
        params={"min_approach_speed": 0.18, "contact_distance": 0.14},
    )
    reward_ball_speed = RewTerm(func=mdp.reward_ball_speed, weight=14.0, params={"min_speed": 0.08})
    reward_ball_impulse = RewTerm(func=mdp.reward_ball_impulse, weight=10.0, params={"window_steps": 10})
    reward_ball_goal_direction = RewTerm(
        func=mdp.reward_ball_goal_direction,
        weight=0.0,
        params={"angle_threshold_deg": 25.0},
    )
    reward_ball_goal_speed = RewTerm(func=mdp.reward_ball_goal_speed, weight=0.0)

    tracking_anchor_pos = RewTerm(func=mdp.tracking_anchor_pos_gated, weight=0.20)
    tracking_anchor_ori = RewTerm(func=mdp.tracking_anchor_ori_gated, weight=0.18)
    tracking_body_pos = RewTerm(
        func=mdp.tracking_body_pos_gated,
        weight=0.26,
        params={"body_names": _TRACKING_BODY_NAMES},
    )
    tracking_body_ori = RewTerm(
        func=mdp.tracking_body_ori_gated,
        weight=0.22,
        params={"body_names": _TRACKING_BODY_NAMES},
    )
    tracking_body_vel = RewTerm(
        func=mdp.tracking_body_vel_gated,
        weight=0.18,
        params={"body_names": _TRACKING_BODY_NAMES},
    )

    penalty_excess_travel = RewTerm(func=mdp.penalty_excess_travel, weight=-0.02)
    penalty_bad_ball_contact = RewTerm(func=mdp.penalty_bad_ball_contact, weight=-1.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-0.04)

    # All-stage anti-toe terms (A/B/C), weighted by curriculum.
    penalty_right_ankle_pitch_staged = RewTerm(func=mdp.penalty_right_ankle_pitch_staged, weight=0.0)
    penalty_right_toe_only_contact = RewTerm(func=mdp.penalty_right_toe_only_contact, weight=0.0)
    penalty_right_toe_dominant_force = RewTerm(func=mdp.penalty_right_toe_dominant_force, weight=0.0)
    reward_right_foot_parallel = RewTerm(func=mdp.reward_right_foot_parallel, weight=0.0)

    # Post-kick recovery rewards / penalties.
    # NOTE: Stage A keeps these effectively off via curriculum weights.
    reward_post_kick_upright = RewTerm(
        func=mdp.reward_post_kick_upright,
        weight=0.0,
    )
    reward_post_kick_base_height = RewTerm(
        func=mdp.reward_post_kick_base_height,
        weight=0.0,
    )
    reward_kick_leg_retract = RewTerm(
        func=mdp.reward_kick_leg_retract,
        weight=0.0,
    )
    reward_post_kick_recontact = RewTerm(
        func=mdp.reward_post_kick_recontact,
        weight=0.0,
    )
    reward_post_kick_velocity_damping = RewTerm(
        func=mdp.reward_post_kick_velocity_damping,
        weight=0.0,
    )
    reward_post_kick_joint_nominal = RewTerm(
        func=mdp.reward_post_kick_joint_nominal,
        weight=0.0,
    )
    penalty_leg_spread = RewTerm(
        func=mdp.penalty_leg_spread,
        weight=0.0,
    )
    penalty_post_joint_limit_stronger = RewTerm(
        func=mdp.penalty_post_joint_limit_stronger,
        weight=0.0,
    )

    # New: anti-crouch and "stand up & stand stably" rewards (only active in B/C via curriculum).
    penalty_post_kick_crouch = RewTerm(
        func=mdp.penalty_post_kick_crouch,
        weight=0.0,
    )
    reward_post_kick_stand_height = RewTerm(
        func=mdp.reward_post_kick_stand_height,
        weight=0.0,
    )
    reward_post_kick_stable_stand = RewTerm(
        func=mdp.reward_post_kick_stable_stand,
        weight=0.0,
    )

    # Right support foot (左脚踢球 右脚支撑): only active in Stage B/C via curriculum.
    penalty_support_toe_stance = RewTerm(
        func=mdp.penalty_support_toe_stance,
        weight=0.0,
    )
    reward_support_foot_flat_contact = RewTerm(
        func=mdp.reward_support_foot_flat_contact,
        weight=0.0,
    )
    reward_support_foot_stability = RewTerm(
        func=mdp.reward_support_foot_stability,
        weight=0.0,
    )
    penalty_support_knee_drop = RewTerm(
        func=mdp.penalty_support_knee_drop,
        weight=0.0,
    )

    # Right support foot geometry (Stage B/C): flat/parallel, toe scrape, yaw alignment, stumble force.
    reward_support_foot_parallel = RewTerm(func=mdp.reward_support_foot_parallel, weight=0.0)
    reward_support_foot_parallel_rp = RewTerm(func=mdp.reward_support_foot_parallel_rp, weight=0.0)
    penalty_support_foot_toe_scrape = RewTerm(func=mdp.penalty_support_foot_toe_scrape, weight=0.0)
    reward_support_foot_yaw = RewTerm(func=mdp.reward_support_foot_yaw, weight=0.0)
    penalty_support_foot_stumble = RewTerm(
        func=mdp.penalty_support_foot_stumble,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    # Diagnostics (weight=0): Diag/right_support_ankle_pitch_mean, etc.
    # Keep tiny non-zero weights so diagnostics are visible in RewardManager logs/TensorBoard.
    metric_diag_right_support_ankle_pitch_mean = RewTerm(
        func=mdp.metric_diag_right_support_ankle_pitch_mean, weight=0.1
    )
    metric_diag_right_knee_flex_mean = RewTerm(
        func=mdp.metric_diag_right_knee_flex_mean, weight=0.1
    )
    metric_diag_right_support_foot_ang_vel_mean = RewTerm(
        func=mdp.metric_diag_right_support_foot_ang_vel_mean, weight=0.1
    )
    metric_diag_right_support_flat_rate = RewTerm(
        func=mdp.metric_diag_right_support_flat_rate, weight=0.1
    )
    metric_diag_right_support_foot_pitch_mean = RewTerm(
        func=mdp.metric_diag_right_support_foot_pitch_mean, weight=0.1
    )
    metric_diag_right_support_yaw_err_mean = RewTerm(func=mdp.metric_diag_right_support_yaw_err_mean, weight=0.1)
    metric_diag_right_toe_height = RewTerm(func=mdp.metric_diag_right_toe_height, weight=0.1)
    metric_diag_right_heel_height = RewTerm(func=mdp.metric_diag_right_heel_height, weight=0.1)
    metric_diag_right_heel_minus_toe_height = RewTerm(
        func=mdp.metric_diag_right_heel_minus_toe_height, weight=0.1
    )
    metric_diag_right_support_stumble_force_ratio = RewTerm(
        func=mdp.metric_diag_right_support_stumble_force_ratio, weight=0.1
    )

    metric_gate_dist = RewTerm(func=mdp.metric_gate_dist, weight=0.0)
    metric_gate_phase = RewTerm(func=mdp.metric_gate_phase, weight=0.0)
    metric_gate_leg = RewTerm(func=mdp.metric_gate_leg, weight=0.0)
    metric_gate_contact = RewTerm(func=mdp.metric_gate_contact, weight=0.0)

    metric_curriculum_stage = RewTerm(func=mdp.metric_curriculum_stage, weight=0.0)
    metric_curriculum_steps_in_stage = RewTerm(func=mdp.metric_curriculum_steps_in_stage, weight=0.0)
    metric_curriculum_promotion_fired = RewTerm(func=mdp.metric_curriculum_promotion_fired, weight=0.0)
    metric_curriculum_demotion_fired = RewTerm(func=mdp.metric_curriculum_demotion_fired, weight=0.0)
    metric_curriculum_recovery_quality = RewTerm(func=mdp.metric_curriculum_recovery_quality, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Height/orientation terminations are wrapped to give Stage B/C a post-kick survival window.
    # Stage A still uses the original thresholds and behavior inside the wrappers.
    base_height = DoneTerm(
        func=mdp.kick_root_height_below_minimum_with_window,
        params={"minimum_height": 0.22, "grace_window_s": 3.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.kick_bad_orientation_with_window,
        params={"limit_angle": 1.0, "grace_window_s": 3.0},
    )
    # NOTE: We intentionally do NOT early-terminate on \"good\" kicks anymore.
    # Ball flying out should not end the episode if the robot is still standing;
    # let episodes run to max_episode_length_s (or until base_height / bad_orientation triggers).
    # If needed for debugging, re-enable with:
    # early_kick_finish = DoneTerm(
    #     func=mdp.early_terminate_after_kick,
    #     params={\"eval_window_steps\": 16, \"min_ball_speed\": 0.6},
    # )
    bad_ball_stuck = DoneTerm(
        func=mdp.bad_ball_stuck,
        params={"min_eval_steps": 12, "max_ball_speed": 0.12, "max_ball_distance": 0.20},
    )


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.22, 0.22)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.12, 0.12),
            "velocity_range": (-0.10, 0.10),
        },
    )

    reset_ball = EventTerm(
        func=mdp.reset_ball_state,
        mode="reset",
        params={
            # 把球放在机器人“面朝方向”一侧：如果机器人视觉上朝 -X，则球放在 -X 方向。
            # StageB asked: move initial ball 8cm backward along x.
            "position_range": {"x": (0.12, 0.15), "y": (0.02, 0.04), "z": (0.00, 0.01)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )


@configclass
class CurriculumCfg:
    kick_skill = CurrTerm(func=mdp.kick_skill_curriculum)


@configclass
class KickSkillEnvCfg(ManagerBasedRLEnvCfg):
    scene: SoccerBallSceneCfg = SoccerBallSceneCfg(num_envs=4096, env_spacing=3.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        # Let each episode run long enough after the kick for post-kick recovery to unfold.
        # 3.0s @ 0.02s env step ≈ 150 steps per episode.
        self.episode_length_s = 5.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        # Optional debug view for contact points/forces (set env var KICK_CONTACT_DEBUG_VIS=1).
        # Default off so existing training behavior is unchanged.
        self.scene.contact_forces.debug_vis = os.getenv("KICK_CONTACT_DEBUG_VIS", "0") == "1"


@configclass
class KickSkillPlayEnvCfg(KickSkillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
