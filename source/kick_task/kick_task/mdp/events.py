from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def reset_ball_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    if position_range is None:
        position_range = {}
    if velocity_range is None:
        velocity_range = {}

    ball: RigidObject = env.scene[ball_cfg.name]

    root_states = ball.data.default_root_state[env_ids].clone()

    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([position_range.get(k, (0.0, 0.0)) for k in pose_keys], device=ball.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=ball.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_samples[:, 0:3]
    ori_delta = math_utils.quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], ori_delta)

    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in vel_keys], device=ball.device)
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=ball.device
    )
    velocities = root_states[:, 7:13] + vel_samples

    ball.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_and_ball_right_front(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    root_pose_range: dict[str, tuple[float, float]] | None = None,
    root_velocity_range: dict[str, tuple[float, float]] | None = None,
    ball_relative_position_range: dict[str, tuple[float, float]] | None = None,
    ball_velocity_range: dict[str, tuple[float, float]] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Reset robot root and place ball in robot right-front local region.

    Ball XY is sampled in the robot yaw frame, so placement stays consistent
    under randomized robot heading.
    """
    if root_pose_range is None:
        root_pose_range = {}
    if root_velocity_range is None:
        root_velocity_range = {}
    if ball_relative_position_range is None:
        ball_relative_position_range = {}
    if ball_velocity_range is None:
        ball_velocity_range = {}

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    # 1) Reset robot root state.
    robot_root = robot.data.default_root_state[env_ids].clone()
    root_pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    root_pose_ranges = torch.tensor([root_pose_range.get(k, (0.0, 0.0)) for k in root_pose_keys], device=robot.device)
    root_pose_samples = math_utils.sample_uniform(
        root_pose_ranges[:, 0], root_pose_ranges[:, 1], (len(env_ids), 6), device=robot.device
    )
    robot_pos = robot_root[:, 0:3] + env.scene.env_origins[env_ids] + root_pose_samples[:, 0:3]
    root_ori_delta = math_utils.quat_from_euler_xyz(
        root_pose_samples[:, 3], root_pose_samples[:, 4], root_pose_samples[:, 5]
    )
    robot_ori = math_utils.quat_mul(robot_root[:, 3:7], root_ori_delta)

    root_vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    root_vel_ranges = torch.tensor(
        [root_velocity_range.get(k, (0.0, 0.0)) for k in root_vel_keys], device=robot.device
    )
    root_vel_samples = math_utils.sample_uniform(
        root_vel_ranges[:, 0], root_vel_ranges[:, 1], (len(env_ids), 6), device=robot.device
    )
    robot_vel = robot_root[:, 7:13] + root_vel_samples
    robot.write_root_state_to_sim(torch.cat([robot_pos, robot_ori, robot_vel], dim=-1), env_ids=env_ids)

    # 2) Reset ball relative to robot yaw frame (right-front placement).
    ball_root = ball.data.default_root_state[env_ids].clone()

    rel_keys = ["x", "y", "z"]
    rel_ranges = torch.tensor([ball_relative_position_range.get(k, (0.0, 0.0)) for k in rel_keys], device=ball.device)
    rel_samples = math_utils.sample_uniform(rel_ranges[:, 0], rel_ranges[:, 1], (len(env_ids), 3), device=ball.device)

    # Rotate local XY offsets with robot yaw only, then place around robot XY.
    # IMPORTANT: ball Z should stay in world/terrain frame (not robot-root-relative),
    # otherwise the ball can spawn above the robot and fall onto it.
    yaw_only_quat = math_utils.yaw_quat(robot_ori)
    rel_world = math_utils.quat_apply(yaw_only_quat, rel_samples)
    ball_pos = ball_root[:, 0:3] + env.scene.env_origins[env_ids]
    ball_pos[:, :2] = robot_pos[:, :2] + rel_world[:, :2]
    ball_pos[:, 2] += rel_samples[:, 2]

    ball_ori = ball_root[:, 3:7]
    ball_vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    ball_vel_ranges = torch.tensor([ball_velocity_range.get(k, (0.0, 0.0)) for k in ball_vel_keys], device=ball.device)
    ball_vel_samples = math_utils.sample_uniform(
        ball_vel_ranges[:, 0], ball_vel_ranges[:, 1], (len(env_ids), 6), device=ball.device
    )
    ball_vel = ball_root[:, 7:13] + ball_vel_samples

    ball.write_root_pose_to_sim(torch.cat([ball_pos, ball_ori], dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(ball_vel, env_ids=env_ids)
