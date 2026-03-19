from __future__ import annotations

import torch
from typing import cast

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.math import yaw_quat

from .commands import KickMotionCommand


def _ball_robot(env: ManagerBasedRLEnv, ball_asset_name: str = "ball", robot_asset_name: str = "robot"):
    ball: RigidObject = env.scene[ball_asset_name]
    robot: Articulation = env.scene[robot_asset_name]
    return ball, robot


def ball_pos_rel(env: ManagerBasedRLEnv, ball_asset_name: str = "ball", robot_asset_name: str = "robot") -> torch.Tensor:
    ball, robot = _ball_robot(env, ball_asset_name, robot_asset_name)
    rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)


def ball_vel_rel(env: ManagerBasedRLEnv, ball_asset_name: str = "ball", robot_asset_name: str = "robot") -> torch.Tensor:
    ball, robot = _ball_robot(env, ball_asset_name, robot_asset_name)
    rel_vel_w = ball.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_vel_w)


def goal_dir_rel(env: ManagerBasedRLEnv, command_name: str = "ball_target_velocity", robot_asset_name: str = "robot") -> torch.Tensor:
    robot: Articulation = env.scene[robot_asset_name]
    goal_vec_w = env.command_manager.get_command(command_name)[:, :3]
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), goal_vec_w)


def ball_to_goal_dir_rel(
    env: ManagerBasedRLEnv,
    command_name: str = "ball_target_velocity",
    robot_asset_name: str = "robot",
    eps: float = 1e-6,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_asset_name]
    goal_vec_w = env.command_manager.get_command(command_name)[:, :2]
    goal_dir_w = goal_vec_w / torch.linalg.vector_norm(goal_vec_w, dim=1, keepdim=True).clamp(min=eps)
    goal_dir_w_3d = torch.zeros((env.num_envs, 3), device=goal_dir_w.device, dtype=goal_dir_w.dtype)
    goal_dir_w_3d[:, :2] = goal_dir_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), goal_dir_w_3d)


def kick_phase(env: ManagerBasedRLEnv, command_name: str = "kick_motion") -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    return command.phase.unsqueeze(-1)


def motion_anchor_pos_b(env: ManagerBasedRLEnv, command_name: str = "kick_motion") -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    anchor_rel_w = command.anchor_pos_w - command.robot_anchor_pos_w
    return quat_apply_inverse(yaw_quat(command.robot_anchor_quat_w), anchor_rel_w)
