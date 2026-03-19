from __future__ import annotations

import math
import os
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import yaw_quat


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: list[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid motion_file: {motion_file}"
        data = np.load(motion_file)
        self.fps = float(data["fps"])
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class BallCenterVelocityCommand(UniformVelocityCommand):
    cfg: UniformLevelVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        goal_dir = torch.tensor(self.cfg.goal_direction_xy, device=self.device, dtype=torch.float32)
        goal_norm = torch.linalg.norm(goal_dir).clamp_min(1e-6)
        self._goal_dir_xy = goal_dir / goal_norm
        self._goal_heading = torch.atan2(self._goal_dir_xy[1], self._goal_dir_xy[0])
        self._goal_arrow_quat_single = math_utils.quat_from_euler_xyz(
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
            self._goal_heading,
        ).view(1, 4)
        self._vel_heading_cache = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if self.cfg.goal_heading_range_deg is None or self.cfg.goal_speed_range is None:
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.numel() == 0:
            return

        heading_min_deg, heading_max_deg = self.cfg.goal_heading_range_deg
        heading_min = math.radians(heading_min_deg)
        heading_max = math.radians(heading_max_deg)
        heading = torch.empty(env_ids_tensor.numel(), device=self.device).uniform_(heading_min, heading_max)
        speed = torch.empty(env_ids_tensor.numel(), device=self.device).uniform_(*self.cfg.goal_speed_range)

        self.vel_command_b[env_ids_tensor, 0] = speed * torch.cos(heading)
        self.vel_command_b[env_ids_tensor, 1] = speed * torch.sin(heading)

        standing_mask = self.is_standing_env[env_ids_tensor]
        self.vel_command_b[env_ids_tensor[standing_mask], :2] = 0.0

    def _debug_vis_callback(self, event):
        # Place arrows at ball center and only use world-frame XY velocity.
        # This keeps arrows from rotating with the ball's own orientation/spin.
        ball = self._env.scene["ball"]
        if not ball.is_initialized:
            return

        base_pos_w = ball.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.2

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_goal_direction_to_arrow(self.num_envs)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            ball.data.root_lin_vel_w[:, :2], self._vel_heading_cache
        )

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, heading_cache: torch.Tensor, min_speed: float = 0.08
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = getattr(self.goal_vel_visualizer.cfg.markers["arrow"], "scale", (1.0, 1.0, 1.0))
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        speed = torch.linalg.norm(xy_velocity, dim=1)
        arrow_scale[:, 0] *= speed * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        valid_mask = speed > min_speed
        heading_cache[valid_mask] = heading_angle[valid_mask]
        heading_angle = heading_cache
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, arrow_quat

    def _resolve_goal_direction_to_arrow(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = getattr(self.goal_vel_visualizer.cfg.markers["arrow"], "scale", (1.0, 1.0, 1.0))
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(batch_size, 1)
        arrow_quat = self._goal_arrow_quat_single.repeat(batch_size, 1)
        return arrow_scale, arrow_quat


class KickMotionCommand(CommandTerm):
    cfg: KickMotionCommandCfg

    def __init__(self, cfg: "KickMotionCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = list(self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0])

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        num_bodies = len(cfg.body_names)
        self.body_pos_relative_w = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, num_bodies, 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["kick_phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_strike_gate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def phase(self) -> torch.Tensor:
        denom = max(int(self.motion.time_step_total) - 1, 1)
        return self.time_steps.float() / float(denom)

    @property
    def strike_gate(self) -> torch.Tensor:
        phase = self.phase
        start, end = self.cfg.strike_phase_window
        return ((phase >= start) & (phase <= end)).float()

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids_seq = env_ids_t.tolist()
        # Phase/frame initialization:
        # - Default (existing behavior): start from phase 0.0 (time_step = 0).
        # - Optional: if cfg.start_phase_range is provided, uniformly sample an
        #   integer frame index whose normalized phase lies within [start, end].
        if getattr(self.cfg, "start_phase_range", None) is None:
            self.time_steps[env_ids_t] = 0
        else:
            start_phase, end_phase = self.cfg.start_phase_range
            total_steps = int(self.motion.time_step_total)
            denom = max(total_steps - 1, 1)
            start_idx = int(max(0, min(denom, math.floor(start_phase * denom))))
            end_idx = int(max(0, min(denom, math.floor(end_phase * denom))))
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            if end_idx == start_idx:
                sampled = torch.full_like(env_ids_t, start_idx, dtype=torch.long)
            else:
                sampled = torch.randint(
                    low=start_idx,
                    high=end_idx + 1,
                    size=(env_ids_t.numel(),),
                    device=self.device,
                    dtype=torch.long,
                )
            self.time_steps[env_ids_t] = sampled

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        pose_ranges = torch.tensor(
            [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        pose_samples = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids_t), 6), device=self.device)
        root_pos[env_ids_t] += pose_samples[:, 0:3]
        root_ori_delta = quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
        root_ori[env_ids_t] = quat_mul(root_ori_delta, root_ori[env_ids_t])

        vel_ranges = torch.tensor(
            [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        vel_samples = sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids_t), 6), device=self.device)
        root_lin_vel[env_ids_t] += vel_samples[:, :3]
        root_ang_vel[env_ids_t] += vel_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        limits = self.robot.data.soft_joint_pos_limits[env_ids_t]
        joint_pos[env_ids_t] = torch.clip(joint_pos[env_ids_t], limits[:, :, 0], limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos[env_ids_t], joint_vel[env_ids_t], env_ids=env_ids_t)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids_t], root_ori[env_ids_t], root_lin_vel[env_ids_t], root_ang_vel[env_ids_t]], dim=-1),
            env_ids=env_ids_t,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids.tolist())

        num_bodies = len(self.cfg.body_names)
        anchor_pos_rep = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        anchor_quat_rep = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_pos_rep = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_quat_rep = self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

        delta_pos_w = robot_anchor_pos_rep
        delta_pos_w[..., 2] = anchor_pos_rep[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_rep, quat_inv(anchor_quat_rep)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_rep)

        self.metrics["kick_phase"] = self.phase
        self.metrics["kick_strike_gate"] = self.strike_gate

    def _update_metrics(self):
        self.metrics["kick_phase"] = self.phase
        self.metrics["kick_strike_gate"] = self.strike_gate

    def _set_debug_vis_impl(self, debug_vis: bool):
        return


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = BallCenterVelocityCommand

    goal_heading_range_deg: tuple[float, float] | None = None
    goal_speed_range: tuple[float, float] | None = None
    goal_direction_xy: tuple[float, float] = (1.0, 0.0)

    limit_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(0.0, 0.0),
    )


@configclass
class KickMotionCommandCfg(CommandTermCfg):
    class_type: type = KickMotionCommand

    asset_name: str = "robot"
    motion_file: str = ""
    anchor_body_name: str = "torso_link"
    body_names: list[str] = []

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # Optional: restrict initial phase sampling window [start, end] in normalized phase units [0, 1].
    # If None, fall back to the original behavior (always start from phase 0.0 / first frame).
    # The bootstrap task uses this to sample only from stable-stand / pre-kick frames.
    start_phase_range: tuple[float, float] | None = None

    strike_phase_window: tuple[float, float] = (0.35, 0.62)
    pre_kick_phase_end: float = 0.34
    recover_phase_start: float = 0.72

    kicking_foot_body_name: str = "right_ankle_roll_link"
    supporting_foot_body_name: str = "left_ankle_roll_link"
    critical_body_names: list[str] = []
