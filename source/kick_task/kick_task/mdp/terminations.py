from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .rewards import kick_contact_gate


def _post_kick_elapsed(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (has_contact, elapsed_steps, valid_step_mask) using shared kick buffers.

    If no contact buffers are initialized yet, returns zero tensors so callers can
    treat it as "no valid contact so far".
    """
    kick_contact_gate(env)
    contact_latched = getattr(env, "_kick_contact_latched", None)
    contact_step = getattr(env, "_kick_contact_step", None)
    if contact_latched is None or contact_step is None:
        zero_bool = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        return zero_bool, torch.zeros(env.num_envs, device=env.device, dtype=torch.long), zero_bool

    valid_step = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(
        valid_step, contact_step, env.episode_length_buf.long()
    )
    return contact_latched, elapsed, valid_step


def early_terminate_after_kick(
    env: ManagerBasedRLEnv,
    eval_window_steps: int = 16,
    min_ball_speed: float = 0.6,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    kick_contact_gate(env)
    contact_latched = getattr(env, "_kick_contact_latched", None)
    contact_step = getattr(env, "_kick_contact_step", None)
    if contact_latched is None or contact_step is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    ball: RigidObject = env.scene[ball_cfg.name]
    contact_happened = contact_latched
    valid_step = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_step, contact_step, env.episode_length_buf.long())
    window_ready = elapsed >= eval_window_steps

    ball_speed = torch.linalg.vector_norm(ball.data.root_lin_vel_w[:, :2], dim=1)
    evaluated = contact_happened & window_ready
    return evaluated & (ball_speed >= min_ball_speed)


def bad_ball_stuck(
    env: ManagerBasedRLEnv,
    min_eval_steps: int = 12,
    max_ball_speed: float = 0.12,
    max_ball_distance: float = 0.20,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    kick_contact_gate(env)
    contact_latched = getattr(env, "_kick_contact_latched", None)
    contact_step = getattr(env, "_kick_contact_step", None)
    if contact_latched is None or contact_step is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    valid_step = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_step, contact_step, env.episode_length_buf.long())
    after_contact = contact_latched & (elapsed >= min_eval_steps)

    ball_speed = torch.linalg.vector_norm(ball.data.root_lin_vel_w[:, :2], dim=1)
    dist = torch.linalg.vector_norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    stuck = (ball_speed <= max_ball_speed) & (dist <= max_ball_distance)
    return after_contact & stuck


def kick_root_height_below_minimum_with_window(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.22,
    grace_window_s: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Height-based termination with a post-kick grace window for Stage B/C.

    - Stage A: behaves exactly like the upstream ``root_height_below_minimum``.
    - Stage B/C pre-contact: same as Stage A.
    - Stage B/C post-contact (within grace window): ignores this termination so that
      post-kick recovery has time to unfold (episodes end via time_out or bad_ball_stuck).
    """
    from isaaclab.envs.mdp import terminations as base_terms  # local import to avoid cycles

    stage = getattr(env, "_kick_curr_stage", "A")
    # Stage A: keep original behavior.
    if stage == "A":
        return base_terms.root_height_below_minimum(env, minimum_height=minimum_height, asset_cfg=asset_cfg)

    # For B/C, allow normal termination before any kick contact.
    contact_latched, elapsed, valid_step = _post_kick_elapsed(env)
    has_contact = contact_latched & valid_step
    if not has_contact.any():
        return base_terms.root_height_below_minimum(env, minimum_height=minimum_height, asset_cfg=asset_cfg)

    # After contact, within grace window, do not terminate based on height.
    # Use env.step_dt if available; otherwise assume 0.02s.
    step_dt = getattr(env, "step_dt", 0.02)
    grace_steps = int(grace_window_s / max(step_dt, 1e-6))
    in_grace = has_contact & (elapsed < grace_steps)

    base_term = base_terms.root_height_below_minimum(env, minimum_height=minimum_height, asset_cfg=asset_cfg)
    # Suppress termination for envs still inside the grace window.
    return base_term & ~in_grace


def kick_bad_orientation_with_window(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.0,
    grace_window_s: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Orientation-based termination with a post-kick grace window for Stage B/C.

    - Stage A: behaves exactly like upstream ``bad_orientation``.
    - Stage B/C pre-contact: same as Stage A.
    - Stage B/C post-contact (within grace window): ignores orientation termination so
      that the agent can explore recovery strategies.
    """
    from isaaclab.envs.mdp import terminations as base_terms  # local import to avoid cycles

    stage = getattr(env, "_kick_curr_stage", "A")
    if stage == "A":
        return base_terms.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)

    contact_latched, elapsed, valid_step = _post_kick_elapsed(env)
    has_contact = contact_latched & valid_step
    if not has_contact.any():
        return base_terms.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)

    step_dt = getattr(env, "step_dt", 0.02)
    grace_steps = int(grace_window_s / max(step_dt, 1e-6))
    in_grace = has_contact & (elapsed < grace_steps)

    base_term = base_terms.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)
    return base_term & ~in_grace
