# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import types

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--video_visible_envs",
    type=int,
    default=32,
    help="Number of environments to keep visible in recorded videos (others are hidden from rendering only).",
)
parser.add_argument(
    "--video_render_spacing",
    type=float,
    default=2.0,
    help="Grid spacing used for video rendering layout when showing a subset of environments.",
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--style_reward_weight", type=float, default=None, help="Weight of the style reward (if applicable)."
)
parser.add_argument(
    "--task_reward_weight", type=float, default=None, help="Weight of the style reward (if applicable)."
)
parser.add_argument(
    "--kick_stage",
    type=str,
    default=None,
    choices=["A", "B", "C"],
    help="Kick skill: force curriculum stage A/B/C (overrides task-based stage). Ignored for non–kick-skill tasks.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
import isaaclab.utils.math as math_utils

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import kick_task  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _set_kick_force_stage_from_cli(task: str, kick_stage: str | None) -> None:
    """Configure kick curriculum behavior from CLI.

    Rules:
    - g1-kick-skill-amp:
        * default: full A/B/C curriculum (no env override)
        * with --kick_stage: hard-pin that stage via KICK_SKILL_FORCE_STAGE
    - g1-kick-skill-stageB-amp:
        * default: start from Stage B and allow B↔C curriculum only (no pin)
        * with --kick_stage: honor explicit pin (debug only)
    - g1-kick-skill-stageC-amp:
        * default: hard-pin Stage C (pure C finetune)
        * with --kick_stage: honor explicit pin
    """
    # Explicit --kick_stage always means "hard pin this stage".
    if kick_stage is not None:
        os.environ["KICK_SKILL_FORCE_STAGE"] = kick_stage.upper()
        return

    if task == "g1-kick-skill-stageB-amp":
        # B-start curriculum: let curriculum start in B and move B↔C, but never visit A.
        os.environ["KICK_SKILL_START_STAGE"] = "B"
    elif task == "g1-kick-skill-stageC-amp":
        # Legacy behavior: pure Stage C finetune.
        os.environ["KICK_SKILL_FORCE_STAGE"] = "C"


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    _set_kick_force_stage_from_cli(args_cli.task, getattr(args_cli, "kick_stage", None))
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # when recording training videos, keep only a subset of environments visible to reduce visual clutter
    if args_cli.video and args_cli.video_visible_envs is not None and args_cli.video_visible_envs > 0:
        total_envs = env.unwrapped.scene.num_envs
        visible_envs = min(args_cli.video_visible_envs, total_envs)
        if visible_envs < total_envs:
            scene = env.unwrapped.scene

            # switch to a deterministic grid origin layout for clearer video composition
            if getattr(scene, "terrain", None) is not None and hasattr(scene.terrain, "configure_env_origins"):
                if args_cli.video_render_spacing is not None:
                    scene.terrain.cfg.env_spacing = float(args_cli.video_render_spacing)
                scene.terrain.configure_env_origins()

            # choose visible envs nearest to origin so all selected envs are likely inside camera view
            env_origins = scene.env_origins
            dist2 = torch.sum(env_origins[:, :2] * env_origins[:, :2], dim=-1)
            visible_env_ids = torch.argsort(dist2)[:visible_envs]

            all_env_ids = torch.arange(total_envs, device=env.unwrapped.device)
            visible_mask = torch.zeros(total_envs, dtype=torch.bool, device=env.unwrapped.device)
            visible_mask[visible_env_ids] = True
            hidden_env_ids = all_env_ids[~visible_mask]

            # apply visibility only to primary scene assets to avoid hiding unintended visuals
            robot_asset = scene.articulations.get("robot", None)
            ball_asset = scene.rigid_objects.get("ball", None)

            if robot_asset is not None and hasattr(robot_asset, "set_visibility"):
                robot_asset.set_visibility(False, env_ids=hidden_env_ids)
                robot_asset.set_visibility(True, env_ids=visible_env_ids)

            if ball_asset is not None and hasattr(ball_asset, "set_visibility"):
                ball_asset.set_visibility(False, env_ids=hidden_env_ids)
                ball_asset.set_visibility(True, env_ids=visible_env_ids)

            # keep command arrows enabled but only draw for visible envs
            command_terms = getattr(env.unwrapped.command_manager, "_terms", {})
            for command_term in command_terms.values():
                has_vis_attrs = all(
                    hasattr(command_term, attr)
                    for attr in [
                        "robot",
                        "command",
                        "goal_vel_visualizer",
                        "current_vel_visualizer",
                    ]
                )
                if not has_vis_attrs:
                    continue

                def _subset_debug_vis_callback(self, event, env_ids=visible_env_ids):
                    if not self.robot.is_initialized:
                        return
                    base_pos_w = self.robot.data.root_pos_w[env_ids].clone()
                    base_pos_w[:, 2] += 0.5

                    def _resolve_subset_xy_velocity_to_arrow(xy_velocity: torch.Tensor):
                        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
                        arrow_scale = torch.tensor(default_scale, device=xy_velocity.device).repeat(xy_velocity.shape[0], 1)
                        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

                        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
                        zeros = torch.zeros_like(heading_angle)
                        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
                        base_quat_w = self.robot.data.root_quat_w[env_ids]
                        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
                        return arrow_scale, arrow_quat

                    vel_des_arrow_scale, vel_des_arrow_quat = _resolve_subset_xy_velocity_to_arrow(self.command[env_ids, :2])
                    vel_arrow_scale, vel_arrow_quat = _resolve_subset_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[env_ids, :2])
                    self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
                    self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

                command_term._debug_vis_callback = types.MethodType(_subset_debug_vis_callback, command_term)

            print(
                f"[INFO] Video rendering limited to centered {visible_envs}/{total_envs} environments "
                "(physics/training still uses all environments)."
            )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl or amp-rsl-rl
    if args_cli.amp:
        from beyondAMP.isaaclab.rsl_rl.amp_wrapper import AMPEnvWrapper
        from beyondAMP.motion.motion_dataset import MotionDataset

        motion_dataset = MotionDataset(agent_cfg.amp_data, env.unwrapped, device=agent_cfg.device)
        env = AMPEnvWrapper(
            env,
            clip_actions=getattr(agent_cfg, "clip_actions", None),
            motion_dataset=motion_dataset,
        )
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    # create runner from rsl-rl
    if args_cli.amp:
        if args_cli.style_reward_weight is not None:
            agent_cfg.amp_reward_coef = args_cli.style_reward_weight
        if args_cli.task_reward_weight is not None:
            agent_cfg.amp_task_reward_lerp = args_cli.task_reward_weight

        from rsl_rl_amp.runners import AMPOnPolicyRunner
        runner: OnPolicyRunner = AMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    if hasattr(runner, "add_git_repo_to_log"):
        runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
