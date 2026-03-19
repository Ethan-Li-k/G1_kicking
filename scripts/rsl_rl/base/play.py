# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_visible_envs",
    type=int,
    default=32,
    help="Number of environments to keep visible in recorded videos (others are hidden from rendering only).",
)
parser.add_argument(
    "--video_render_spacing",
    type=float,
    default=3.0,
    help="Grid spacing used for video rendering layout when showing a subset of environments.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--camera_follow_offset",
    type=float,
    nargs=3,
    default=(-2.2, 0.0, 0.9),
    help="Camera follow offset (x y z) in robot local frame for centered close-up videos.",
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
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import rsl_rl_utils
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import kick_task  # noqa: F401


def _set_kick_force_stage_from_cli(task: str, kick_stage: str | None) -> None:
    """Configure kick curriculum behavior from CLI for play.

    Same rules as train.py:
    - g1-kick-skill-amp:
        * default: full A/B/C curriculum
        * with --kick_stage: hard-pin that stage via KICK_SKILL_FORCE_STAGE
    - g1-kick-skill-stageB-amp:
        * default: start from Stage B and allow B↔C curriculum only (no pin)
        * with --kick_stage: honor explicit pin
    - g1-kick-skill-stageC-amp:
        * default: hard-pin Stage C
        * with --kick_stage: honor explicit pin
    """
    if kick_stage is not None:
        os.environ["KICK_SKILL_FORCE_STAGE"] = kick_stage.upper()
        return

    if task == "g1-kick-skill-stageB-amp":
        os.environ["KICK_SKILL_START_STAGE"] = "B"
    elif task == "g1-kick-skill-stageC-amp":
        os.environ["KICK_SKILL_FORCE_STAGE"] = "C"


def main():
    """Play with RSL-RL agent."""
    _set_kick_force_stage_from_cli(args_cli.task, getattr(args_cli, "kick_stage", None))
    # parse configuration
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
            entry_point_key="play_env_cfg_entry_point",
        )
    except TypeError as e:
        if "entry_point_key" not in str(e):
            raise
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # make a smaller scene for play
    # env_cfg.scene.num_envs = 50
    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        controller = Se2Keyboard(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # when recording videos, keep only a subset of environments visible to reduce visual clutter
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

            # apply visibility only to primary scene assets
            robot_asset = scene.articulations.get("robot", None)
            ball_asset = scene.rigid_objects.get("ball", None)

            if robot_asset is not None and hasattr(robot_asset, "set_visibility"):
                robot_asset.set_visibility(False, env_ids=hidden_env_ids)
                robot_asset.set_visibility(True, env_ids=visible_env_ids)

            if ball_asset is not None and hasattr(ball_asset, "set_visibility"):
                ball_asset.set_visibility(False, env_ids=hidden_env_ids)
                ball_asset.set_visibility(True, env_ids=visible_env_ids)

            print(
                f"[INFO] Video rendering limited to centered {visible_envs}/{total_envs} environments "
                "(physics still runs for all environments)."
            )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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
        env = AMPEnvWrapper(env, clip_actions=agent_cfg.clip_actions, motion_dataset=motion_dataset)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model (no optimizer for inference)
    if args_cli.amp:
        from rsl_rl_amp.runners import AMPOnPolicyRunner
        ppo_runner: OnPolicyRunner = AMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        from rsl_rl.runners import OnPolicyRunner
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    obs_normalizer = getattr(ppo_runner, "obs_normalizer", None)
    if obs_normalizer is None:
        obs_normalizer = getattr(policy_nn, "actor_obs_normalizer", None)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy=policy_nn,
        normalizer=obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    export_policy_as_jit(
        policy=policy_nn,
        normalizer=obs_normalizer,
        path=export_model_dir,
        filename="policy.pt",
    )
    
    print(f"[INFO] Joint orders: {env.unwrapped.scene['robot'].joint_names}")

    dt = env.unwrapped.step_dt

    # reset environment
    if args_cli.amp:
        obs = env.get_observations()
    else:
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.video or args_cli.keyboard:
            rsl_rl_utils.camera_follow(env, camera_offset=args_cli.camera_follow_offset)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
