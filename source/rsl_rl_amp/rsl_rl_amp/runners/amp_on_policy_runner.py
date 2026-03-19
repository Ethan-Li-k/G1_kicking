import time
import os
from collections import deque
import statistics
import re

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl_amp.algorithms import AMPPPO, PPO, AMPPPOWeighted
from rsl_rl_amp.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl_amp.env import VecEnv
from rsl_rl_amp.modules.amp_discriminator import AMPDiscriminator
from rsl_rl_amp.utils.utils import Normalizer
from beyondAMP.isaaclab.rsl_rl.amp_wrapper import AMPEnvWrapper

from beyondAMP.motion.motion_dataset import MotionDataset

class AMPOnPolicyRunner:
    TB_PREFIXES = ('Diag/', 'Reward/', 'Term/', 'Curriculum/', 'Loss/', 'Train/', 'Policy/', 'Perf/', 'Misc/')
    DIAG_PRIORITY = [
        'Diag/attack_minus_jitter',
        'Diag/hesitation_gap',
        'Diag/directional_release_score',
        'Diag/upper_body_zombie_score',
        'Diag/attack_chain_score',
        'Diag/jitter_score',
        'Diag/stage1_progress_minus_freeze',
        'Diag/stage1_forward_progress_score',
        'Diag/stage1_freeze_score',
    ]
    REWARD_PRIORITY = [
        'Reward/stage2_pre_shot_pose',
        'Reward/stage2_strike_quality',
        'Reward/post_shot_stability',
        'Reward/stage2_miss_direction',
        'Reward/action_rate',
        'Reward/upper_body_pose_deviation',
        'Reward/upper_body_motion',
        'Reward/stage1_walk_to_ball_simple',
        'Reward/stage1_freeze_and_crowd',
    ]
    TERM_PRIORITY = [
        'Term/time_out',
        'Term/base_height',
        'Term/bad_orientation',
        'Term/knee_contact',
        'Term/ball_out_of_bounds',
    ]
    CURRICULUM_PRIORITY = [
        'Curriculum/stage2_progression',
        'Curriculum/direction_window_deg',
        'Curriculum/ball_speed_level',
        'Curriculum/jitter_penalty_scale',
        'Curriculum/motor_scale_level',
        'Curriculum/stage1_walk_progression',
    ]

    def _route_episode_key(self, raw_key: str) -> str:
        key = raw_key.strip()
        if key.startswith(self.TB_PREFIXES):
            return key

        if key.startswith('Episode_'):
            key = key[len('Episode_'):]

        diag_name_map = {
            'metric_stage2_attack_minus_jitter': 'attack_minus_jitter',
            'metric_stage2_hesitation_gap': 'hesitation_gap',
            'metric_stage2_directional_release_score': 'directional_release_score',
            'metric_stage2_upper_body_zombie_score': 'upper_body_zombie_score',
            'metric_stage2_attack_chain_score': 'attack_chain_score',
            'metric_stage2_jitter_score': 'jitter_score',
            'stage2_attack_minus_jitter': 'attack_minus_jitter',
            'stage2_hesitation_gap': 'hesitation_gap',
            'stage2_directional_release_score': 'directional_release_score',
            'stage2_upper_body_zombie_score': 'upper_body_zombie_score',
            'stage2_attack_chain_score': 'attack_chain_score',
            'stage2_jitter_score': 'jitter_score',
            'attack_minus_jitter': 'attack_minus_jitter',
            'hesitation_gap': 'hesitation_gap',
            'directional_release_score': 'directional_release_score',
            'upper_body_zombie_score': 'upper_body_zombie_score',
            'attack_chain_score': 'attack_chain_score',
            'jitter_score': 'jitter_score',
            'metric_stage1_progress_minus_freeze': 'stage1_progress_minus_freeze',
            'metric_stage1_forward_progress_score': 'stage1_forward_progress_score',
            'metric_stage1_freeze_score': 'stage1_freeze_score',
            'stage1_progress_minus_freeze': 'stage1_progress_minus_freeze',
            'stage1_forward_progress_score': 'stage1_forward_progress_score',
            'stage1_freeze_score': 'stage1_freeze_score',
        }
        leaf_from_any_prefix = key.split('/')[-1].lower()
        if leaf_from_any_prefix in diag_name_map:
            return f"Diag/{diag_name_map[leaf_from_any_prefix]}"

        if key.startswith('Reward/'):
            reward_leaf = key.split('/', 1)[1]
            reward_aliases = {
                'reward_stage2_pre_shot_pose': 'stage2_pre_shot_pose',
                'reward_stage2_strike_quality': 'stage2_strike_quality',
                'reward_post_shot_stability': 'post_shot_stability',
                'penalty_stage2_miss_direction': 'stage2_miss_direction',
                'penalty_stage1_freeze_and_crowd': 'stage1_freeze_and_crowd',
            }
            reward_leaf = reward_aliases.get(reward_leaf, reward_leaf)
            if reward_leaf.startswith('reward_'):
                reward_leaf = reward_leaf[len('reward_'):]
            if reward_leaf.startswith('penalty_'):
                reward_leaf = reward_leaf[len('penalty_'):]
            return f'Reward/{reward_leaf}'
        if key.startswith('Termination/'):
            return 'Term/' + key.split('/', 1)[1]
        if key.startswith('Term/'):
            return key
        if key.startswith('Curriculum/'):
            curriculum_leaf = key.split('/', 1)[1]
            curriculum_aliases = {
                'ball_velocity_levels': 'ball_speed_level',
            }
            curriculum_leaf = curriculum_aliases.get(curriculum_leaf, curriculum_leaf)
            return f'Curriculum/{curriculum_leaf}'
        if key.startswith('Diag/'):
            return key
        if key.startswith('Metrics/'):
            metric_name = key.split('/', 1)[1]
            metric_name_l = metric_name.lower()
            if metric_name_l in diag_name_map:
                return f"Diag/{diag_name_map[metric_name_l]}"
            if metric_name_l.startswith('metric_stage1_'):
                return f"Diag/{metric_name_l[len('metric_stage1_'):]}"
            if metric_name_l.startswith('metric_stage2_'):
                return f"Diag/{metric_name_l[len('metric_stage2_'):]}"
            if metric_name.startswith('metric_stage2_'):
                metric_name = metric_name[len('metric_stage2_'):]
            if metric_name in diag_name_map.values():
                return f'Diag/{metric_name}'
            return f'Diag/detail/{metric_name}'

        leaf = key.split('/')[-1]
        name = leaf.lower()

        if name in diag_name_map:
            return f"Diag/{diag_name_map[name]}"

        if name.startswith('reward_'):
            return f"Reward/{leaf[len('reward_'):]}"
        if name.startswith('penalty_'):
            return f"Reward/{leaf[len('penalty_'):]}"

        term_suffixes = {'time_out', 'base_height', 'bad_orientation', 'knee_contact', 'ball_out_of_bounds'}
        if name.startswith('termination_'):
            return f"Term/{leaf[len('termination_'):]}"
        if name in term_suffixes:
            return f'Term/{leaf}'

        if name.startswith('curriculum_'):
            return f"Curriculum/{leaf[len('curriculum_'):]}"
        if name.startswith(('progression', 'level', 'window')):
            return f'Curriculum/{leaf}'

        if name.startswith('metric_'):
            if name.startswith('metric_stage1_'):
                return f"Diag/{leaf[len('metric_stage1_'):]}"
            if name.startswith('metric_stage2_'):
                return f"Diag/{leaf[len('metric_stage2_'):]}"
            diag_like = ('score', 'gap', 'hesitation', 'attack', 'jitter', 'release', 'zombie')
            if any(token in name for token in diag_like):
                clean = re.sub(r'^metric_stage2_', '', leaf)
                clean = re.sub(r'^metric_stage1_', '', clean)
                clean = re.sub(r'^metric_', '', clean)
                return f'Diag/{clean}'
            clean = re.sub(r'^metric_', '', leaf)
            return f'Diag/detail/{clean}'

        return f'Misc/{leaf}'

    def _ordered_tags(self, category_to_scalars):
        ordered = []
        priority = {
            'Diag': self.DIAG_PRIORITY,
            'Reward': self.REWARD_PRIORITY,
            'Term': self.TERM_PRIORITY,
            'Curriculum': self.CURRICULUM_PRIORITY,
        }
        category_order = ['Diag', 'Reward', 'Term', 'Curriculum', 'Misc']
        for category in category_order:
            tags = [tag for tag in category_to_scalars if tag.startswith(f'{category}/')]
            if not tags:
                continue
            tag_set = set(tags)
            for tag in priority.get(category, []):
                if tag in tag_set and tag not in ordered:
                    ordered.append(tag)
            for tag in sorted(tags):
                if tag not in ordered:
                    ordered.append(tag)
        return ordered

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.amp_data_cfg = train_cfg["amp_data"]
        self.device = device
        self.env:AMPEnvWrapper = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.policy_cfg["class_name"]) # ActorCritic
        num_actor_obs = self.env.num_obs
        actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)

        amp_data = env.motion_dataset
        amp_obs_dim = env.get_amp_observations().shape[-1] # amp_data.observation_dim
        amp_normalizer = Normalizer(amp_obs_dim)
        discriminator = AMPDiscriminator(
            amp_obs_dim * 2,
            train_cfg['amp_reward_coef'],
            train_cfg['amp_discr_hidden_dims'], device,
            train_cfg['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.alg_cfg["class_name"]) # PPO
        min_std = (
            torch.tensor(self.cfg["amp_min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[0, :, 1] - self.env.dof_pos_limits[0, :, 0])))
        self.alg: AMPPPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        ampbuffer = deque(maxlen=100)
        discribuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_amp_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_discri_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) 
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions, not_amp=False)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    lerp_rewards, d_logits, amp_rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)
                    d_logits = d_logits.squeeze(-1)
                    amp_rewards = amp_rewards.squeeze(-1)
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(lerp_rewards, dones, infos, next_amp_obs_with_term)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'log' in infos:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        cur_amp_sum += amp_rewards
                        cur_discri_sum += d_logits
                        cur_episode_length += 1
                        self.log_loc(cur_reward_sum, dones, rewbuffer)
                        self.log_loc(cur_amp_sum, dones, ampbuffer)
                        self.log_loc(cur_discri_sum, dones, discribuffer)
                        self.log_loc(cur_episode_length, dones, lenbuffer)

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, \
            mean_amp_loss, mean_grad_pen_loss, \
            mean_policy_pred, mean_expert_pred, \
            mean_approx_kl, mean_entropy, mean_clip_fraction, \
            mean_action_std, mean_action_mean_abs = \
                self.alg.update()
                
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log_loc(self, cur_sum, dones, buffer):
        new_ids = (dones > 0).nonzero(as_tuple=False)
        buffer.extend(cur_sum[new_ids][:, 0].cpu().numpy().tolist())
        cur_sum[new_ids] = 0

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_scalars = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                routed_key = self._route_episode_key(key)
                ep_scalars[routed_key] = value.item()

        for tag in self._ordered_tags(ep_scalars):
            self.writer.add_scalar(tag, ep_scalars[tag], locs['it'])

        ep_category_order = ['Diag', 'Reward', 'Term', 'Curriculum', 'Misc']
        ep_string = ''
        for category in ep_category_order:
            category_tags = [tag for tag in self._ordered_tags(ep_scalars) if tag.startswith(f'{category}/')]
            if not category_tags:
                continue
            for tag in category_tags:
                ep_string += f"""{f'Mean {tag}:':>{pad}} {ep_scalars[tag]:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/approx_kl', locs['mean_approx_kl'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy'], locs['it'])
        self.writer.add_scalar('Loss/clip_fraction', locs['mean_clip_fraction'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Policy/action_std', locs['mean_action_std'], locs['it'])
        self.writer.add_scalar('Policy/action_mean_abs', locs['mean_action_mean_abs'], locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_amp_reward', statistics.mean(locs['ampbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_discri_logits', statistics.mean(locs['discribuffer']), locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Approx KL:':>{pad}} {locs['mean_approx_kl']:.6f}\n"""
                          f"""{'Entropy:':>{pad}} {locs['mean_entropy']:.4f}\n"""
                          f"""{'Clip fraction:':>{pad}} {locs['mean_clip_fraction']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Action std:':>{pad}} {locs['mean_action_std']:.4f}\n"""
                          f"""{'Action mean abs:':>{pad}} {locs['mean_action_mean_abs']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                                                    f"""{'Approx KL:':>{pad}} {locs['mean_approx_kl']:.6f}\n"""
                                                    f"""{'Entropy:':>{pad}} {locs['mean_entropy']:.4f}\n"""
                                                    f"""{'Clip fraction:':>{pad}} {locs['mean_clip_fraction']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
