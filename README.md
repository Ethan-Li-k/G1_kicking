# G1 Kicking: Humanoid Kick Skill with AMP

Unitree G1 humanoid kick skill training using Adversarial Motion Priors (AMP) on IsaacLab.


https://github.com/user-attachments/assets/53555206-3752-42f9-ba94-2ee34a956cf5


## Prerequisites

- **NVIDIA Isaac Sim** (2023.1.1+)
- **IsaacLab** installed and activated (`isaaclab`, `isaaclab_rl`, `isaaclab_tasks`, `rsl_rl`)

## Setup

```bash
git clone https://github.com/Ethan-Li-k/G1_kicking.git
cd G1_kicking
bash setup.sh
```

## Training

### Stage A (from scratch, with AMP)

```bash
python scripts/rsl_rl/base/train.py \
  --task g1-kick-skill-amp --amp --headless --num_envs 4096
```

### Stage B (resume from Stage A checkpoint)

```bash
python scripts/rsl_rl/base/train.py \
  --task g1-kick-skill-stageB-amp --amp --headless \
  --resume True --load_run <stage_A_run_folder> --checkpoint <model_xxx.pt>
```

Or use the helper script:

```bash
bash scripts/rsl_rl/base/run_kick_stageB.sh <stage_A_run_folder> <model_xxx.pt>
```

### Stage C (resume from Stage B checkpoint)

```bash
python scripts/rsl_rl/base/train.py \
  --task g1-kick-skill-stageC-amp --amp --headless \
  --resume True --load_run <stage_B_run_folder> --checkpoint <model_xxx.pt>
```

### Bootstrap (alternative from-scratch variant)

```bash
python scripts/rsl_rl/base/train.py \
  --task g1-kick-skill-bootstrap-amp --amp --headless
```

### Without AMP (pure PPO)

```bash
python scripts/rsl_rl/base/train.py --task g1-kick-skill --headless
```

## Evaluation

```bash
python scripts/rsl_rl/base/play.py \
  --task g1-kick-skill-amp --amp --num_envs 50 \
  --checkpoint <path/to/model.pt> --video
```

## Registered Tasks

| Task ID | Description |
|---------|-------------|
| `g1-kick-skill` | Pure PPO (no AMP) |
| `g1-kick-skill-amp` | Full A/B/C curriculum with AMP |
| `g1-kick-skill-bootstrap-amp` | Bootstrap variant (weaker AMP prior) |
| `g1-kick-skill-stageB-amp` | Stage B finetune (resume only) |
| `g1-kick-skill-stageC-amp` | Stage C finetune (resume only) |

## Known Issues & Tuning Guide

### Known Issues (v0.1)

The current version has two primary issues that are actively being worked on. A patch release is planned.

**1. Right foot tiptoe (右脚踮脚)**

During the kick phase, the right (support) foot tends to stand on its toes rather than maintaining flat contact with the ground. The `right_ankle_pitch_joint` drifts into excessive plantarflexion. Although multiple penalty terms have been implemented (`penalty_right_ankle_pitch_staged`, `penalty_right_toe_only_contact`, `penalty_right_toe_dominant_force`, `penalty_support_foot_toe_scrape`), the policy still finds tiptoe postures as a local optimum, especially in Stage A where anti-toe weights are relatively weak.

**2. Post-kick standing instability (踢后站立不稳)**

After completing the kick, the robot struggles to recover a stable standing posture. Common failure modes include deep crouching (knee over-flexion), slow base height recovery, and lateral sway. The Stage B/C curriculum introduces post-kick recovery rewards (`reward_post_kick_upright`, `reward_post_kick_base_height`, `reward_post_kick_stable_stand`, `penalty_post_kick_crouch`), but transitions from Stage A learned policies remain fragile.

### Tuning Guide

All reward weights are controlled through the curriculum in `kick_task/mdp/curriculums.py`. Key parameters by category:

**Anti-tiptoe (right support foot)**

| Parameter | Stage A | Stage B | Stage C | Effect |
|-----------|---------|---------|---------|--------|
| `penalty_right_ankle_pitch_staged` | -2.0 | -3.0 | -4.0 | Penalizes ankle pitch (toe-down). Increase magnitude to force flatter foot. |
| `penalty_right_toe_only_contact` | -2.5 | -3.5 | -4.5 | Penalizes toe-lower-than-heel geometry. |
| `penalty_right_toe_dominant_force` | -1.5 | -2.2 | -3.0 | Penalizes toe-dominant ground reaction. |
| `reward_support_foot_parallel_rp` | 0.0 | 3.0 | 4.0 | Rewards foot roll/pitch near zero. |

Tuning tips:
- If tiptoe persists in Stage A, increase `penalty_right_ankle_pitch_staged` to -4.0 or higher from the start. Be cautious: too strong will hurt kick power.
- The `margin` param in `penalty_right_toe_only_contact` (default 0.010m) controls sensitivity -- decrease it to penalize even slight toe-down.
- Consider adding anti-toe penalties to AMP motion data preprocessing (`kick_amp_data_cfg.py` has `_ANTI_TOE_FILTER` which downweights clips with bad toe frames).

**Post-kick recovery (standing stability)**

| Parameter | Stage B (B2) | Stage C | Effect |
|-----------|-------------|---------|--------|
| `reward_post_kick_upright` | 1.3 | 2.2 | Rewards torso staying vertical after kick. |
| `reward_post_kick_base_height` | 1.1 | 1.8 | Rewards base height returning to ~0.72m. |
| `reward_post_kick_stable_stand` | 0.4 | 1.0 | Composite: upright + height + low velocity + feet spacing. |
| `penalty_post_kick_crouch` | -2.4 | -3.0 | Penalizes base staying too low + knee over-flexion. |
| `reward_post_kick_recontact` | 2.4 | 2.4 | Rewards kicking foot returning to ground. |
| `reward_post_kick_velocity_damping` | 1.2 | 1.8 | Rewards low linear/angular velocity after kick. |

Tuning tips:
- The most impactful lever for standing stability is `penalty_post_kick_crouch` -- increase its magnitude (e.g. -4.0 to -5.0 in Stage C) to force the policy to stand up.
- `reward_post_kick_stable_stand` is a composite metric. Increase `h_stand` param (default 0.90) if the robot consistently crouches too low.
- If the kicking leg stays extended too long, increase `reward_kick_leg_retract` weight and decrease its `x_max` param.
- Curriculum promotion thresholds (`force_stage_b_step`, `force_stage_c_step` in `curriculums.py`) control when harder recovery constraints kick in. Consider lowering `force_stage_b_step` (default 250k) to introduce recovery rewards earlier.

**AMP reward balance (in `rsl_rl_amp_cfg.py`)**

| Config | Stage A | Stage B | Stage C |
|--------|---------|---------|---------|
| `amp_reward_coef` | 0.5 | 0.45 | 0.35 |
| `amp_task_reward_lerp` | 0.85 | 0.85 | 0.75 |

- `amp_reward_coef`: Higher = stronger style prior. Decrease in later stages to let task rewards dominate.
- `amp_task_reward_lerp`: Interpolation between AMP and task reward. Lower = more AMP influence.
- Can be overridden at runtime via CLI: `--style_reward_weight 0.3` (sets `amp_reward_coef`) and `--task_reward_weight 0.9` (sets `amp_task_reward_lerp`).

## Project Structure

```
G1_kicking/
├── setup.sh                        # One-click setup
├── scripts/rsl_rl/                 # Training & evaluation scripts
│   ├── base/train.py               # Main training entry
│   ├── base/play.py                # Evaluation / video recording
│   └── base/run_kick_stageB.sh     # Stage B helper
├── source/
│   ├── kick_task/                  # Kick skill task (gym envs, MDP, terrain)
│   ├── beyondAMP/                  # AMP framework (from beyondAMP)
│   └── rsl_rl_amp/                 # AMP PPO algorithm (from beyondAMP)
└── data/
    ├── assets/                     # Robot & ball URDF/USD
    └── datasets/g1_kick_skill/     # AMP motion capture data (.npz)
```

## Source Attribution

| Component | Origin |
|-----------|--------|
| `source/beyondAMP/` | [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP) |
| `source/rsl_rl_amp/` | [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP) |
| `source/kick_task/kick_task/robot/` | [robotlib](https://github.com/Renforce-Dynamics/robotlib) (G1 config inlined) |
| `source/kick_task/kick_task/terrain/` | soccerLab terrain module |
| `data/assets/unitree/` | [assetslib](https://github.com/Renforce-Dynamics/assetslib) |
| `data/datasets/` | AMP motion data from beyondAMP |
| Training scripts | Adapted from soccerLab |

## Acknowledgement

- [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP) - AMP integration for IsaacLab
- [robotlib](https://github.com/Renforce-Dynamics/robotlib) - Robot configurations
- [assetslib](https://github.com/Renforce-Dynamics/assetslib) - Robot assets (URDF/USD)
- [AMP for Hardware](https://github.com/escontra/AMP_for_hardware) - AMP implementation reference
