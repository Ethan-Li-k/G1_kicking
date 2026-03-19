# G1 Kicking: Humanoid Kick Skill with AMP

Unitree G1 humanoid kick skill training using Adversarial Motion Priors (AMP) on IsaacLab.

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
