# G1 Kick Skill AMP

Pure kick skill task（非 locomotion）：上层摆球，下层 0~2 步微调 + 起脚 + 大力踢出 + 不追球。极简 reward，强 AMP + tracking 辅助动作成型，curriculum 驱动 Stage A → B → C。

## 任务注册 ID（与 play 一致）

**训练 / play 时 `--task` 必须与下表 ID 完全一致：**

| 任务 ID | 说明 |
|---------|------|
| **`g1-kick-skill-amp`** | **主入口**：curriculum 内自动 Stage A → B → C（训练加 `--amp`） |
| `g1-kick-skill` | 纯 PPO，无 AMP（训练/play 都不加 `--amp`） |
| `g1-kick-skill-stageB-amp` | 可选：固定 Stage B 微调（加 `--amp`） |
| `g1-kick-skill-stageC-amp` | 可选：固定 Stage C 微调（加 `--amp`） |

Stage B/C 的 task 名就是带 `-stageB-amp` / `-stageC-amp` 后缀，不要写成别的。

## 推荐训练命令

**必须加 `--amp`**。

```bash
cd /root/soccerLab

python scripts/rsl_rl/base/train.py --task g1-kick-skill-amp --amp --headless
```

可选：`--num_envs 4096`、`--max_iterations 120000`、`--seed 42`。  
日志与 checkpoint：`logs/rsl_rl/g1_kick_skill/<timestamp>_<run_name>/`（见下「日志路径与 PPO/AMP 区分」）。

## 日志路径与 PPO/AMP 区分

**不是路径命名错了**：PPO 和 AMP 的 `experiment_name` 都是 `g1_kick_skill`，所以日志都在 `logs/rsl_rl/g1_kick_skill/` 下。  
区分靠 **run_name**（来自各自 runner 的默认值或 `--run_name`）：

| 训练方式 | 默认 run_name | 典型目录名 |
|----------|----------------|------------|
| 主任务 AMP（g1-kick-skill-amp） | `stageA` | `<timestamp>_stageA` |
| 纯 PPO（g1-kick-skill） | `ppo` | `<timestamp>_ppo` |
| 固定 Stage B AMP | `stageB` | `<timestamp>_stageB` |
| 固定 Stage C AMP | `stageC` | `<timestamp>_stageC` |

所以 **`*_stageA` 的 run 按默认配置就是 AMP 训练**（主任务 curriculum），不是纯 PPO。  
若不确定，看该 run 目录下 `params/agent.yaml`：`algorithm.class_name` 为 `AMPPPO`、或存在 `amp_data` / `runner_type: ... AMPOnPolicyRunner` 即为 AMP。

## Play / 推理用哪个 task

**`--task` 必须和当时训练用的任务 ID 一致：**

- **主任务 AMP（curriculum）** → 目录多为 `*_stageA` → play 用 `--task g1-kick-skill-amp --amp`，例如：
  ```bash
  python scripts/rsl_rl/base/play.py --task g1-kick-skill-amp --amp --checkpoint /path/to/model_XXXX.pt --num_envs 64 --headless
  ```
- **纯 PPO** → 目录多为 `*_ppo` → play 用 `--task g1-kick-skill`（不加 `--amp`），例如：
  ```bash
  python scripts/rsl_rl/base/play.py --task g1-kick-skill --checkpoint /path/to/model_XXXX.pt --num_envs 64 --headless
  ```
- **固定 Stage B 微调** → `--task g1-kick-skill-stageB-amp --amp`
- **固定 Stage C 微调** → `--task g1-kick-skill-stageC-amp --amp`

例如 `2026-03-17_00-47-29_stageA`：其 `params/agent.yaml` 里为 `AMPPPO`，**是 AMP 训练**，play 应用 `--task g1-kick-skill-amp --amp`。

## 当前推荐训练流程（针对「已会大力踢、post-kick 一字马 exploit」）

**固定 Stage 的两种方式（二选一，无需环境变量）：**

1. **用 `--task` 直接选阶段**  
   - Stage B：`--task g1-kick-skill-stageB-amp`  
   - Stage C：`--task g1-kick-skill-stageC-amp`  
   脚本会根据 task 自动固定对应阶段。
2. **用主任务 + `--kick_stage`**  
   - `--task g1-kick-skill-amp --kick_stage B` 或 `--kick_stage C`  
   与上面等价，便于和 resume 等参数一起用。

当前阶段主要问题是 **踢后劈叉/一字马/关节发散**，不是「不会踢」。因此推荐：

1. **先从 Stage A checkpoint 强制跑 Stage B**  
   用 `--task g1-kick-skill-stageB-amp`（或 `--task g1-kick-skill-amp --kick_stage B`），从已有 Stage A 的 .pt 接着训（resume）。一上来就上恢复项，重点在收腿/回正/一字马惩罚。
2. **若 Stage B 有改善但仍有明显劈叉**  
   再改用 `--task g1-kick-skill-stageC-amp`（或 `--kick_stage C`）继续训。
3. **自动 curriculum 作为后续长期策略**  
   等 B/C 稳定后，用 `--task g1-kick-skill-amp` 且**不**加 `--kick_stage`，让自动 A→B→C 做长训。

示例（从 Stage A 的 checkpoint 强制 B 开始）：
```bash
cd /root/soccerLab
python scripts/rsl_rl/base/train.py --task g1-kick-skill-stageB-amp --amp --headless \
  --resume True --load_run 2026-03-17_00-50-21_stageA --checkpoint model_6100.pt
```
或主任务 + 参数：
```bash
python scripts/rsl_rl/base/train.py --task g1-kick-skill-amp --amp --headless --kick_stage B \
  --resume True --load_run 2026-03-17_00-50-21_stageA --checkpoint model_6100.pt
```
（`--load_run` / `--checkpoint` 换成你的 run 与 .pt。）

环境变量 `KICK_SKILL_FORCE_STAGE=A|B|C` 仍可用，但推荐用 `--task` 或 `--kick_stage` 避免手动设环境变量。

## Curriculum 自动切换与 Play 时固定 Stage

**自动切换（仅训练时，且未设 KICK_SKILL_FORCE_STAGE 时）**：在 `mdp/curriculums.py` 的 `kick_skill_curriculum` 里，每个 episode 边界根据「刚结束的一批 env 的 episode 均值」和「当前全局 step」决定升/降阶段：

- **A→B**：仍以「会踢」为主：`touch_avg >= 2.4` 且 `power_avg >= 1.8`，**或** `step >= 250_000`。
- **B→C**：**踢球质量够好** + **recover 已成型**：  
  - 踢球：`touch_avg >= 2.8` 且 `dir_avg >= 1.4` 且 `goal_speed_avg >= 1.3`  
  - 恢复：`reward_post_kick_upright`、`reward_kick_leg_retract` 的 episode 均值达到阈值，或综合 `recovery_agg` 达标；**或** `step >= 700_000`。
- **最短驻留**：进入 B 后至少 `min_stage_b_steps`（默认 100k）步内不允许 B→A；进入 C 后至少 `min_stage_c_steps`（默认 150k）步内不允许 C→B，避免短时波动导致立刻降回。
- **B→A（降回）**：仅在驻留时间已满且 `touch_avg`/`power_avg` **严重**塌掉时（阈值更保守：`demote_b_to_a_touch=1.0`, `demote_b_to_a_power=0.6`）才退回 A。
- **C→B（降回）**：仅在驻留时间已满且触球/球速/方向/目标球速明显掉（见 `demote_c_to_b_*`）时才退回 B。

Stage A 未动：post-kick 恢复项在 A 里权重全是 0，只在 B/C 才打开。

**Play 时固定 Stage**：用 `--task g1-kick-skill-stageB-amp` 或 `g1-kick-skill-stageC-amp`，或主任务加 `--kick_stage B`/`C`，即可让 post-kick 项生效。例如：
```bash
python scripts/rsl_rl/base/play.py --task g1-kick-skill-stageB-amp --amp --checkpoint /path/to/model_XXXX.pt --num_envs 64 --headless
# 或
python scripts/rsl_rl/base/play.py --task g1-kick-skill-amp --amp --kick_stage B --checkpoint /path/to/model_XXXX.pt --num_envs 64 --headless
```
不设则按 curriculum 逻辑（默认从 A 起）。

## Stage A / B / C reward scale（最终表）

| Term | Stage A | Stage B | Stage C |
|------|---------|---------|---------|
| reward_approach_ball | 0.5 | 0.35 | 0.30 |
| reward_kick_leg_swing | 5.0 | 4.0 | 3.5 |
| reward_kick_foot_contact_ball | 10.0 | 8.5 | 7.5 |
| reward_first_clean_contact_bonus | 4.0 | 3.0 | 2.5 |
| reward_ball_speed | **12.0** | 7.5 | 7.0 |
| reward_ball_goal_direction | 0.0 | 3.0 | 5.0 |
| reward_ball_goal_speed | 0.0 | 3.5 | 5.5 |
| tracking_anchor_pos | 0.5 | 0.45 | 0.55 |
| tracking_anchor_ori | 0.4 | 0.42 | 0.52 |
| tracking_body_pos | 0.6 | 0.58 | 0.68 |
| tracking_body_ori | 0.5 | 0.50 | 0.58 |
| tracking_body_vel | 0.4 | 0.45 | 0.52 |
| penalty_excess_travel | -0.12 | -0.30 | -0.60 |
| penalty_bad_ball_contact | -2.8 | -2.8 | -3.2 |
| action_rate | -4e-4 | -6e-4 | -8e-4 |
| joint_limit | -0.08 | -0.10 | -0.13 |

新增/加强的 post-kick 项（A 中权重保持 0，仅在 B/C 打开）：

| Term | Stage A | Stage B | Stage C |
|------|---------|---------|---------|
| reward_post_kick_upright | 0.0 | 2.8 | 3.6 |
| reward_post_kick_base_height | 0.0 | 1.8 | 2.4 |
| reward_kick_leg_retract | 0.0 | 2.8 | 3.2 |
| reward_post_kick_recontact | 0.0 | 2.0 | 2.4 |
| reward_post_kick_velocity_damping | 0.0 | 1.0 | 1.8 |
| reward_post_kick_joint_nominal | 0.0 | 1.2 | 1.8 |
| penalty_leg_spread | 0.0 | -2.6 | -3.8 |
| penalty_post_joint_limit_stronger | 0.0 | -0.18 | -0.26 |
| penalty_post_kick_crouch | 0.0 | -2.4 | -3.0 |
| reward_post_kick_stand_height | 0.0 | 2.4 | 2.8 |
| reward_post_kick_stable_stand | 0.0 | 2.6 | 3.2 |

Stage A：触球 + 球速 + 挥腿为主；tracking 辅助；方向/幅度几乎不约束。

## Tracking body 集合（最终）

仅关键链，无支撑腿/上肢主跟：

- `pelvis`
- `torso_link`
- `right_hip_roll_link`, `right_hip_pitch_link`, `right_knee_link`, `right_ankle_roll_link`（踢击腿）

## Gate 实现位置

均在 **`mdp/rewards.py`** 内：

- **g_dist**：`kick_distance_gate` — 脚-球水平距离，近则放大
- **g_phase**：`kick_phase_gate` — 进入 strike window 增强
- **g_leg**：`kick_leg_velocity_gate` — 踢击腿速度沿目标方向（A 偏 ball，B 偏 goal）
- **g_contact**：`kick_contact_gate` — 有效触球后短窗（hold_steps=6）内放大 ball_speed 等

`reward_kick_leg_swing` 使用 `g_dist * g_phase * g_leg`；`reward_ball_speed` 使用 `g_contact * speed_score`。  
metric 项（metric_gate_dist/phase/leg/contact）weight=0，仅写 TensorBoard。

## first_clean_contact bonus 实现方式

- **函数**：`mdp/rewards.reward_first_clean_contact_bonus`
- **逻辑**：在 `_update_kick_state` 后，若 `contact_step == episode_length_buf` 且 `contact_step >= 0`，则判定为本步发生「首次有效触球」，返回 1，否则 0。
- **效果**：每 episode 仅在发生触球的那一帧给一次 bonus，不依赖 latched 持续给；强化「真的踢到」而非蹭到后一直算 contact。

## 已禁用 / 极小化的 reward

本任务**未使用**且不加入：

- track_lin_vel_xy / track_ang_vel_yaw  
- 大权重 upright / base height shaping  
- stand_still / gait / symmetry / foot_air_time  
- 长时平稳走路、鼓励「站稳别动」的项  

仅保留：action_rate、joint_limit 极小权重，防数值爆炸，不压制爆发踢击。

## 保留意见 / 后续迭代（已落实部分）

| 项 | 状态 | 说明 |
|----|------|------|
| **spread / retract 坐标系** | 已改 | 已改为 **robot (root) frame**：`quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)` 取 body 的 y 分量作 lateral，朝向随机化下更稳。 |
| **post_kick_joint_nominal 关节集** | 已扩 | 在原有 4 个基础上增加 `right_hip_pitch_joint`、`left_hip_pitch_joint`、`left_knee_joint`；未加 ankle，按现象可再补。 |
| **Stage C 踢球强度** | 已抬 | Stage C 现为 `ball_speed=6.0`、`kick_foot_contact_ball=7.0`，略高于此前 5.5/6.5，减轻「稳但不够炸」。 |
| **post_kick_recontact** | 仍 proxy | 仍用脚高 `foot_z <= max_foot_height`。若出现低脚悬空 exploit 再改为真实 contact/force 判定。 |
| **upright 量测** | 未改 | 仍用 `projected_gravity_b` 的 XY 范数；需更鲁棒时可再细化。 |

## 数据与路径

- **AMP 数据**：`source/third_party/beyondAMP/data/datasets/g1_kick_skill/motion_files_kick.txt`（仅 kick 类 motion）
- **单条 motion**：`KICK_SKILL_MOTION_FILE` 或默认 `cf_shoot_10_01.npz`（见 `kick_skill_env_cfg._resolve_kick_motion_default`）

## 主要文件

| 文件 | 作用 |
|------|------|
| `kick_skill_env_cfg.py` | 极简 reward、窄球 reset、tracking body 最小集、curriculum 挂载 |
| `kick_skill_amp_env_cfg.py` | AMP 观测组 |
| `kick_amp_data_cfg.py` | kick-only AMP 数据 |
| `mdp/commands.py` | Ball 目标速度 + KickMotionCommand |
| `mdp/rewards.py` | Task reward + first_clean_contact_bonus + tracking（gated）+ gate |
| `mdp/curriculums.py` | A/B/C 权重表 |
| `mdp/observations.py` | 球/目标/phase/anchor 观测 |
| `mdp/events.py`, `mdp/terminations.py` | 近距 reset、早停/坏接触终止 |
| `rsl_rl_amp_cfg.py` | 主 AMP runner（g1-kick-skill-amp）；StageB/C 为可选 runner |
