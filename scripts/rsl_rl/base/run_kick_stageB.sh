#!/usr/bin/env bash
# 约束一字马：从 Stage A checkpoint 强制 Stage B 训练（post-kick 恢复项打开）
# 用法: ./run_kick_stageB.sh [LOAD_RUN] [CHECKPOINT]  例: ./run_kick_stageB.sh 2026-03-17_00-50-21_stageA model_6100.pt

set -e
cd "$(dirname "$0")/../../.."

LOAD_RUN="${1:-2026-03-17_00-50-21_stageA}"
CHECKPOINT="${2:-model_6100.pt}"

python scripts/rsl_rl/base/train.py \
  --task g1-kick-skill-stageB-amp \
  --amp \
  --headless \
  --resume True \
  --load_run "$LOAD_RUN" \
  --checkpoint "$CHECKPOINT" \
  --run_name stageB \
  "${@:3}"
