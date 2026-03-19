#!/bin/bash
# One-click setup for G1_kicking.
# Prerequisite: IsaacLab must be installed (provides isaaclab, isaaclab_rl, isaaclab_tasks, rsl_rl).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing local editable packages..."

pip install -e ./source/beyondAMP
pip install -e ./source/rsl_rl_amp
pip install -e ./source/kick_task

echo ""
echo "Setup complete. You can now train with:"
echo "  python scripts/rsl_rl/base/train.py --task g1-kick-skill-amp --amp --headless"
