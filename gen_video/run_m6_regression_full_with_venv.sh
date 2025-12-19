#!/bin/bash
set -euo pipefail

# 一键运行 M6 全套回归（自动激活 venv）
# 默认使用：/vepfs-dev/shawn/venv/py312
#
# 用法：
#   ./run_m6_regression_full_with_venv.sh
#   ./run_m6_regression_full_with_venv.sh --include-battle-occlusion

VENV_PATH="${VENV_PATH:-/vepfs-dev/shawn/venv/py312}"
GEN_VIDEO_DIR="/vepfs-dev/shawn/vid/fanren/gen_video"

if [ ! -f "$VENV_PATH/bin/activate" ]; then
  echo "❌ venv 不存在: $VENV_PATH"
  exit 1
fi

echo "📦 激活虚拟环境: $VENV_PATH"
source "$VENV_PATH/bin/activate"

cd "$GEN_VIDEO_DIR"
python3 tools/run_m6_regression_full.py "$@"


