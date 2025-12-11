#!/bin/bash
# CosyVoice 环境激活脚本（自动设置环境变量）

VENV_PATH="/vepfs-dev/shawn/venv/cosyvoice"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ 虚拟环境不存在: $VENV_PATH"
    echo "请先运行: bash gen_video/tools/create_cosyvoice_env_fixed.sh"
    exit 1
fi

# 激活虚拟环境
source "$VENV_PATH/bin/activate"

# 设置环境变量（使用 /vepfs-dev 上的空间）
export PIP_CACHE_DIR="/vepfs-dev/shawn/.pip_cache"
export TMPDIR="/vepfs-dev/shawn/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

# 创建目录（如果不存在）
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$TMPDIR"

echo "✓ CosyVoice 环境已激活"
echo "✓ 环境变量已设置:"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "  TMPDIR=$TMPDIR"
echo ""
echo "现在可以运行 pip install 命令了"

