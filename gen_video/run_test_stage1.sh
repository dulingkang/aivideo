#!/bin/bash
# 阶段1功能测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 设置缓存目录（避免占用根目录空间）
export PIP_CACHE_DIR="/vepfs-dev/shawn/.cache/pip"
export HF_HOME="/vepfs-dev/shawn/.cache/huggingface"
export TRANSFORMERS_CACHE="/vepfs-dev/shawn/.cache/huggingface"
export HF_DATASETS_CACHE="/vepfs-dev/shawn/.cache/huggingface"

# 运行测试脚本
echo "============================================================"
echo "运行阶段1功能测试"
echo "============================================================"
echo ""
python test_stage1.py

