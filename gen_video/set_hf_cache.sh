#!/bin/bash
# 设置 HuggingFace 缓存目录到 /vepfs-dev（避免根目录空间不足）
# 注意: 缓存目录已经在挂载盘上，此脚本用于确保环境变量正确设置

# 设置 HuggingFace 缓存目录环境变量（使用挂载盘）
export HF_HOME=/vepfs-dev/shawn/.cache/huggingface
export TRANSFORMERS_CACHE=/vepfs-dev/shawn/.cache/huggingface
export HF_DATASETS_CACHE=/vepfs-dev/shawn/.cache/huggingface/datasets

# 设置 pip 缓存到挂载盘（避免占用根目录空间）
export PIP_CACHE_DIR=/vepfs-dev/shawn/.cache/pip

# 确保目录存在
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$PIP_CACHE_DIR"

echo "✓ 缓存目录已设置为挂载盘:"
echo "  HuggingFace: $HF_HOME"
echo "  Pip: $PIP_CACHE_DIR"
echo ""
echo "使用此脚本运行 Python 脚本，或手动设置环境变量："
echo "  export HF_HOME=/vepfs-dev/shawn/.cache/huggingface"
echo "  export TRANSFORMERS_CACHE=/vepfs-dev/shawn/.cache/huggingface"
echo "  export PIP_CACHE_DIR=/vepfs-dev/shawn/.cache/pip"

