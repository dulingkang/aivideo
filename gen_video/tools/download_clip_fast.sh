#!/bin/bash
# 快速下载 CLIP 模型（使用多种镜像源）

# 设置 HuggingFace 缓存目录
export HF_HOME=/vepfs-dev/shawn/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

# 确保缓存目录存在
mkdir -p "$HF_HOME"

echo "📦 HuggingFace 缓存目录: $HF_HOME"
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")/.."

# 方法 1: 尝试使用 ModelScope（如果已安装）
if python3 -c "import modelscope" 2>/dev/null; then
    echo "✅ 检测到 ModelScope，使用 ModelScope 下载（推荐，速度快）..."
    python3 tools/download_clip_with_mirror.py
    exit $?
fi

# 方法 2: 使用 HuggingFace 镜像站
echo "🔄 使用 HuggingFace 镜像站下载..."
export HF_ENDPOINT=https://hf-mirror.com
python3 tools/download_clip_with_mirror.py
if [ $? -eq 0 ]; then
    exit 0
fi

# 方法 3: 使用 proxychains4（如果前两种方法失败）
echo "🔄 使用 proxychains4 下载..."
proxychains4 python3 tools/download_clip_model.py

