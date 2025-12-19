#!/bin/bash
# 下载 CLIP 模型到 models 目录（使用镜像站加速）

# 设置 HuggingFace 缓存目录（临时，用于下载）
export HF_HOME=/vepfs-dev/shawn/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

# 设置镜像站（加速下载）
export HF_ENDPOINT=https://hf-mirror.com

echo "🌐 使用镜像站: $HF_ENDPOINT"
echo "📦 目标目录: gen_video/models/clip/openai-clip-vit-large-patch14"
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")/.."

# 运行下载脚本
python3 tools/download_clip_to_models.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 下载完成！"
    echo "💡 模型已保存到: models/clip/openai-clip-vit-large-patch14"
else
    echo ""
    echo "❌ 下载失败"
    echo "💡 可以尝试："
    echo "   1. 使用 proxychains4: proxychains4 python3 tools/download_clip_to_models.py"
    echo "   2. 或安装 ModelScope: pip install modelscope && python3 tools/download_clip_with_mirror.py"
    exit 1
fi

