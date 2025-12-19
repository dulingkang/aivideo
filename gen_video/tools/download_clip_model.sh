#!/bin/bash
# 通过 proxychains4 下载 CLIP 模型

# 设置 HuggingFace 缓存目录
export HF_HOME=/vepfs-dev/shawn/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

# 确保缓存目录存在
mkdir -p "$HF_HOME"

echo "📦 HuggingFace 缓存目录: $HF_HOME"
echo "📥 开始通过 proxychains4 下载 CLIP 模型..."
echo ""

# 切换到脚本所在目录
cd "$(dirname "$0")/.."

# 通过 proxychains4 运行 Python 脚本
proxychains4 python3 tools/download_clip_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 下载完成！"
else
    echo ""
    echo "❌ 下载失败，请检查网络连接和代理设置"
    exit 1
fi

