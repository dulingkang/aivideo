#!/bin/bash
# 阶段1模型下载脚本
# 使用外部配置的代理（通过 HTTP_PROXY/HTTPS_PROXY 环境变量）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "阶段1模型下载脚本"
echo "============================================================"

# 检查代理配置
if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "检测到代理配置:"
    [ -n "$HTTP_PROXY" ] && echo "  HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && echo "  HTTPS_PROXY=$HTTPS_PROXY"
    echo ""
else
    echo "提示: 未检测到代理配置（HTTP_PROXY/HTTPS_PROXY）"
    echo "      如需使用代理，请在执行前设置环境变量，例如："
    echo "      export HTTP_PROXY=http://proxy.example.com:8080"
    echo "      export HTTPS_PROXY=http://proxy.example.com:8080"
    echo ""
fi

# 设置缓存目录到挂载盘（避免占用根目录空间）
export HF_HOME=/vepfs-dev/shawn/.cache/huggingface
export TRANSFORMERS_CACHE=/vepfs-dev/shawn/.cache/huggingface
export HF_DATASETS_CACHE=/vepfs-dev/shawn/.cache/huggingface/datasets

# 确保缓存目录存在
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

echo "✓ HuggingFace 缓存目录: $HF_HOME"
echo ""

# 激活 Python 环境（如果需要）
if [ -f "/vepfs-dev/shawn/venv/py312/bin/activate" ]; then
    source /vepfs-dev/shawn/venv/py312/bin/activate
    echo "✓ 已激活 Python 环境"
fi

# 运行 Python 下载脚本（使用 -u 参数禁用输出缓冲，实时显示日志）
echo "开始下载模型..."
echo ""

python3 -u download_stage1_models.py 2>&1 | tee download_log.txt

echo ""
echo "============================================================"
echo "下载完成！"
echo "============================================================"

