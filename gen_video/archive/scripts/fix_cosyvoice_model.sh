#!/bin/bash
# 修复 CosyVoice 模型下载问题

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 设置缓存目录
export PIP_CACHE_DIR="/vepfs-dev/shawn/.cache/pip"
export HF_HOME="/vepfs-dev/shawn/.cache/huggingface"
export TRANSFORMERS_CACHE="/vepfs-dev/shawn/.cache/huggingface"
export HF_DATASETS_CACHE="/vepfs-dev/shawn/.cache/huggingface"

COSYVOICE_MODEL_DIR="/vepfs-dev/shawn/vid/fanren/CosyVoice/pretrained_models/CosyVoice2-0.5B"

echo "============================================================"
echo "修复 CosyVoice 模型"
echo "============================================================"
echo ""

# 检查模型文件大小
echo "检查模型文件..."
if [ -d "$COSYVOICE_MODEL_DIR" ]; then
    echo "模型目录: $COSYVOICE_MODEL_DIR"
    echo "文件大小:"
    du -sh "$COSYVOICE_MODEL_DIR"/*.pt 2>/dev/null | head -5 || echo "  未找到 .pt 文件"
    
    # 检查是否有损坏的文件（小于 1MB）
    small_files=$(find "$COSYVOICE_MODEL_DIR" -name "*.pt" -size -1M 2>/dev/null | wc -l)
    if [ "$small_files" -gt 0 ]; then
        echo ""
        echo "⚠ 发现损坏的模型文件（小于 1MB），需要重新下载"
        echo "删除损坏的模型文件..."
        rm -f "$COSYVOICE_MODEL_DIR"/*.pt
        rm -f "$COSYVOICE_MODEL_DIR"/*.safetensors 2>/dev/null || true
        echo "✓ 已删除损坏的文件"
    fi
fi

echo ""
echo "重新下载 CosyVoice 模型..."
echo "使用 ModelScope 下载（可能需要较长时间）..."
python download_stage1_models.py --models cosyvoice

echo ""
echo "============================================================"
echo "下载完成，请重新运行测试脚本"
echo "============================================================"

