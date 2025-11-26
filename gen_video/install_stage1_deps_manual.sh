#!/bin/bash
# 阶段1依赖手动安装脚本（如果自动安装失败）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "阶段1依赖手动安装脚本"
echo "============================================================"

# 激活 Python 环境
if [ -f "/vepfs-dev/shawn/venv/py312/bin/activate" ]; then
    source /vepfs-dev/shawn/venv/py312/bin/activate
    echo "✓ 已激活 Python 环境"
fi

echo ""
echo "手动安装 InstantID 依赖..."
echo ""

# InstantID 需要的依赖（根据 GitHub 文档）
echo "[1] 安装 InstantID 基础依赖..."
pip install opencv-python transformers accelerate insightface diffusers || {
    echo "⚠ 部分依赖安装失败，继续..."
}

echo ""
echo "[2] 从 GitHub 克隆并安装 InstantID..."
INSTANTID_DIR="/tmp/InstantID"
if [ -d "$INSTANTID_DIR" ]; then
    rm -rf "$INSTANTID_DIR"
fi

git clone https://github.com/instantX-research/InstantID.git "$INSTANTID_DIR" || {
    echo "✗ Git 克隆失败"
    exit 1
}

cd "$INSTANTID_DIR"
pip install -e . || {
    echo "⚠ 安装失败，尝试安装 requirements..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    if [ -f "gradio_demo/requirements.txt" ]; then
        pip install -r gradio_demo/requirements.txt
    fi
}

cd "$SCRIPT_DIR"
echo "✓ InstantID 安装完成"

echo ""
echo "[3] 安装 CosyVoice..."
pip install cosyvoice || {
    echo "⚠ CosyVoice 从 PyPI 安装失败，尝试从 GitHub..."
    pip install git+https://github.com/FunAudioLLM/CosyVoice.git || {
        echo "✗ CosyVoice 安装失败"
    }
}

echo ""
echo "[4] 更新 pyannote.audio..."
pip install --upgrade pyannote.audio || {
    echo "⚠ pyannote.audio 更新失败，但可能不影响使用"
}

echo ""
echo "============================================================"
echo "✓ 手动安装完成！"
echo "============================================================"
echo ""
echo "验证安装："
pip show instantid 2>/dev/null && echo "✓ instantid" || echo "✗ instantid (可能需要检查)"
pip show cosyvoice 2>/dev/null && echo "✓ cosyvoice" || echo "✗ cosyvoice"
pip show whisperx 2>/dev/null && echo "✓ whisperx" || echo "✗ whisperx"

