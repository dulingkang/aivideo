#!/bin/bash
# CosyVoice 独立环境创建脚本

set -e

echo "=========================================="
echo "创建 CosyVoice 独立环境"
echo "=========================================="
echo ""

# 配置
PROJECT_ROOT="/vepfs-dev/shawn/vid/fanren"
VENV_NAME="venv_cosyvoice"
PYTHON_VERSION="3.10"  # 推荐使用 Python 3.10（CosyVoice 兼容性更好）

# 检查 Python 版本
if ! command -v python${PYTHON_VERSION} &> /dev/null; then
    echo "⚠️  警告: Python ${PYTHON_VERSION} 未找到，尝试使用 python3"
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python${PYTHON_VERSION}"
fi

cd "$PROJECT_ROOT"

# 1. 创建虚拟环境
echo "[1/5] 创建虚拟环境..."
if [ -d "$VENV_NAME" ]; then
    echo "  ⚠️  虚拟环境已存在: $VENV_NAME"
    read -p "  是否删除并重新创建？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        $PYTHON_CMD -m venv "$VENV_NAME"
        echo "  ✓ 虚拟环境已重新创建"
    else
        echo "  ℹ️  使用现有虚拟环境"
    fi
else
    $PYTHON_CMD -m venv "$VENV_NAME"
    echo "  ✓ 虚拟环境已创建"
fi

# 2. 激活虚拟环境
echo ""
echo "[2/5] 激活虚拟环境..."
source "$VENV_NAME/bin/activate"
echo "  ✓ 虚拟环境已激活"

# 3. 升级 pip
echo ""
echo "[3/5] 升级 pip..."
pip install --upgrade pip
echo "  ✓ pip 已升级"

# 4. 安装 CosyVoice 依赖
echo ""
echo "[4/5] 安装 CosyVoice 依赖..."
cd "$PROJECT_ROOT/CosyVoice"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "  ✓ CosyVoice 依赖已安装"
else
    echo "  ⚠️  警告: requirements.txt 不存在，跳过依赖安装"
fi

# 5. 安装特定版本的 transformers
echo ""
echo "[5/5] 安装 transformers 4.51.3..."
pip install transformers==4.51.3 --force-reinstall
echo "  ✓ transformers 4.51.3 已安装"

# 验证安装
echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="
python -c "import transformers; print(f'transformers 版本: {transformers.__version__}')"
python -c "from transformers import modeling_layers; print('❌ transformers.modeling_layers 存在（不应该存在）')" 2>/dev/null || echo "✓ transformers.modeling_layers 不存在（正确）"

echo ""
echo "=========================================="
echo "环境创建完成！"
echo "=========================================="
echo ""
echo "环境路径: $PROJECT_ROOT/$VENV_NAME"
echo "Python 路径: $PROJECT_ROOT/$VENV_NAME/bin/python"
echo ""
echo "使用方法:"
echo "  1. 在 config.yaml 中设置:"
echo "     tts:"
echo "       cosyvoice:"
echo "         use_subprocess: true"
echo "         subprocess_python: $PROJECT_ROOT/$VENV_NAME/bin/python"
echo ""
echo "  2. 测试子进程调用:"
echo "     $PROJECT_ROOT/$VENV_NAME/bin/python gen_video/tools/cosyvoice_subprocess_wrapper.py \\"
echo "       --text '测试文本' \\"
echo "       --output test.wav \\"
echo "       --prompt-speech gen_video/assets/prompts/haoran_prompt_5s.wav \\"
echo "       --prompt-text '欢迎来到本期知识探索' \\"
echo "       --mode zero_shot"
echo ""

