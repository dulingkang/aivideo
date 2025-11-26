#!/bin/bash
# 安装缺失的 Python 模块

set -e

echo "============================================================"
echo "安装缺失的 Python 模块"
echo "============================================================"

# 检查虚拟环境
VENV_PATH="../.venv"
if [ -d "$VENV_PATH" ]; then
    echo "激活虚拟环境: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "⚠ 未找到虚拟环境，使用系统 Python"
fi

# 升级 pip
echo ""
echo "升级 pip..."
python -m pip install --upgrade pip

# 安装基础依赖
echo ""
echo "安装基础依赖..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch torchvision torchaudio

pip install transformers>=4.30.0
pip install diffusers>=0.21.0
pip install accelerate>=0.20.0
pip install safetensors>=0.3.0
pip install pillow>=9.5.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install pyyaml>=6.0

# 检查 InstantID（通过路径导入，不需要 pip 安装）
echo ""
echo "检查 InstantID..."
if [ -d "../InstantID" ]; then
    echo "  ✓ InstantID 目录存在，将通过路径导入"
else
    echo "  ⚠ InstantID 目录不存在，请确保 ../InstantID 目录存在"
    echo "    或从 GitHub 克隆: git clone https://github.com/instantX-research/InstantID.git ../InstantID"
fi

# 安装 InsightFace（InstantID 需要）
echo ""
echo "安装 InsightFace..."
pip install insightface>=0.7.3
pip install onnxruntime onnxruntime-gpu 2>/dev/null || pip install onnxruntime

# 检查 CosyVoice（通过路径导入，不需要 pip 安装）
echo ""
echo "检查 CosyVoice..."
if [ -d "../CosyVoice" ]; then
    echo "  ✓ CosyVoice 目录存在，将通过路径导入"
    # 检查是否需要安装 CosyVoice 的依赖
    if [ -f "../CosyVoice/requirements.txt" ]; then
        echo "  安装 CosyVoice 依赖..."
        pip install -r ../CosyVoice/requirements.txt
    fi
else
    echo "  ⚠ CosyVoice 仓库不存在，请先克隆："
    echo "    cd .. && git clone https://github.com/FunAudioLLM/CosyVoice.git"
fi

# 安装 ModelScope（用于下载 CosyVoice 模型）
echo ""
echo "安装 ModelScope..."
pip install modelscope>=1.9.0

# 安装 WhisperX 相关
echo ""
echo "安装 WhisperX 相关..."
pip install whisperx>=3.1.1
pip install faster-whisper>=0.10.0
pip install pyannote.audio>=3.1.0

# 安装其他依赖
echo ""
echo "安装其他依赖..."
pip install ffmpeg-python>=0.2.0
pip install imageio>=2.31.0
pip install imageio-ffmpeg>=0.4.9
pip install librosa>=0.10.0
pip install soundfile>=0.12.0
pip install huggingface-hub>=0.19.0

echo ""
echo "============================================================"
echo "✓ 模块安装完成！"
echo "============================================================"
echo ""
echo "下一步："
echo "  1. 运行 python3 check_models_and_modules.py 检查是否还有缺失"
echo "  2. 运行 python3 download_stage1_models.py 下载模型（如果需要）"

