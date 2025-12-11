#!/bin/bash
# HunyuanVideo 1.5模型下载脚本（推荐版本）

set -e

echo "=========================================="
echo "HunyuanVideo 1.5模型下载脚本（推荐版本）"
echo "=========================================="

# 配置
# 推荐使用diffusers格式模型（完整pipeline，包含所有组件）
MODEL_NAME="zai-org/CogVideoX-5b"  # 720p图生视频（推荐）
# 其他可选版本：
# - hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v（480p图生视频，速度快）
# - hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled（720p蒸馏版，更快）
# - hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v（720p文生视频）
LOCAL_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/CogVideoX-5b"
VENV_PATH="/vepfs-dev/shawn/venv/py312/bin/activate"

# 创建目录
echo "1. 创建模型目录..."
mkdir -p "$LOCAL_DIR"
echo "   ✓ 目录已创建: $LOCAL_DIR"

# 激活虚拟环境
echo "2. 激活虚拟环境..."
source "$VENV_PATH"
echo "   ✓ 虚拟环境已激活"

# 检查huggingface-cli是否安装
echo "3. 检查huggingface-cli..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "   ⚠ huggingface-cli未安装，正在安装..."
    pip install -U huggingface_hub[cli]
    echo "   ✓ huggingface-cli已安装"
else
    echo "   ✓ huggingface-cli已安装"
fi

# 检查是否已登录HuggingFace
echo "4. 检查HuggingFace登录状态..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "   ⚠ 未登录HuggingFace，请先登录..."
    echo "   提示: 运行 huggingface-cli login"
    echo "   或者设置环境变量: export HF_TOKEN=your_token"
    
    # 尝试从环境变量读取token
    if [ -n "$HF_TOKEN" ]; then
        echo "   ℹ 检测到HF_TOKEN环境变量，使用它登录..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    else
        echo "   ❌ 请先登录HuggingFace或设置HF_TOKEN环境变量"
        exit 1
    fi
else
    echo "   ✓ 已登录HuggingFace"
fi

# 下载模型
echo "5. 开始下载HunyuanVideo 1.5模型..."
echo "   模型: $MODEL_NAME"
echo "   目标目录: $LOCAL_DIR"
echo "   类型: 720p图生视频（Image-to-Video）"
echo "   格式: diffusers标准格式（完整pipeline）"
echo "   预计大小: 15-25GB"
echo "   这可能需要一些时间，请耐心等待..."

huggingface-cli download "$MODEL_NAME" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "=========================================="
echo "✅ HunyuanVideo 1.5模型下载完成！"
echo "=========================================="
echo "模型位置: $LOCAL_DIR"
echo ""
echo "下一步:"
echo "1. 更新config.yaml中的model_path为本地路径"
echo "2. 确保use_v15: true（使用1.5版本）"
echo "3. 运行测试脚本验证: python gen_video/test_hunyuanvideo.py"

