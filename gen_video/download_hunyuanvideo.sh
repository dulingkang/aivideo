#!/bin/bash
# HunyuanVideo模型下载脚本

set -e

echo "=========================================="
echo "HunyuanVideo模型下载脚本"
echo "=========================================="

# 配置
# 注意：原版HunyuanVideo的正确名称是 hunyuanvideo-community/HunyuanVideo-I2V
# 但推荐使用1.5版本：hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v
MODEL_NAME="hunyuanvideo-community/HunyuanVideo-I2V"  # 原版
# 推荐使用1.5版本，请使用 download_hunyuanvideo_1.5.sh
LOCAL_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-video"
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
echo "5. 开始下载HunyuanVideo模型..."
echo "   模型: $MODEL_NAME"
echo "   目标目录: $LOCAL_DIR"
echo "   预计大小: 20-30GB"
echo "   这可能需要一些时间，请耐心等待..."

huggingface-cli download "$MODEL_NAME" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "=========================================="
echo "✅ HunyuanVideo模型下载完成！"
echo "=========================================="
echo "模型位置: $LOCAL_DIR"
echo ""
echo "下一步:"
echo "1. 更新config.yaml中的model_path为本地路径"
echo "2. 运行测试脚本验证: python gen_video/test_hunyuanvideo.py"

