#!/bin/bash
# 快速启动MVP API服务

set -e

echo "=========================================="
echo "🚀 启动AI视频生成平台MVP"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到python3，请先安装Python"
    exit 1
fi

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  警告: 未检测到虚拟环境"
    echo "   建议先激活虚拟环境:"
    echo "   source /vepfs-dev/shawn/venv/py312/bin/activate"
    echo "   或"
    echo "   conda activate fanren"
    echo ""
    read -p "是否继续？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 切换到项目目录
cd "$(dirname "$0")"

# 检查依赖
echo "🔍 检查依赖..."
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "❌ 缺少依赖，请先安装:"
    echo "   pip install fastapi uvicorn"
    exit 1
}

# 检查生成器
echo "🔍 检查生成器..."
python3 -c "
import sys
sys.path.insert(0, 'gen_video')
try:
    from image_generator import ImageGenerator
    print('✅ 图像生成器可用')
except ImportError as e:
    print(f'⚠️  图像生成器导入失败: {e}')
    print('   请确保在正确的环境中运行')
" || true

# 检查是否使用代理
USE_PROXY="${USE_PROXY:-false}"
PROXY_CMD=""

if [ "$USE_PROXY" = "true" ] || [ "$USE_PROXY" = "1" ]; then
    # 检查proxychains4是否可用
    if command -v proxychains4 &> /dev/null; then
        PROXY_CMD="proxychains4"
        echo "✅ 使用 proxychains4 代理"
    elif command -v proxychains &> /dev/null; then
        PROXY_CMD="proxychains"
        echo "✅ 使用 proxychains 代理"
    else
        echo "⚠️  警告: 未找到 proxychains4 或 proxychains"
        echo "   请安装: sudo apt install proxychains4"
        echo "   或设置环境变量: export HTTP_PROXY=... HTTPS_PROXY=..."
        read -p "是否继续不使用代理？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# 启动服务
echo ""
echo "=========================================="
echo "🚀 启动API服务..."
echo "=========================================="
echo ""
echo "📖 API文档: http://localhost:8000/docs"
echo "🔑 测试API Key: test-key-123 (免费版)"
echo "🔑 演示API Key: demo-key-456 (付费版)"
if [ -n "$PROXY_CMD" ]; then
    echo "🌐 代理: 已启用 ($PROXY_CMD)"
fi
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

if [ -n "$PROXY_CMD" ]; then
    $PROXY_CMD python3 gen_video/api/mvp_main.py
else
    python3 gen_video/api/mvp_main.py
fi

