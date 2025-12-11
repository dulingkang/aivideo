#!/bin/bash
# 启动同步模式API服务器（使用指定虚拟环境）

VENV_PATH="/vepfs-dev/shawn/venv/py312"
PROJECT_ROOT="/vepfs-dev/shawn/vid/fanren"

echo "=========================================="
echo "🚀 启动同步模式API服务器（虚拟环境）"
echo "=========================================="
echo ""

# 检查虚拟环境是否存在
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "❌ 错误: 虚拟环境不存在: $VENV_PATH"
    exit 1
fi

# 激活虚拟环境
echo "📦 激活虚拟环境: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# 检查Python
PYTHON_VERSION=$(python3 --version 2>&1)
echo "✅ Python版本: $PYTHON_VERSION"
echo ""

# 检查FastAPI
if python3 -c "import fastapi" 2>/dev/null; then
    FASTAPI_VERSION=$(python3 -c "import fastapi; print(fastapi.__version__)" 2>/dev/null)
    echo "✅ FastAPI已安装: v$FASTAPI_VERSION"
else
    echo "⚠️  警告: FastAPI未安装"
    echo "   正在安装必要依赖..."
    pip install fastapi uvicorn pydantic python-multipart
fi

# 切换到项目目录
cd "$PROJECT_ROOT/gen_video/api"

# 检查配置文件
if [ ! -f "../config.yaml" ]; then
    echo "❌ 错误: 配置文件不存在: gen_video/config.yaml"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""
echo "📍 工作目录: $(pwd)"
echo "🌐 API地址: http://localhost:8000"
echo "📚 API文档: http://localhost:8000/docs"
echo ""
echo "⚠️  注意："
echo "   - 此模式不依赖Redis，直接同步调用生成器"
echo "   - 图像生成可能需要30-60秒，请耐心等待"
echo "   - 按 Ctrl+C 停止服务器"
echo ""
echo "=========================================="
echo "🚀 正在启动API服务器..."
echo "=========================================="
echo ""

# 启动API服务器
python3 main_sync.py

