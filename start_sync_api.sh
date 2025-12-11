#!/bin/bash
# 启动同步模式API服务器（不依赖Redis）

echo "=========================================="
echo "🚀 启动同步模式API服务器"
echo "=========================================="
echo ""
echo "⚠️  注意："
echo "   - 此模式不依赖Redis，直接同步调用生成器"
echo "   - 图像生成可能需要30-60秒，请耐心等待"
echo "   - 按 Ctrl+C 停止服务器"
echo ""
echo "=========================================="
echo ""

cd "$(dirname "$0")/gen_video/api"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

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
echo "=========================================="
echo "🚀 正在启动API服务器..."
echo "=========================================="
echo ""

# 启动API服务器
python3 main_sync.py

