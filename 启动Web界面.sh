#!/bin/bash
# 启动Web界面（Gradio快速版）

set -e

echo "=========================================="
echo "🚀 启动AI视频生成平台Web界面"
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
    echo "   建议先激活虚拟环境"
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
python3 -c "import gradio" 2>/dev/null || {
    echo "❌ 缺少gradio，正在安装..."
    pip install gradio
}

# 检查API服务是否运行
echo "🔍 检查API服务..."
if ! curl -s http://localhost:8000/api/v1/health > /dev/null; then
    echo "⚠️  API服务未运行，请先启动API服务："
    echo "   python gen_video/api/mvp_main.py"
    echo ""
    read -p "是否现在启动API服务？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 启动API服务（后台）..."
        python3 gen_video/api/mvp_main.py > /dev/null 2>&1 &
        API_PID=$!
        echo "   API服务PID: $API_PID"
        sleep 3
    else
        exit 1
    fi
fi

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

# 启动Web界面
echo ""
echo "=========================================="
echo "🚀 启动Web界面..."
echo "=========================================="
echo ""
echo "🌐 Web界面: http://localhost:7860"
echo "📖 API文档: http://localhost:8000/docs"
echo "🔑 默认API Key: test-key-123"
if [ -n "$PROXY_CMD" ]; then
    echo "🌐 代理: 已启用 ($PROXY_CMD)"
fi
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

if [ -n "$PROXY_CMD" ]; then
    $PROXY_CMD python3 gen_video/api/web_ui.py
else
    python3 gen_video/api/web_ui.py
fi

