#!/bin/bash
# 使用代理启动Web界面

set -e

echo "=========================================="
echo "🚀 启动Web界面（使用代理）"
echo "=========================================="
echo ""

# 设置使用代理
export USE_PROXY=true

# 执行原启动脚本
exec "$(dirname "$0")/启动Web界面.sh"

