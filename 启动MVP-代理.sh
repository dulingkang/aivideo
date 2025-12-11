#!/bin/bash
# 使用代理启动MVP API服务

set -e

echo "=========================================="
echo "🚀 启动AI视频生成平台MVP（使用代理）"
echo "=========================================="
echo ""

# 设置使用代理
export USE_PROXY=true

# 执行原启动脚本
exec "$(dirname "$0")/启动MVP.sh"

