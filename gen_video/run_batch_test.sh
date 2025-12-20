#!/bin/bash
# 批量生成测试脚本

set -e

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 进入项目目录
cd /vepfs-dev/shawn/vid/fanren

# 设置环境变量（禁用 proxy，避免卡住）
unset ALL_PROXY
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

echo "============================================================"
echo "开始批量生成测试"
echo "============================================================"
echo ""

# 运行批量生成（只生成前2个场景，用于测试）
python3 gen_video/tools/batch_novel_generator.py \
    --json lingjie/episode/1.v2-1.json \
    --output-dir gen_video/outputs/batch_test_$(date +%Y%m%d_%H%M%S) \
    --start 0 \
    --end 2 \
    --enable-m6 \
    --quick

echo ""
echo "============================================================"
echo "批量生成测试完成"
echo "============================================================"

