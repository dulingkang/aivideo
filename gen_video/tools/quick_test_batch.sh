#!/bin/bash
# 快速测试批量生成工具

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 激活虚拟环境（如果存在）
if [ -f "/vepfs-dev/shawn/venv/py312/bin/activate" ]; then
    source /vepfs-dev/shawn/venv/py312/bin/activate
fi

# 测试 JSON 文件路径（相对于 gen_video 目录）
JSON_FILE="../lingjie/episode/1.v2-1.json"

if [ ! -f "$JSON_FILE" ]; then
    echo "❌ JSON 文件不存在: $JSON_FILE"
    exit 1
fi

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/outputs/batch_novel_quick_test_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "快速测试批量生成工具"
echo "=========================================="
echo "JSON 文件: $JSON_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "模式: 快速模式（24帧，前3个场景）"
echo "=========================================="
echo ""

# 运行批量生成（快速模式，只生成前3个场景）
python3 "$SCRIPT_DIR/batch_novel_generator.py" \
    --json "$JSON_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --quick \
    --start 0 \
    --end 3 \
    --max-retries 1

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "报告文件: $OUTPUT_DIR/batch_report.md"
echo ""
echo "查看报告:"
echo "  cat $OUTPUT_DIR/batch_report.md"
echo ""

