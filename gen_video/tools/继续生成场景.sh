#!/bin/bash
# 继续生成剩余场景

# 设置环境
cd "$(dirname "$0")/.."
source /vepfs-dev/shawn/venv/py312/bin/activate

# 场景文件
SCENE_FILE="../lingjie/episode/1.v2-1.json"
OUTPUT_DIR="outputs/lingjie_ep1_v2_continue"

echo "📋 继续生成剩余场景"
echo "   场景文件: $SCENE_FILE"
echo "   输出目录: $OUTPUT_DIR"
echo ""

# 检查场景文件
if [ ! -f "$SCENE_FILE" ]; then
    echo "❌ 场景文件不存在: $SCENE_FILE"
    exit 1
fi

# 运行批量生成（跳过已完成的场景）
python3 tools/batch_novel_generator.py \
    --json-path "$SCENE_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --skip-existing \
    --continue-on-error

echo ""
echo "✅ 生成完成！"
echo "   查看报告: $OUTPUT_DIR/batch_report.json"

