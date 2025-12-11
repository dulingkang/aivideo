#!/bin/bash
# 快速运行测试场景的脚本

echo "=========================================="
echo "🧪 测试场景质量评估"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

# 运行测试脚本
python test_scenes_quality.py

# 检查退出码
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 测试完成！"
    echo "=========================================="
    echo ""
    echo "📁 查看结果："
    echo "   - 图像: outputs/test_scenes_quality/images/"
    echo "   - 报告: outputs/test_scenes_quality/quality_report.md"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 测试失败"
    echo "=========================================="
    echo ""
    exit 1
fi

