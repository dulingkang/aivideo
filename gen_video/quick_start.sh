#!/bin/bash
# 快速启动脚本

echo "=========================================="
echo "AI视频生成系统 - 快速启动"
echo "=========================================="

# 进入脚本目录
cd "$(dirname "$0")"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 运行测试
echo ""
echo "运行系统测试..."
python3 test_script.py

if [ $? -ne 0 ]; then
    echo ""
    echo "警告: 部分测试未通过，但可以继续运行"
    echo "按 Enter 继续，或 Ctrl+C 退出"
    read
fi

# 运行流水线
echo ""
echo "开始生成视频..."
echo ""

python3 run_pipeline.py \
    --markdown ../灵界/2.md \
    --image-dir ../灵界/img2/jpgsrc \
    --output lingjie_ep2

echo ""
echo "完成！"

