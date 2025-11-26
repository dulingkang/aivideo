#!/bin/bash
# SVD 流程测试脚本

set -e

echo "============================================================"
echo "SVD 流程测试"
echo "============================================================"

# 激活虚拟环境
VENV_PATH="/vepfs-dev/shawn/venv/py312"
if [ -d "$VENV_PATH" ]; then
    echo "激活虚拟环境: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "⚠ 未找到虚拟环境，使用系统 Python"
fi

# 进入工作目录
cd /vepfs-dev/shawn/vid/fanren/gen_video

# 检查脚本文件
SCRIPT_JSON="../lingjie/5-青罗沙漠.json"
if [ ! -f "$SCRIPT_JSON" ]; then
    echo "✗ 脚本文件不存在: $SCRIPT_JSON"
    echo "  请先运行: python build_markdown_from_json.py --json $SCRIPT_JSON --episode 5 --title 青罗沙漠 --output temp/ep5.md"
    exit 1
fi

# 如果 Markdown 文件不存在，先生成
MARKDOWN_FILE="temp/ep5.md"
if [ ! -f "$MARKDOWN_FILE" ]; then
    echo "生成 Markdown 文件..."
    python build_markdown_from_json.py \
        --json "$SCRIPT_JSON" \
        --episode 5 \
        --title "青罗沙漠" \
        --output "$MARKDOWN_FILE"
    echo "✓ Markdown 文件已生成: $MARKDOWN_FILE"
fi

# 检查图像目录
IMAGE_DIR="../lingjie/img2/jpgsrc"
if [ ! -d "$IMAGE_DIR" ]; then
    echo "⚠ 图像目录不存在: $IMAGE_DIR"
    echo "  将使用自动生成的图像"
fi

# 运行完整流程（使用 SVD）
echo ""
echo "开始运行 SVD 流程..."
echo "============================================================"

python run_pipeline.py \
    --config config.yaml \
    --markdown "$MARKDOWN_FILE" \
    --image-dir "$IMAGE_DIR" \
    --output "lingjie_ep5_svd_test" \
    --max-scenes 3

echo ""
echo "============================================================"
echo "✓ SVD 流程测试完成！"
echo "============================================================"
echo ""
echo "输出目录: outputs/lingjie_ep5_svd_test/"
echo "  - images/     : 生成的图像"
echo "  - videos/     : 生成的视频片段"
echo "  - audio.wav   : 生成的配音"
echo "  - subtitle.srt: 生成的字幕"
echo "  - lingjie_ep5_svd_test.mp4: 最终视频"
echo ""
echo "查看结果:"
echo "  ls -lh outputs/lingjie_ep5_svd_test/"
echo ""

