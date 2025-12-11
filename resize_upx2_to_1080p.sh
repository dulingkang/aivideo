#!/bin/bash
# 批量将 upx2 视频文件缩放到 1920x1080
# 使用方法: ./resize_upx2_to_1080p.sh [--in-place]

set -e

cd /vepfs-dev/shawn/vid/fanren

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 检查参数
IN_PLACE=false
if [ "$1" = "--in-place" ]; then
    IN_PLACE=true
    echo "⚠️  警告: 将直接覆盖原文件"
    read -p "确认继续? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
fi

# 处理 renjie 目录下的文件
echo "处理 renjie 目录下的 upx2 文件..."
if [ "$IN_PLACE" = true ]; then
    python tools/video_processing/resize_videos_to_1080p.py \
        --input "renjie/episode_*_video_upx2.mp4" \
        --in-place \
        --crf 23 \
        --preset medium
else
    python tools/video_processing/resize_videos_to_1080p.py \
        --input "renjie/episode_*_video_upx2.mp4" \
        --crf 23 \
        --preset medium
    echo ""
    echo "✅ 转换完成！新文件已保存为 *_1080p.mp4"
    echo "   如需覆盖原文件，请运行: $0 --in-place"
fi









