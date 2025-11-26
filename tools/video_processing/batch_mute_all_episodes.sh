#!/bin/bash
# 批量处理所有集数的场景视频静音
# 统一预处理，后续拼接时可以直接使用 -c copy 快速拼接

set -e

cd /vepfs-dev/shawn/vid/fanren

# 集数列表
EPISODES=(142 151 165 170 171)

echo "============================================================"
echo "批量处理所有集数的场景视频静音"
echo "============================================================"
echo ""

total_files=0
processed_episodes=0

for EPISODE in "${EPISODES[@]}"; do
    INPUT_DIR="processed/episode_${EPISODE}/scenes"
    OUTPUT_DIR="processed/episode_${EPISODE}/scenes_muted"
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "⚠️  跳过: 输入目录不存在: $INPUT_DIR"
        continue
    fi
    
    # 统计文件数
    file_count=$(find "$INPUT_DIR" -name "*.mp4" | wc -l)
    total_files=$((total_files + file_count))
    
    echo "============================================================"
    echo "处理集数: $EPISODE"
    echo "  输入: $INPUT_DIR"
    echo "  输出: $OUTPUT_DIR"
    echo "  文件数: $file_count"
    echo "============================================================"
    
    # 执行批量静音处理
    python3 tools/video_processing/batch_mute_videos.py \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --pattern "*.mp4"
    
    if [ $? -eq 0 ]; then
        processed_episodes=$((processed_episodes + 1))
        echo "✅ 集 $EPISODE 处理完成"
    else
        echo "❌ 集 $EPISODE 处理失败"
    fi
    
    echo ""
done

echo "============================================================"
echo "批量处理完成"
echo "============================================================"
echo "处理集数: $processed_episodes / ${#EPISODES[@]}"
echo "总文件数: $total_files"
echo ""
echo "💡 后续使用："
echo "  1. 拼接时使用 scenes_muted 目录中的静音视频"
echo "  2. 可以使用 -c copy 快速拼接（无需重新编码）"
echo "  3. 节省大量时间"
echo ""

