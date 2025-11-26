#!/bin/bash
# 重新生成场景描述和索引
# 使用新的逻辑：总是生成视觉描述，优先使用视觉描述

set -e  # 遇到错误退出

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

cd /vepfs-dev/shawn/vid/fanren

# 集数列表
EPISODES=(142 151 165 170 171)

echo "============================================================"
echo "重新生成场景描述（使用新的逻辑：总是生成视觉描述）"
echo "============================================================"
echo ""

# 为每个集数重新生成场景描述
for EPISODE in "${EPISODES[@]}"; do
    echo "============================================================"
    echo "处理集数: $EPISODE"
    echo "============================================================"
    
    EPISODE_DIR="processed/episode_${EPISODE}"
    KEYFRAMES_DIR="${EPISODE_DIR}/keyframes"
    METADATA_JSON="${EPISODE_DIR}/scene_metadata.json"
    ALIGNED_SUBTITLES="${EPISODE_DIR}/aligned_subtitles.json"
    
    # 检查关键帧目录是否存在
    if [ ! -d "$KEYFRAMES_DIR" ]; then
        echo "⚠ 跳过: 关键帧目录不存在: $KEYFRAMES_DIR"
        echo "  提示: 需要先运行 extract_keyframes.py"
        continue
    fi
    
    # 检查关键帧文件
    KEYFRAME_COUNT=$(find "$KEYFRAMES_DIR" -name "*.jpg" | wc -l)
    if [ "$KEYFRAME_COUNT" -eq 0 ]; then
        echo "⚠ 跳过: 关键帧目录为空: $KEYFRAMES_DIR"
        continue
    fi
    
    echo "关键帧数量: $KEYFRAME_COUNT"
    
    # 删除旧的metadata（强制重新生成）
    if [ -f "$METADATA_JSON" ]; then
        echo "删除旧的场景描述: $METADATA_JSON"
        rm -f "$METADATA_JSON"
    fi
    
    # 构建命令（使用proxychains4挂代理）
    DESCRIBE_CMD=(
        "proxychains4" "python3" "tools/video_processing/describe_scenes.py"
        "--input" "$KEYFRAMES_DIR"
        "--output" "$METADATA_JSON"
    )
    
    # 如果有对齐的字幕，添加字幕参数
    if [ -f "$ALIGNED_SUBTITLES" ]; then
        DESCRIBE_CMD+=("--subtitles" "$ALIGNED_SUBTITLES")
        echo "将整合字幕信息到描述中"
    else
        echo "⚠ 警告: 对齐字幕文件不存在: $ALIGNED_SUBTITLES"
        echo "  将继续处理（只使用视觉描述）"
    fi
    
    # 执行场景描述生成
    echo "执行: ${DESCRIBE_CMD[*]}"
    echo ""
    
    if "${DESCRIBE_CMD[@]}"; then
        echo "✓ 集 $EPISODE 场景描述生成完成"
    else
        echo "✗ 集 $EPISODE 场景描述生成失败"
        exit 1
    fi
    
    echo ""
done

echo ""
echo "============================================================"
echo "重新构建跨集索引"
echo "============================================================"
echo ""

# 收集所有场景metadata文件
METADATA_FILES=()
for EPISODE in "${EPISODES[@]}"; do
    METADATA_FILE="processed/episode_${EPISODE}/scene_metadata.json"
    if [ -f "$METADATA_FILE" ]; then
        METADATA_FILES+=("$METADATA_FILE")
        echo "  ✓ 找到: $METADATA_FILE"
    else
        echo "  ✗ 未找到: $METADATA_FILE"
    fi
done

if [ ${#METADATA_FILES[@]} -eq 0 ]; then
    echo "错误: 没有找到任何场景metadata文件"
    exit 1
fi

echo ""
echo "将构建索引，包含 ${#METADATA_FILES[@]} 个集数的场景"

# 删除旧索引
if [ -f "processed/global_index.faiss" ]; then
    echo "删除旧索引: processed/global_index.faiss"
    rm -f "processed/global_index.faiss"
fi

if [ -f "processed/index_metadata.json" ]; then
    echo "删除旧索引metadata: processed/index_metadata.json"
    rm -f "processed/index_metadata.json"
fi

# 构建索引（使用proxychains4挂代理）
BUILD_INDEX_CMD=(
    "proxychains4" "python3" "tools/video_processing/build_index.py"
    "--input" "${METADATA_FILES[@]}"
    "--index" "processed/global_index.faiss"
    "--metadata" "processed/index_metadata.json"
)

echo ""
echo "执行: ${BUILD_INDEX_CMD[*]}"
echo ""

if "${BUILD_INDEX_CMD[@]}"; then
    echo ""
    echo "============================================================"
    echo "✓ 所有场景描述和索引重新生成完成！"
    echo "============================================================"
    echo ""
    echo "下一步: 运行 find_opening_ending_scenes.py 查找开头和结尾场景"
    echo ""
else
    echo ""
    echo "✗ 索引构建失败"
    exit 1
fi

