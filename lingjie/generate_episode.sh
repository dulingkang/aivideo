#!/bin/bash
# 根据 episode_XXX.json 生成完整视频（完整工作流）
# 包含：TTS配音生成 → 视频匹配/生成 → 拼接
# 使用方法: 
#   单个集数: ./generate_episode.sh [集数]
#   所有集数: ./generate_episode.sh --all
#   跳过已存在: ./generate_episode.sh --all --skip-existing
# 示例: 
#   ./generate_episode.sh 1
#   ./generate_episode.sh --all
#   ./generate_episode.sh --all --skip-existing

# 注意：批量处理时不使用 set -e，以便单个文件失败时继续处理其他文件
# 单个处理时会在 process_episode 函数中检查错误

cd /vepfs-dev/shawn/vid/fanren

# 解析参数
PROCESS_ALL=false
SKIP_EXISTING=false
EPISODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            PROCESS_ALL=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        *)
            if [ -z "$EPISODE" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
                EPISODE=$1
            fi
            shift
            ;;
    esac
done

# 如果没有指定参数，默认处理所有
if [ -z "$EPISODE" ] && [ "$PROCESS_ALL" = false ]; then
    PROCESS_ALL=true
    echo "⚠️  未指定集数，将处理所有 JSON 文件"
    echo "  用法: $0 [集数] 或 $0 --all"
    echo "  示例: $0 1 或 $0 --all"
    echo ""
fi

# 激活虚拟环境
source /vepfs-dev/shawn/venv/py312/bin/activate

# 处理单个集数的函数
process_episode() {
    local EPISODE=$1
    local JSON_FILE="lingjie/episode/${EPISODE}.json"
    
    echo ""
    echo "============================================================"
    echo "处理集数: ${EPISODE}"
    echo "============================================================"
    echo ""
    
    # 检查文件
    if [ ! -f "$JSON_FILE" ]; then
        echo "❌ 错误: $JSON_FILE 不存在，跳过"
        return 1
    fi
    
    # 检查输出文件是否已存在
    OUTPUT_FILE="lingjie/episode_${EPISODE}_video.mp4"
    if [ "$SKIP_EXISTING" = true ] && [ -f "$OUTPUT_FILE" ]; then
        echo "⏭️  跳过: $OUTPUT_FILE 已存在"
        return 0
    fi

    # 检查开头视频（可选）
    if [ ! -f "lingjie/opening_video_raw.mp4" ]; then
        echo "⚠️  警告: lingjie/opening_video_raw.mp4 不存在，将跳过开头"
        OPENING_FLAG=""
    else
        OPENING_FLAG="--opening lingjie/opening_video_raw.mp4"
    fi
    
    # 检查结尾视频（可选）
    if [ ! -f "lingjie/ending_video_raw.mp4" ]; then
        echo "⚠️  警告: lingjie/ending_video_raw.mp4 不存在，将跳过结尾"
        ENDING_FLAG=""
    else
        ENDING_FLAG="--ending lingjie/ending_video_raw.mp4"
    fi
    
    # 检查gen_video配置
    GEN_VIDEO_CONFIG="gen_video/config.yaml"
    if [ ! -f "$GEN_VIDEO_CONFIG" ]; then
        echo "⚠️  警告: gen_video配置文件不存在: $GEN_VIDEO_CONFIG"
        echo "  将只支持原视频检索，不支持AI生成"
        GEN_CONFIG_FLAG=""
    else
        GEN_CONFIG_FLAG="--gen-video-config $GEN_VIDEO_CONFIG"
    fi
    
    # 自动发现所有可用的场景元数据文件
    # 这样可以包含所有已处理的集数，而不需要硬编码
    SCENE_FILES=""
    for SCENE_FILE in processed/episode_*/scene_metadata.json; do
        if [ -f "$SCENE_FILE" ]; then
            SCENE_FILES="$SCENE_FILES $SCENE_FILE"
        fi
    done
    
    # 去除开头的空格
    SCENE_FILES=$(echo "$SCENE_FILES" | sed 's/^[[:space:]]*//')
    
    if [ -z "$SCENE_FILES" ]; then
        echo "⚠️  警告: 未找到任何场景元数据文件"
        echo "  视频检索功能可能受限"
        SCENE_FILES=""
    else
        SCENE_COUNT=$(echo "$SCENE_FILES" | wc -w)
        echo "ℹ️  找到 $SCENE_COUNT 个场景元数据文件"
        if [ "$SCENE_COUNT" -le 5 ]; then
            echo "  文件列表: $SCENE_FILES"
        else
            echo "  文件列表: $(echo "$SCENE_FILES" | tr ' ' '\n' | head -5 | tr '\n' ' ') ... (共 $SCENE_COUNT 个)"
        fi
    fi
    
    # 执行生成
    echo "开始生成视频..."
    
    # 构建命令参数
    CMD_ARGS=(
        --json "$JSON_FILE"
        --index processed/global_index.faiss
        --metadata processed/index_metadata.json
        --video-dir processed/
        --output "$OUTPUT_FILE"
    )
    
    # 如果有场景文件，添加 --scenes 参数
    if [ -n "$SCENE_FILES" ]; then
        # 将 SCENE_FILES 字符串拆分为数组
        read -ra SCENE_ARRAY <<< "$SCENE_FILES"
        CMD_ARGS+=(--scenes "${SCENE_ARRAY[@]}")
    fi
    
    # 添加开头视频（如果存在）
    if [ -n "$OPENING_FLAG" ]; then
        CMD_ARGS+=(--opening lingjie/opening_video_raw.mp4)
    fi
    
    # 添加结尾视频（如果存在）
    if [ -n "$ENDING_FLAG" ]; then
        CMD_ARGS+=(--ending lingjie/ending_video_raw.mp4)
    fi
    
    # 添加 gen_video 配置（如果存在）
    if [ -n "$GEN_CONFIG_FLAG" ]; then
        CMD_ARGS+=(--gen-video-config "$GEN_VIDEO_CONFIG")
    fi
    
    if python3 generate_video_from_json_complete.py "${CMD_ARGS[@]}"; then
        echo ""
        echo "✅ 视频生成完成: $OUTPUT_FILE"
        return 0
    else
        echo ""
        echo "❌ 视频生成失败: $OUTPUT_FILE"
        return 1
    fi
}

# 主逻辑
if [ "$PROCESS_ALL" = true ]; then
    # 处理所有 JSON 文件
    echo "============================================================"
    echo "批量处理所有 JSON 文件"
    echo "============================================================"
    if [ "$SKIP_EXISTING" = true ]; then
        echo "⏭️  将跳过已存在的输出文件"
    fi
    echo ""
    
    # 获取所有 JSON 文件并按数字大小排序
    JSON_FILES=($(ls -1 lingjie/episode/*.json | sort -V))
    TOTAL=${#JSON_FILES[@]}
    SUCCESS=0
    FAILED=0
    SKIPPED=0
    
    for JSON_FILE in "${JSON_FILES[@]}"; do
        # 从文件名提取集数（例如：lingjie/episode/1.json -> 1）
        EPISODE_NUM=$(basename "$JSON_FILE" .json)
        
        if [ "$SKIP_EXISTING" = true ] && [ -f "lingjie/episode_${EPISODE_NUM}_video.mp4" ]; then
            echo "⏭️  跳过: episode_${EPISODE_NUM}_video.mp4 已存在"
            ((SKIPPED++))
            continue
        fi
        
        if process_episode "$EPISODE_NUM"; then
            ((SUCCESS++))
        else
            ((FAILED++))
        fi
    done
    
    echo ""
    echo "============================================================"
    echo "批量处理完成"
    echo "============================================================"
    echo "总计: $TOTAL 个文件"
    echo "成功: $SUCCESS 个"
    echo "失败: $FAILED 个"
    if [ "$SKIP_EXISTING" = true ]; then
        echo "跳过: $SKIPPED 个（已存在）"
    fi
    echo ""
else
    # 处理单个集数
    process_episode "$EPISODE"
    
    echo ""
    echo "============================================================"
    echo "✅ 视频生成完成！"
    echo "============================================================"
    echo "输出文件: lingjie/episode_${EPISODE}_video.mp4"

fi


