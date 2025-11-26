#!/bin/bash
# 完整工作流：narration -> 检索 -> 智能决策 -> 匹配时长/生成场景

set -e

# 配置
NARRATION_FILE="episode_171_narration.json"
EPISODE_ID=171
BASE_DIR="processed"
OUTPUT_BASE="episode_171_production"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 完整工作流：Narration -> 场景匹配 ===${NC}"
echo ""

# 步骤1: 检索场景
echo -e "${GREEN}[步骤1/3] 检索场景${NC}"
python3 tools/video_processing/narration_to_scenes_workflow.py \
  --narration "$NARRATION_FILE" \
  --index ${BASE_DIR}/global_index.faiss \
  --metadata ${BASE_DIR}/index_metadata.json \
  --scenes ${BASE_DIR}/episode_*/scene_metadata.json \
  --output ${OUTPUT_BASE}_search_results \
  --top-k 5

SEARCH_DIR="${OUTPUT_BASE}_search_results"

# 步骤2: 智能决策
echo ""
echo -e "${GREEN}[步骤2/3] 智能决策（检索 vs AI生成）${NC}"
python3 tools/video_processing/smart_scene_matcher.py \
  --narration "$NARRATION_FILE" \
  --search-results-dir "$SEARCH_DIR" \
  --video-base-dir "$BASE_DIR" \
  --output ${OUTPUT_BASE}_decisions \
  --score-threshold-high 0.7 \
  --score-threshold-low 0.5 \
  --duration-tolerance 0.3

DECISIONS_DIR="${OUTPUT_BASE}_decisions"
DECISIONS_FILE="${DECISIONS_DIR}/scene_decisions.json"

# 步骤3: 处理检索场景（匹配时长）
echo ""
echo -e "${GREEN}[步骤3/3] 匹配检索场景的时长${NC}"
python3 tools/video_processing/match_scenes_to_narration.py \
  --narration "$NARRATION_FILE" \
  --search-results-dir "$SEARCH_DIR" \
  --video-base-dir "$BASE_DIR" \
  --output ${OUTPUT_BASE}_matched_videos \
  --top-k 5 \
  --tolerance 0.5

echo ""
echo -e "${BLUE}=== 工作流完成 ===${NC}"
echo ""
echo "输出文件:"
echo "  - 检索结果: ${SEARCH_DIR}/"
echo "  - 决策结果: ${DECISIONS_FILE}"
echo "  - 匹配视频: ${OUTPUT_BASE}_matched_videos/"
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo "  1. 查看决策结果: cat ${DECISIONS_FILE} | jq"
echo "  2. 对于标记为'ai_generate'的场景，使用AI生成工具"
echo "  3. 合并检索场景和AI生成的场景"

