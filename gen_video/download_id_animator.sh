#!/bin/bash
# =============================================================================
# ID-Animator 模型下载脚本
# 用于视频身份保持（M6 里程碑）
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   ID-Animator 模型下载脚本${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 目标目录
MODELS_DIR="${1:-models}"
ID_ANIMATOR_DIR="${MODELS_DIR}/ID-Animator"

echo -e "${YELLOW}目标目录: ${ID_ANIMATOR_DIR}${NC}"

# 创建目录
mkdir -p "${ID_ANIMATOR_DIR}"

# =============================================================================
# 1. 下载 ID-Animator Face Adapter（核心组件）
# =============================================================================
echo ""
echo -e "${GREEN}[1/4] 下载 ID-Animator Face Adapter...${NC}"

# ID-Animator 官方模型
if [ ! -f "${ID_ANIMATOR_DIR}/id_animator.pth" ]; then
    echo "  下载 ID-Animator 核心模型..."
    # 从 HuggingFace 下载
    huggingface-cli download ID-Animator/ID-Animator \
        --local-dir "${ID_ANIMATOR_DIR}" \
        --include "*.pth" "*.safetensors" "*.bin" \
        --exclude "*.md" "*.txt" || {
        echo -e "${YELLOW}  尝试使用 wget 下载...${NC}"
        wget -c "https://huggingface.co/ID-Animator/ID-Animator/resolve/main/id_animator.pth" \
            -O "${ID_ANIMATOR_DIR}/id_animator.pth" || {
            echo -e "${RED}  下载失败，请手动下载${NC}"
        }
    }
else
    echo -e "${GREEN}  ✓ ID-Animator 核心模型已存在${NC}"
fi

# =============================================================================
# 2. 下载 AnimateDiff Motion Module
# =============================================================================
echo ""
echo -e "${GREEN}[2/4] 下载 AnimateDiff Motion Module...${NC}"

MOTION_DIR="${MODELS_DIR}/AnimateDiff"
mkdir -p "${MOTION_DIR}"

if [ ! -f "${MOTION_DIR}/mm_sd_v15_v2.ckpt" ]; then
    echo "  下载 AnimateDiff v2 Motion Module..."
    huggingface-cli download guoyww/animatediff-motion-adapter-v1-5-2 \
        --local-dir "${MOTION_DIR}" \
        --include "*.ckpt" "*.safetensors" || {
        echo -e "${YELLOW}  尝试直接下载...${NC}"
        wget -c "https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/resolve/main/diffusion_pytorch_model.safetensors" \
            -O "${MOTION_DIR}/mm_sd_v15_v2.safetensors" || {
            echo -e "${RED}  下载失败，请手动下载${NC}"
        }
    }
else
    echo -e "${GREEN}  ✓ AnimateDiff Motion Module 已存在${NC}"
fi

# =============================================================================
# 3. 下载 SD 1.5 Base Model（如果不存在）
# =============================================================================
echo ""
echo -e "${GREEN}[3/4] 检查 SD 1.5 Base Model...${NC}"

SD15_DIR="${MODELS_DIR}/stable-diffusion-v1-5"

if [ ! -d "${SD15_DIR}" ] || [ ! -f "${SD15_DIR}/v1-5-pruned.safetensors" ]; then
    echo "  下载 SD 1.5 Base Model..."
    echo -e "${YELLOW}  注意: 这会下载约 4GB 的模型文件${NC}"
    
    # 尝试从 HuggingFace 下载
    huggingface-cli download runwayml/stable-diffusion-v1-5 \
        --local-dir "${SD15_DIR}" \
        --include "v1-5-pruned*.safetensors" "*.json" "tokenizer/*" "text_encoder/*" "vae/*" \
        --exclude "*.md" "*.txt" "*.png" || {
        echo -e "${RED}  SD 1.5 下载失败，请手动下载${NC}"
    }
else
    echo -e "${GREEN}  ✓ SD 1.5 Base Model 已存在${NC}"
fi

# =============================================================================
# 4. 检查 InsightFace 模型（已有，复用）
# =============================================================================
echo ""
echo -e "${GREEN}[4/4] 检查 InsightFace 模型...${NC}"

ANTELOPEV2_DIR="${MODELS_DIR}/antelopev2"

if [ -d "${ANTELOPEV2_DIR}" ] && [ -f "${ANTELOPEV2_DIR}/glintr100.onnx" ]; then
    echo -e "${GREEN}  ✓ InsightFace (antelopev2) 已存在，可复用${NC}"
else
    echo -e "${YELLOW}  InsightFace 模型不存在，需要单独下载${NC}"
    echo "  参考: download_pulid_sam2_yolo.sh"
fi

# =============================================================================
# 汇总
# =============================================================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   下载完成${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "模型目录结构:"
echo "  ${MODELS_DIR}/"
echo "  ├── ID-Animator/          # ID-Animator Face Adapter"
echo "  ├── AnimateDiff/          # Motion Module"
echo "  ├── stable-diffusion-v1-5/ # SD 1.5 Base"
echo "  └── antelopev2/           # InsightFace (复用)"
echo ""
echo -e "${GREEN}预计总大小: ~5.5GB${NC}"
echo ""

# 显示实际下载的文件
echo "已下载的文件:"
if [ -d "${ID_ANIMATOR_DIR}" ]; then
    ls -lh "${ID_ANIMATOR_DIR}" 2>/dev/null | head -5
fi
if [ -d "${MOTION_DIR}" ]; then
    ls -lh "${MOTION_DIR}" 2>/dev/null | head -5
fi

echo ""
echo -e "${GREEN}✅ ID-Animator 模型准备完成！${NC}"
echo ""
echo "下一步: 运行测试"
echo "  python test_id_animator_full.py --quick"
echo "  python test_id_animator_full.py"
