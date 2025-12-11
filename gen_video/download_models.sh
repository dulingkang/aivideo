#!/bin/bash
# 多模型下载脚本
# 使用 proxychains4 和虚拟环境下载所有需要的模型

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 虚拟环境路径
VENV_PATH="/vepfs-dev/shawn/venv/py312"
ACTIVATE_SCRIPT="${VENV_PATH}/bin/activate"

# 检查虚拟环境是否存在
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo -e "${RED}错误: 虚拟环境不存在: $ACTIVATE_SCRIPT${NC}"
    exit 1
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境: $VENV_PATH${NC}"
source "$ACTIVATE_SCRIPT"

# 检查 proxychains4 是否可用
if command -v proxychains4 &> /dev/null; then
    PROXY_CMD="proxychains4"
    echo -e "${GREEN}使用 proxychains4 进行下载${NC}"
else
    PROXY_CMD=""
    echo -e "${YELLOW}警告: proxychains4 未找到，将直接下载（可能需要配置代理）${NC}"
fi

# 基础路径
BASE_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models"

# 创建模型目录
mkdir -p "$BASE_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}开始下载多模型组合方案所需模型${NC}"
echo -e "${GREEN}========================================${NC}"

# ============================================
# 1. SD3 Turbo（优先级最高，最简单）
# ============================================
echo -e "\n${YELLOW}[1/4] 下载 SD3 Turbo 模型...${NC}"
SD3_DIR="${BASE_DIR}/sd3-turbo"
mkdir -p "$SD3_DIR"

if [ -d "$SD3_DIR" ] && [ "$(ls -A $SD3_DIR 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ SD3 Turbo 模型已存在，跳过下载${NC}"
else
    echo -e "${YELLOW}下载 SD3 Turbo 模型到: $SD3_DIR${NC}"
    $PROXY_CMD python3 << EOF
import os
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path("$SD3_DIR")
model_id = "stabilityai/stable-diffusion-3-turbo"

print(f"开始下载: {model_id}")
print(f"保存到: {model_dir}")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ SD3 Turbo 下载完成")
except Exception as e:
    print(f"✗ SD3 Turbo 下载失败: {e}")
    raise
EOF
fi

# ============================================
# 2. Flux 1-dev（人物生成）
# ============================================
echo -e "\n${YELLOW}[2/4] 下载 Flux 1-dev 模型...${NC}"
FLUX_DIR="${BASE_DIR}/flux1-dev"
mkdir -p "$FLUX_DIR"

if [ -d "$FLUX_DIR" ] && [ "$(ls -A $FLUX_DIR 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ Flux 1-dev 模型已存在，跳过下载${NC}"
else
    echo -e "${YELLOW}下载 Flux 1-dev 模型到: $FLUX_DIR${NC}"
    echo -e "${YELLOW}注意: Flux 1-dev 模型较大（约 24GB），下载可能需要较长时间${NC}"
    $PROXY_CMD python3 << EOF
import os
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path("$FLUX_DIR")
model_id = "black-forest-labs/FLUX.1-dev"

print(f"开始下载: {model_id}")
print(f"保存到: {model_dir}")
print("注意: 模型较大，请耐心等待...")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ Flux 1-dev 下载完成")
except Exception as e:
    print(f"✗ Flux 1-dev 下载失败: {e}")
    print("提示: 如果下载失败，可能需要：")
    print("  1. 检查 HuggingFace 访问权限")
    print("  2. 配置 HuggingFace token: huggingface-cli login")
    raise
EOF
fi

# ============================================
# 3. Hunyuan-DiT（中文场景）
# ============================================
echo -e "\n${YELLOW}[3/4] 下载 Hunyuan-DiT 模型...${NC}"
HUNYUAN_DIR="${BASE_DIR}/hunyuan-dit"
mkdir -p "$HUNYUAN_DIR"

if [ -d "$HUNYUAN_DIR" ] && [ "$(ls -A $HUNYUAN_DIR 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ Hunyuan-DiT 模型已存在，跳过下载${NC}"
else
    echo -e "${YELLOW}下载 Hunyuan-DiT 模型到: $HUNYUAN_DIR${NC}"
    echo -e "${YELLOW}注意: Hunyuan-DiT 可能需要特殊授权，请参考官方文档${NC}"
    $PROXY_CMD python3 << EOF
import os
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path("$HUNYUAN_DIR")
model_id = "Tencent-Hunyuan/HunyuanDiT"

print(f"开始下载: {model_id}")
print(f"保存到: {model_dir}")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ Hunyuan-DiT 下载完成")
except Exception as e:
    print(f"✗ Hunyuan-DiT 下载失败: {e}")
    print("提示: 如果下载失败，可能需要：")
    print("  1. 检查 HuggingFace 访问权限")
    print("  2. 配置 HuggingFace token: huggingface-cli login")
    print("  3. Hunyuan-DiT 可能需要特殊授权，请访问: https://huggingface.co/Tencent-Hunyuan/HunyuanDiT")
    raise
EOF
fi

# ============================================
# 4. Kolors（真实感场景）
# ============================================
echo -e "\n${YELLOW}[4/4] 下载 Kolors 模型...${NC}"
KOLORS_DIR="${BASE_DIR}/kolors"
mkdir -p "$KOLORS_DIR"

if [ -d "$KOLORS_DIR" ] && [ "$(ls -A $KOLORS_DIR 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ Kolors 模型已存在，跳过下载${NC}"
else
    echo -e "${YELLOW}注意: Kolors 模型在 HuggingFace 上不存在${NC}"
    echo -e "${YELLOW}替代方案: 可以使用其他真实感模型，如 Realistic Vision 或使用 SDXL/Flux 配合真实感 LoRA${NC}"
    echo -e "${YELLOW}跳过 Kolors 下载${NC}"
    # Kolors 模型在 HuggingFace 上不存在，跳过下载
    # 替代方案：
    # 1. 使用 Realistic Vision v5.1: SG161222/Realistic_Vision_V5.1_noVAE
    # 2. 使用 SDXL 或 Flux 配合真实感 LoRA
    echo -e "${GREEN}💡 建议使用替代方案实现真实感场景生成${NC}"
fi

# ============================================
# 完成
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}模型下载完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n已下载的模型："
echo -e "  - SD3 Turbo: ${SD3_DIR}"
echo -e "  - Flux 1-dev: ${FLUX_DIR}"
echo -e "  - Hunyuan-DiT: ${HUNYUAN_DIR}"
echo -e "  - Kolors: ${KOLORS_DIR}"
echo -e "\n${YELLOW}注意:${NC}"
echo -e "  1. 请检查各模型目录是否包含完整的模型文件"
echo -e "  2. 如果某些模型下载失败，请检查："
echo -e "     - HuggingFace 访问权限"
echo -e "     - 网络连接和代理配置"
echo -e "     - 模型是否需要特殊授权"
echo -e "  3. 模型文件较大，请确保有足够的存储空间（约 50-60GB）"

