#!/bin/bash
# 下载 Kolors 模型（快手可图团队开发）
# 官方链接: https://huggingface.co/Kwai-Kolors/Kolors
# GitHub: https://github.com/Kwai-Kolors/Kolors

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 虚拟环境路径
VENV_PATH="/vepfs-dev/shawn/venv/py312"
ACTIVATE_SCRIPT="${VENV_PATH}/bin/activate"

# 检查虚拟环境
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo -e "${RED}错误: 虚拟环境不存在: $ACTIVATE_SCRIPT${NC}"
    exit 1
fi

# 激活虚拟环境
echo -e "${GREEN}激活虚拟环境: $VENV_PATH${NC}"
source "$ACTIVATE_SCRIPT"

# 检查 proxychains4
if command -v proxychains4 &> /dev/null; then
    PROXY_CMD="proxychains4"
    echo -e "${GREEN}使用 proxychains4 进行下载${NC}"
else
    PROXY_CMD=""
    echo -e "${YELLOW}警告: proxychains4 未找到，将直接下载${NC}"
fi

# 模型目录
KOLORS_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/kolors"
CHATGLM3_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/chatglm3"

mkdir -p "$KOLORS_DIR"
mkdir -p "$CHATGLM3_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}下载 Kolors 模型${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}特点:${NC}"
echo -e "  - 真人质感强"
echo -e "  - 肤色真实"
echo -e "  - 五官清晰"
echo -e "  - 光影自然"
echo -e "  - 色彩稳定，不会脏"
echo -e "  - 中文 prompt 理解优秀"
echo -e "${GREEN}========================================${NC}"

# 下载 Kolors-IP-Adapter-FaceID-Plus 模型
echo -e "\n${YELLOW}[步骤 1/1] 下载 Kolors-IP-Adapter-FaceID-Plus 模型...${NC}"
if [ -d "$KOLORS_DIR" ] && [ "$(ls -A $KOLORS_DIR 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ Kolors 模型已存在，跳过下载${NC}"
else
    echo -e "${YELLOW}下载 Kolors-IP-Adapter-FaceID-Plus 模型到: $KOLORS_DIR${NC}"
    echo -e "${YELLOW}注意: 此版本可直接用 diffusers 加载，无需额外的 ChatGLM3${NC}"
    $PROXY_CMD python3 << EOF
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path("$KOLORS_DIR")
model_id = "Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus"

print(f"开始下载: {model_id}")
print(f"保存到: {model_dir}")
print("注意: 如果下载失败，可能需要：")
print("  1. 访问 https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus 申请访问权限")
print("  2. 配置 HuggingFace token: huggingface-cli login")
print("  3. 确保已安装最新版本: pip install -U diffusers transformers accelerate")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ Kolors-IP-Adapter-FaceID-Plus 下载完成")
except Exception as e:
    print(f"✗ Kolors 模型下载失败: {e}")
    print("提示: 可能需要访问 https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus 申请访问权限")
    raise
EOF
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Kolors 模型下载完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n已下载的模型："
echo -e "  - Kolors-IP-Adapter-FaceID-Plus: ${KOLORS_DIR}"
echo -e "\n${YELLOW}注意事项:${NC}"
echo -e "  1. Kolors 可能需要特殊授权，请访问: https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus"
echo -e "  2. 确保已安装最新版本: pip install -U diffusers transformers accelerate"
echo -e "  3. 此版本可直接用 diffusers 加载，使用方式："
echo -e "     from diffusers import DiffusionPipeline"
echo -e "     pipe = DiffusionPipeline.from_pretrained(\"Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus\", dtype=torch.bfloat16, device_map=\"cuda\")"
echo -e "  4. 官方 GitHub: https://github.com/Kwai-Kolors/Kolors"

