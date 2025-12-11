#!/bin/bash
# 下载 Realistic Vision 模型（替代 Kolors）

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
MODEL_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/realistic-vision"
mkdir -p "$MODEL_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}下载 Realistic Vision 模型（替代 Kolors）${NC}"
echo -e "${GREEN}========================================${NC}"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ Realistic Vision 模型已存在，跳过下载${NC}"
else
    echo -e "${YELLOW}下载 Realistic Vision 模型到: $MODEL_DIR${NC}"
    $PROXY_CMD python3 << EOF
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path("$MODEL_DIR")
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

print(f"开始下载: {model_id}")
print(f"保存到: {model_dir}")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ Realistic Vision 下载完成")
except Exception as e:
    print(f"✗ Realistic Vision 下载失败: {e}")
    raise
EOF
fi

echo -e "\n${GREEN}✅ 完成！${NC}"
echo -e "模型路径: $MODEL_DIR"

