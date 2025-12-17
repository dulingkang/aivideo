#!/bin/bash
# PuLID 原生环境设置脚本
# 用于设置 PuLID-FLUX 的完整原生环境

set -e

echo "============================================================"
echo "PuLID 原生环境设置"
echo "============================================================"

# 基础路径
# PuLID 现在作为子模块位于项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PULID_DIR="$PROJECT_ROOT/PuLID"
MODELS_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models"
VENV_DIR="/vepfs-dev/shawn/venv/py312"

# 1. 检查 PuLID 子模块
echo ""
echo "1. 检查 PuLID 子模块..."
if [ ! -d "$PULID_DIR" ]; then
    echo "   ⚠ PuLID 子模块不存在，正在初始化..."
    cd "$PROJECT_ROOT"
    git submodule update --init --recursive PuLID
else
    echo "   ✓ PuLID 子模块已存在: $PULID_DIR"
fi

# 2. 添加 PuLID 到 Python 路径
echo ""
echo "2. 添加 PuLID 到 Python 路径..."
PTH_FILE="$VENV_DIR/lib/python3.12/site-packages/pulid.pth"
echo "$PULID_DIR" > "$PTH_FILE"
echo "   已创建: $PTH_FILE"

# 3. 检查/创建 PuLID 需要的模型目录软链接
echo ""
echo "3. 设置模型目录..."
PULID_MODELS="$PULID_DIR/models"

# 创建 models 目录（如果不存在）
mkdir -p "$PULID_MODELS"

# PuLID 模型软链接
if [ ! -f "$PULID_MODELS/pulid_flux_v0.9.1.safetensors" ]; then
    if [ -f "$MODELS_DIR/pulid/pulid_flux_v0.9.1.safetensors" ]; then
        echo "   创建 PuLID 模型软链接..."
        ln -sf "$MODELS_DIR/pulid/pulid_flux_v0.9.1.safetensors" "$PULID_MODELS/pulid_flux_v0.9.1.safetensors"
    else
        echo "   ⚠ PuLID 模型不存在: $MODELS_DIR/pulid/pulid_flux_v0.9.1.safetensors"
    fi
fi

# AntelopeV2 模型软链接
if [ ! -d "$PULID_MODELS/antelopev2" ]; then
    if [ -d "$MODELS_DIR/antelopev2" ]; then
        echo "   创建 AntelopeV2 软链接..."
        ln -sf "$MODELS_DIR/antelopev2" "$PULID_MODELS/antelopev2"
    else
        echo "   ⚠ AntelopeV2 目录不存在"
    fi
fi

# 4. 检查 Flux 原生模型
echo ""
echo "4. 检查 Flux 原生模型..."

FLUX_DEV="$MODELS_DIR/flux1-dev.safetensors"
AE_MODEL="$MODELS_DIR/ae.safetensors"

if [ ! -f "$FLUX_DEV" ]; then
    echo "   ⚠ Flux 模型不存在: $FLUX_DEV"
    echo "   需要下载: huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir $MODELS_DIR"
    
    # 尝试查找已有的 Flux 模型
    if [ -d "$MODELS_DIR/flux1-dev" ]; then
        echo "   找到 diffusers 格式的 Flux: $MODELS_DIR/flux1-dev"
        echo "   PuLID 需要原生格式 (flux1-dev.safetensors)"
    fi
else
    echo "   ✓ Flux 模型存在: $FLUX_DEV"
    ln -sf "$FLUX_DEV" "$PULID_MODELS/flux1-dev.safetensors" 2>/dev/null || true
fi

if [ ! -f "$AE_MODEL" ]; then
    echo "   ⚠ AutoEncoder 不存在: $AE_MODEL"
    echo "   需要下载: huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir $MODELS_DIR"
else
    echo "   ✓ AutoEncoder 存在: $AE_MODEL"
    ln -sf "$AE_MODEL" "$PULID_MODELS/ae.safetensors" 2>/dev/null || true
fi

# 5. 验证 Python 导入
echo ""
echo "5. 验证 Python 导入..."
cd "$PULID_DIR"

python3 -c "
import sys
sys.path.insert(0, '$PULID_DIR')
try:
    from flux.sampling import denoise, get_noise, get_schedule
    print('   ✓ flux.sampling 导入成功')
except ImportError as e:
    print(f'   ✗ flux.sampling 导入失败: {e}')

try:
    from flux.util import load_flow_model
    print('   ✓ flux.util 导入成功')
except ImportError as e:
    print(f'   ✗ flux.util 导入失败: {e}')

try:
    from pulid.pipeline_flux import PuLIDPipeline
    print('   ✓ pulid.pipeline_flux 导入成功')
except ImportError as e:
    print(f'   ✗ pulid.pipeline_flux 导入失败: {e}')

try:
    from flux.model import Flux
    print('   ✓ flux.model (自定义 Flux) 导入成功')
except ImportError as e:
    print(f'   ✗ flux.model 导入失败: {e}')
"

echo ""
echo "============================================================"
echo "设置完成!"
echo ""
echo "如果缺少 Flux 原生模型，请运行以下命令下载:"
echo "  cd $MODELS_DIR"
echo "  huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir ."
echo "  huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir ."
echo "============================================================"
