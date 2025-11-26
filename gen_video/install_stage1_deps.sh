#!/bin/bash
# 阶段1依赖安装脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "阶段1依赖安装脚本"
echo "============================================================"

# 设置缓存目录到挂载盘（避免占用根目录空间）
export PIP_CACHE_DIR=/vepfs-dev/shawn/.cache/pip
export HF_HOME=/vepfs-dev/shawn/.cache/huggingface
export TRANSFORMERS_CACHE=/vepfs-dev/shawn/.cache/huggingface
export HF_DATASETS_CACHE=/vepfs-dev/shawn/.cache/huggingface/datasets

# 确保缓存目录存在
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

echo "✓ 缓存目录已设置为挂载盘:"
echo "  Pip: $PIP_CACHE_DIR"
echo "  HuggingFace: $HF_HOME"
echo ""

# 激活 Python 环境
if [ -f "/vepfs-dev/shawn/venv/py312/bin/activate" ]; then
    source /vepfs-dev/shawn/venv/py312/bin/activate
    echo "✓ 已激活 Python 环境"
fi

echo ""
echo "安装阶段1所需的 Python 包..."
echo ""

# InstantID 处理
echo "[1/4] 检查 InstantID..."
echo "  官方仓库: https://github.com/instantX-research/InstantID"
echo "  注意: InstantID 仓库没有 setup.py，不需要安装包"
echo "  代码会直接从克隆的仓库导入 pipeline"

# 检查 InstantID 仓库是否存在
if [ -d "/vepfs-dev/shawn/vid/fanren/InstantID" ]; then
    echo "  ✓ InstantID 仓库已存在: /vepfs-dev/shawn/vid/fanren/InstantID"
else
    echo "  ⚠ InstantID 仓库不存在，但代码会尝试自动查找"
fi

# 检查 InstantID 依赖
echo "  检查 InstantID 所需依赖..."
missing_deps=()
for dep in opencv-python transformers accelerate insightface diffusers; do
    if ! pip show "$dep" > /dev/null 2>&1; then
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -eq 0 ]; then
    echo "  ✓ 所有依赖已安装"
else
    echo "  ⚠ 缺少依赖: ${missing_deps[*]}"
    echo "  安装缺失的依赖（使用中科大镜像）..."
    pip install "${missing_deps[@]}" -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host=mirrors.ustc.edu.cn || {
        echo "  ⚠ 中科大镜像失败，尝试官方源..."
        pip install "${missing_deps[@]}" || {
            echo "  ✗ 依赖安装失败"
            exit 1
        }
    }
fi

# 准备 CosyVoice 环境（从 GitHub 克隆并安装依赖）
echo ""
echo "[2/4] 准备 CosyVoice 环境（从 GitHub 克隆）..."
echo "  官方仓库: https://github.com/FunAudioLLM/CosyVoice"
echo "  注意: CosyVoice 不是 pip 包，需要克隆仓库并安装依赖"
echo "  推荐: Python 3.10 (当前环境: $(python3 --version 2>&1 | cut -d' ' -f2))"

COSYVOICE_DIR="/vepfs-dev/shawn/vid/fanren/CosyVoice"

# 先卸载旧版本（如果存在）
pip uninstall -y cosyvoice 2>/dev/null || true

# 克隆或更新 CosyVoice 仓库（使用 --recursive 自动初始化子模块）
if [ -d "$COSYVOICE_DIR" ]; then
    echo "  ✓ CosyVoice 仓库已存在: $COSYVOICE_DIR"
    echo "  更新仓库..."
    cd "$COSYVOICE_DIR"
    git pull || echo "  ⚠ Git 更新失败，继续使用现有版本"
    # 确保子模块已初始化
    echo "  更新子模块..."
    git submodule update --init --recursive || {
        echo "  ⚠ 子模块更新失败，继续..."
    }
else
    echo "  克隆 CosyVoice 仓库（包含子模块）..."
    cd /vepfs-dev/shawn/vid/fanren
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git || {
        echo "✗ Git 克隆失败，尝试分步克隆..."
        git clone https://github.com/FunAudioLLM/CosyVoice.git || {
            echo "✗ Git 克隆失败"
            exit 1
        }
        cd "$COSYVOICE_DIR"
        echo "  初始化子模块..."
        git submodule update --init --recursive || {
            echo "  ⚠ 子模块初始化失败，可能需要手动运行:"
            echo "    cd $COSYVOICE_DIR && git submodule update --init --recursive"
        }
    }
    cd "$COSYVOICE_DIR"
fi

# 安装依赖（使用中科大镜像或官方源）
if [ -f "requirements.txt" ]; then
    echo "  安装依赖（从 requirements.txt，使用中科大镜像）..."
    pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host=mirrors.ustc.edu.cn || {
        echo "  ⚠ 使用中科大镜像安装失败，尝试官方源..."
        pip install -r requirements.txt || {
            echo "  ⚠ 部分依赖安装失败，继续..."
        }
    }
else
    echo "  ⚠ 未找到 requirements.txt"
fi

# 检查是否需要安装系统依赖（sox）
if ! command -v sox &> /dev/null; then
    echo "  ⚠ 未检测到 sox，建议安装:"
    echo "    Ubuntu/Debian: sudo apt-get install sox libsox-dev"
    echo "    CentOS/RHEL: sudo yum install sox sox-devel"
    echo "  注意: 如果不安装 sox，某些功能可能不可用"
else
    echo "  ✓ sox 已安装: $(sox --version 2>&1 | head -1)"
fi

cd "$SCRIPT_DIR"
echo "  ✓ CosyVoice 环境准备完成"
echo "  提示: 代码中需要将 CosyVoice 目录添加到 sys.path"
echo "  模型下载: 首次运行时会自动下载，或手动下载到 pretrained_models/ 目录"

# 检查 WhisperX（可能已安装）
echo ""
echo "[3/4] 检查 WhisperX..."
if pip show whisperx > /dev/null 2>&1; then
    echo "✓ WhisperX 已安装"
else
    echo "安装 WhisperX（使用中科大镜像）..."
    pip install whisperx -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host=mirrors.ustc.edu.cn || {
        echo "  ⚠ 中科大镜像失败，尝试官方源..."
        pip install whisperx
    }
fi

# 更新 pyannote.audio（对齐功能需要）
echo ""
echo "[4/4] 更新 pyannote.audio（使用中科大镜像）..."
pip install --upgrade pyannote.audio -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host=mirrors.ustc.edu.cn || {
    echo "  ⚠ 中科大镜像失败，尝试官方源..."
    pip install --upgrade pyannote.audio || {
        echo "⚠ pyannote.audio 更新失败，但可能不影响使用"
    }
}

echo ""
echo "============================================================"
echo "✓ 依赖安装完成！"
echo "============================================================"
echo ""
echo "已安装的包："
pip show instantid cosyvoice whisperx 2>/dev/null | grep -E "^Name:|^Version:" || echo "部分包可能未正确安装"
echo ""
echo "下一步：运行测试脚本验证功能"
echo "  python3 test_stage1.py"

