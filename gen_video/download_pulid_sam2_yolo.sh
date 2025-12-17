#!/bin/bash
# PuLID / SAM2 / YOLO 模型下载脚本
# 用于角色一致性系统升级

echo "==========================================="
echo "PuLID / SAM2 / YOLO 模型下载脚本"
echo "用于角色一致性系统升级"
echo "==========================================="

# 配置 - 服务器路径
BASE_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models"
VENV_PATH="/vepfs-dev/shawn/venv/py312/bin/activate"

# 检查是否在服务器上运行
if [ ! -d "/vepfs-dev" ]; then
    echo "⚠ 检测到不在服务器上，使用本地路径..."
    BASE_DIR="/home/dss/app/aivideo/gen_video/models"
fi

echo ""
echo "模型目录: $BASE_DIR"
echo ""

# ============================================
# 0. 检查已有模型状态
# ============================================
echo "==========================================="
echo "已有模型检查"
echo "==========================================="

check_exist() {
    local name=$1
    local path=$2
    if [ -f "$path" ] || [ -d "$path" ]; then
        echo "✓ $name"
        return 0
    else
        echo "❌ $name (需要下载)"
        return 1
    fi
}

echo ""
echo "图像生成模型:"
check_exist "Flux 1 Dev" "$BASE_DIR/flux1-dev"
check_exist "Flux 2 Dev" "$BASE_DIR/flux2-dev"
check_exist "SDXL Base" "$BASE_DIR/sdxl-base"
check_exist "SD3 Turbo" "$BASE_DIR/sd3-turbo"

echo ""
echo "身份保持模型:"
check_exist "InstantID" "$BASE_DIR/instantid"
check_exist "IP-Adapter" "$BASE_DIR/ip-adapter"
check_exist "AntelopeV2 (InsightFace)" "$BASE_DIR/antelopev2"
check_exist "PuLID-FLUX" "$BASE_DIR/pulid/pulid_flux_v0.9.1.safetensors"

echo ""
echo "视频生成模型:"
check_exist "CogVideoX-5b" "$BASE_DIR/CogVideoX-5b"
check_exist "CogVideoX 1.5-5B-I2V" "$BASE_DIR/CogVideoX1.5-5B-I2V"
check_exist "HunyuanVideo 1.5 480p" "$BASE_DIR/HunyuanVideo-1.5-Diffusers-480p_i2v"
check_exist "HunyuanVideo 1.5 720p" "$BASE_DIR/hunyuan-video-1.5-720p-i2v"

echo ""
echo "分割/检测模型:"
check_exist "SAM2" "$BASE_DIR/sam2"

echo ""
echo "==========================================="
echo "需要下载的模型: PuLID-FLUX, SAM2, YOLO"
echo "==========================================="

# 创建目录结构（只创建需要的目录）
echo ""
echo "1. 创建模型目录结构..."
mkdir -p "$BASE_DIR/pulid"
mkdir -p "$BASE_DIR/sam2"
echo "   ✓ 目录结构已创建"

# 激活虚拟环境
echo ""
echo "2. 激活虚拟环境..."
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "   ✓ 虚拟环境已激活"
else
    echo "   ⚠ 虚拟环境不存在，使用系统Python"
fi

# 检查huggingface-cli
echo ""
echo "3. 检查huggingface-cli..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "   ⚠ huggingface-cli未安装，正在安装..."
    pip install -U huggingface_hub[cli]
    echo "   ✓ huggingface-cli已安装"
else
    echo "   ✓ huggingface-cli已安装"
fi

# 检查HuggingFace登录状态
echo ""
echo "4. 检查HuggingFace登录状态..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "   ⚠ 未登录HuggingFace"
    if [ -n "$HF_TOKEN" ]; then
        echo "   ℹ 检测到HF_TOKEN环境变量，使用它登录..."
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    else
        echo "   ⚠ 请设置HF_TOKEN环境变量或运行: huggingface-cli login"
        echo "   部分模型可能无法下载"
    fi
else
    echo "   ✓ 已登录HuggingFace"
fi

echo ""
echo "==========================================="
echo "开始下载模型"
echo "==========================================="

# ============================================
# 1. 下载 PuLID-FLUX
# ============================================
echo ""
echo "5. 下载 PuLID-FLUX 模型..."
echo "   来源: guozinan/PuLID"
echo "   目标: $BASE_DIR/pulid/"

PULID_FILE="$BASE_DIR/pulid/pulid_flux_v0.9.1.safetensors"
# 检查文件是否存在且大于 1GB (正常应该约 1.14GB)
PULID_MIN_SIZE=1000000000  # 1GB in bytes

need_download=false
if [ -f "$PULID_FILE" ]; then
    file_size=$(stat -c%s "$PULID_FILE" 2>/dev/null || echo "0")
    if [ "$file_size" -lt "$PULID_MIN_SIZE" ]; then
        echo "   ⚠ PuLID 文件存在但大小异常 (${file_size} bytes), 重新下载..."
        rm -f "$PULID_FILE"
        need_download=true
    else
        echo "   ✓ pulid_flux_v0.9.1.safetensors 已存在 ($(numfmt --to=iec $file_size)), 跳过"
    fi
else
    need_download=true
fi

if [ "$need_download" = true ]; then
    echo "   ℹ 开始下载 PuLID-FLUX-v0.9.1 (约 1.14GB)..."
    
    # 方式1: 使用 huggingface-cli 下载特定文件（新版语法）
    huggingface-cli download guozinan/PuLID \
        pulid_flux_v0.9.1.safetensors \
        --local-dir "$BASE_DIR/pulid"
    
    # 检查下载是否成功
    if [ -f "$PULID_FILE" ]; then
        file_size=$(stat -c%s "$PULID_FILE" 2>/dev/null || echo "0")
        if [ "$file_size" -gt "$PULID_MIN_SIZE" ]; then
            echo "   ✓ PuLID-FLUX 下载完成 ($(numfmt --to=iec $file_size))"
        else
            echo "   ⚠ 下载可能不完整，尝试 wget..."
            rm -f "$PULID_FILE"
            
            # 方式2: 直接用wget下载
            wget -c "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors" \
                -O "$PULID_FILE"
        fi
    else
        echo "   ⚠ huggingface-cli 方式失败，尝试 wget..."
        
        # 方式2: 直接用wget下载
        wget -c "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.1.safetensors" \
            -O "$PULID_FILE"
    fi
    
    # 最终验证
    if [ -f "$PULID_FILE" ]; then
        file_size=$(stat -c%s "$PULID_FILE" 2>/dev/null || echo "0")
        if [ "$file_size" -gt "$PULID_MIN_SIZE" ]; then
            echo "   ✓ PuLID-FLUX 下载验证通过"
        else
            echo "   ❌ PuLID-FLUX 下载失败或不完整"
            echo "   请手动下载: https://huggingface.co/guozinan/PuLID/tree/main"
        fi
    else
        echo "   ❌ PuLID-FLUX 下载失败"
        echo "   请手动下载: https://huggingface.co/guozinan/PuLID/tree/main"
    fi
fi

# ============================================
# 2. 下载 EVA-CLIP (PuLID依赖)
# ============================================
echo ""
echo "6. 下载 EVA-CLIP 模型 (PuLID依赖)..."
echo "   来源: QuanSun/EVA-CLIP"
echo "   目标: $BASE_DIR/clip/"

CLIP_FILE="$BASE_DIR/clip/EVA02_CLIP_L_336_psz14_s6B.pt"
if [ -f "$CLIP_FILE" ]; then
    echo "   ✓ EVA02_CLIP 已存在，跳过"
else
    echo "   ℹ 开始下载 EVA-CLIP..."
    # EVA-CLIP 通常会自动下载，这里手动下载备用
    huggingface-cli download QuanSun/EVA-CLIP \
        --include "EVA02_CLIP_L_336_psz14_s6B.pt" \
        --local-dir "$BASE_DIR/clip" \
        --local-dir-use-symlinks False || echo "   ⚠ EVA-CLIP可能在首次使用时自动下载"
fi

# ============================================
# 3. 检查 InsightFace AntelopeV2
# ============================================
echo ""
echo "7. 检查 InsightFace AntelopeV2 模型..."

# 检查两个可能的位置
ANTELOPE_DIR_1="$BASE_DIR/antelopev2"
ANTELOPE_DIR_2="$BASE_DIR/insightface/models/antelopev2"

if [ -f "$ANTELOPE_DIR_1/1k3d68.onnx" ]; then
    echo "   ✓ AntelopeV2 已存在: $ANTELOPE_DIR_1"
    ANTELOPE_DIR="$ANTELOPE_DIR_1"
elif [ -f "$ANTELOPE_DIR_2/1k3d68.onnx" ]; then
    echo "   ✓ AntelopeV2 已存在: $ANTELOPE_DIR_2"
    ANTELOPE_DIR="$ANTELOPE_DIR_2"
else
    echo "   ℹ AntelopeV2 未找到，开始下载..."
    ANTELOPE_DIR="$ANTELOPE_DIR_1"
    mkdir -p "$ANTELOPE_DIR"
    huggingface-cli download DIAMONIK7777/antelopev2 \
        --local-dir "$ANTELOPE_DIR" \
        --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "   ✓ AntelopeV2 下载完成"
    else
        echo "   ❌ AntelopeV2 下载失败"
        echo "   备用下载方式: https://github.com/deepinsight/insightface/tree/master/python-package"
    fi
fi

# ============================================
# 4. 下载 SAM2 (Segment Anything 2)
# ============================================
echo ""
echo "8. 下载 SAM2 模型..."
echo "   来源: facebook/sam2-hiera-large"
echo "   目标: $BASE_DIR/sam2/"

SAM2_FILE="$BASE_DIR/sam2/sam2_hiera_large.pt"
if [ -f "$SAM2_FILE" ] || [ -f "$BASE_DIR/sam2/model.safetensors" ]; then
    echo "   ✓ SAM2 已存在，跳过"
else
    echo "   ℹ 开始下载 SAM2..."
    
    # 方式1: 从HuggingFace下载
    huggingface-cli download facebook/sam2-hiera-large \
        --local-dir "$BASE_DIR/sam2" \
        --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "   ✓ SAM2 下载完成"
    else
        echo "   ⚠ HuggingFace下载失败，尝试从GitHub下载..."
        
        # 方式2: 从GitHub下载checkpoint
        cd "$BASE_DIR/sam2"
        wget -c "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" \
            -O sam2_hiera_large.pt
        
        if [ $? -eq 0 ]; then
            echo "   ✓ SAM2 从GitHub下载完成"
        else
            echo "   ❌ SAM2 下载失败"
        fi
    fi
fi

# ============================================
# 5. 安装 YOLO (自动下载权重)
# ============================================
echo ""
echo "9. 配置 YOLOv8..."
echo "   来源: ultralytics"
echo "   说明: YOLOv8会在首次使用时自动下载权重"

# 检查ultralytics是否安装
if python -c "import ultralytics" 2>/dev/null; then
    echo "   ✓ ultralytics 已安装"
else
    echo "   ℹ 安装 ultralytics..."
    pip install ultralytics
    echo "   ✓ ultralytics 已安装"
fi

# 预下载YOLOv8n权重（最小版本）
echo "   ℹ 预下载 YOLOv8 权重..."
python -c "
from ultralytics import YOLO

# 下载不同大小的模型
models = ['yolov8n.pt', 'yolov8s.pt']  # nano和small版本
for model_name in models:
    try:
        print(f'   ℹ 下载 {model_name}...')
        model = YOLO(model_name)
        print(f'   ✓ {model_name} 已就绪')
    except Exception as e:
        print(f'   ⚠ {model_name} 下载失败: {e}')
" 2>/dev/null || echo "   ⚠ YOLO预下载失败，将在首次使用时自动下载"

# ============================================
# 6. 安装其他依赖
# ============================================
echo ""
echo "10. 安装相关Python依赖..."

# 检查并安装依赖
pip install insightface onnxruntime-gpu 2>/dev/null || pip install insightface onnxruntime
pip install facexlib 2>/dev/null || true

echo ""
echo "==========================================="
echo "下载完成！模型位置汇总"
echo "==========================================="
echo ""
echo "PuLID-FLUX:     $BASE_DIR/pulid/"
echo "EVA-CLIP:       $BASE_DIR/clip/"
echo "InsightFace:    $BASE_DIR/insightface/"
echo "SAM2:           $BASE_DIR/sam2/"
echo "YOLO:           ~/.cache/ultralytics/ (或自动管理)"
echo ""

# 检查下载状态
echo "下载状态检查:"
echo "-------------------------------------------"

check_file() {
    local name=$1
    local path=$2
    if [ -f "$path" ] || [ -d "$path" ]; then
        echo "✓ $name"
    else
        echo "❌ $name (未找到: $path)"
    fi
}

check_file "PuLID-FLUX" "$BASE_DIR/pulid/pulid_flux_v0.9.1.safetensors"
check_file "InsightFace AntelopeV2" "$BASE_DIR/insightface/models/antelopev2/1k3d68.onnx"
check_file "SAM2" "$BASE_DIR/sam2"
echo "✓ YOLO (自动管理)"

echo ""
echo "==========================================="
echo "下一步操作"
echo "==========================================="
echo "1. 更新 config.yaml 中的模型路径"
echo "2. 测试 PuLID: python -c \"from pulid import PuLIDPipeline\""
echo "3. 测试 SAM2: python -c \"from sam2.build_sam import build_sam2\""
echo "4. 测试 YOLO: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\""
echo ""
