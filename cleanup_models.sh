#!/bin/bash
# 模型清理脚本 - 删除不需要的模型文件以节省存储空间

set -e

MODELS_DIR="gen_video/models"

echo "=========================================="
echo "模型清理脚本"
echo "=========================================="
echo ""
echo "此脚本将删除不需要的模型文件以节省存储空间"
echo ""

# 确认
read -p "是否继续？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

echo ""
echo "开始清理..."
echo ""

# 1. 删除未使用的Whisper模型
echo "1. 删除未使用的Whisper模型..."
if [ -d "${MODELS_DIR}/faster-whisper-large-v3" ]; then
    rm -rf "${MODELS_DIR}/faster-whisper-large-v3/"
    echo "  ✓ 已删除 faster-whisper-large-v3/ (~3GB)"
else
    echo "  - faster-whisper-large-v3/ 不存在，跳过"
fi

if [ -d "${MODELS_DIR}/faster-whisper-small" ]; then
    rm -rf "${MODELS_DIR}/faster-whisper-small/"
    echo "  ✓ 已删除 faster-whisper-small/ (~200MB)"
else
    echo "  - faster-whisper-small/ 不存在，跳过"
fi

# 2. 删除未使用的Real-ESRGAN模型（保留x2plus）
echo ""
echo "2. 删除未使用的Real-ESRGAN模型..."
REALESRGAN_DIR="${MODELS_DIR}/realesrgan"
if [ -f "${REALESRGAN_DIR}/RealESRGAN_x4plus_anime_6B.pth" ]; then
    rm -f "${REALESRGAN_DIR}/RealESRGAN_x4plus_anime_6B.pth"
    echo "  ✓ 已删除 RealESRGAN_x4plus_anime_6B.pth (~67MB)"
fi
if [ -f "${REALESRGAN_DIR}/RealESRGAN_x4plus.pth" ]; then
    rm -f "${REALESRGAN_DIR}/RealESRGAN_x4plus.pth"
    echo "  ✓ 已删除 RealESRGAN_x4plus.pth (~64MB)"
fi
if [ -f "${REALESRGAN_DIR}/realesr-general-wdn-x4v3.pth" ]; then
    echo "  - 保留 realesr-general-wdn-x4v3.pth（如需删除请手动）"
fi

# 3. 删除LoRA训练checkpoint和日志（只保留checkpoint-1000）
echo ""
echo "3. 删除LoRA训练文件（保留checkpoint-1000）..."
LORA_DIR="${MODELS_DIR}/lora"
if [ -d "${LORA_DIR}" ]; then
    # 删除除了checkpoint-1000之外的所有checkpoint目录
    find "${LORA_DIR}" -type d -name "checkpoint-*" ! -name "checkpoint-1000" -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ 已删除 LoRA checkpoint 目录（保留 checkpoint-1000）"
    
    # 删除logs目录
    find "${LORA_DIR}" -type d -name "logs" -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ 已删除 LoRA logs 目录"
else
    echo "  - lora/ 目录不存在，跳过"
fi

# 4. 删除.cache目录（HuggingFace等库的缓存，可以重新下载）
echo ""
echo "4. 删除.cache目录..."
CACHE_DIRS=$(find "${MODELS_DIR}" -type d -name ".cache" 2>/dev/null)
if [ -n "$CACHE_DIRS" ]; then
    # 显示找到的.cache目录和大小
    echo "  找到以下 .cache 目录："
    find "${MODELS_DIR}" -type d -name ".cache" -exec du -sh {} \; 2>/dev/null | sed 's/^/    /'
    find "${MODELS_DIR}" -type d -name ".cache" -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ 已删除 .cache 目录（主要是 ip-adapter/.cache 约3.4GB）"
    echo "    注意：.cache 是 HuggingFace 等库的缓存，删除后下次使用时会重新下载"
else
    echo "  - 未找到 .cache 目录，跳过"
fi

# 5. 删除示例和文档文件
echo ""
echo "5. 删除示例和文档文件..."
find "${MODELS_DIR}" -name "*.png" -type f -delete 2>/dev/null || true
find "${MODELS_DIR}" -name "*.gif" -type f -delete 2>/dev/null || true
find "${MODELS_DIR}" -name "README.md" -type f -delete 2>/dev/null || true
find "${MODELS_DIR}" -name "LICENSE.md" -type f -delete 2>/dev/null || true
find "${MODELS_DIR}" -name "comparison.png" -type f -delete 2>/dev/null || true
echo "  ✓ 已删除示例和文档文件"

# 6. 删除未使用的对齐和VAD模型（可选）
if [ "$1" == "--remove-optional" ]; then
    echo ""
    echo "6. 删除可选模型（对齐和VAD）..."
    if [ -d "${MODELS_DIR}/whisperx-align-zh" ]; then
        rm -rf "${MODELS_DIR}/whisperx-align-zh/"
        echo "  ✓ 已删除 whisperx-align-zh/ (~200MB)"
    fi
    if [ -d "${MODELS_DIR}/pyannote-segmentation" ]; then
        rm -rf "${MODELS_DIR}/pyannote-segmentation/"
        echo "  ✓ 已删除 pyannote-segmentation/ (~100MB)"
    fi
fi

# 7. 提示删除特定领域LoRA
echo ""
echo "7. 特定领域LoRA..."
if [ -d "${LORA_DIR}/hanli" ]; then
    echo "  ⚠ 发现特定领域LoRA: hanli/"
    echo "    建议："
    echo "      - 如果不再使用，可以删除: rm -rf ${LORA_DIR}/hanli/"
    echo "      - 或移动到示例目录: mkdir -p examples/lora && mv ${LORA_DIR}/hanli examples/lora/"
fi

# 8. 提示删除IP-Adapter（如果不用）
echo ""
echo "8. IP-Adapter模型..."
if [ -d "${MODELS_DIR}/ip-adapter" ]; then
    echo "  ⚠ 发现 SDXL IP-Adapter: ip-adapter/"
    echo "    如果config.yaml中不使用SDXL的IP-Adapter（只用InstantID的），可以删除:"
    echo "      rm -rf ${MODELS_DIR}/ip-adapter/"
fi

# 9. 提示CosyVoice模型（必需，不能删除）
echo ""
echo "9. CosyVoice模型..."
COSYVOICE_DIR="CosyVoice/pretrained_models"
if [ -d "$COSYVOICE_DIR" ]; then
    COSYVOICE_SIZE=$(du -sh "$COSYVOICE_DIR" 2>/dev/null | cut -f1)
    echo "  ✓ CosyVoice 模型目录: $COSYVOICE_DIR (${COSYVOICE_SIZE})"
    echo "    注意：CosyVoice 是 TTS 核心模型，不能删除"
    echo "    必需模型：CosyVoice2-0.5B（约4.8GB）"
    echo "    如果不需要其他版本，可以删除："
    echo "      - CosyVoice-ttsfrd/（如果不用）"
    echo "      - 其他版本的 CosyVoice 模型（如果不用）"
else
    echo "  - CosyVoice 模型目录不存在，跳过"
fi

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo ""
echo "预计节省空间：~9-12GB（根据删除的内容，包括.cache约3.4GB）"
echo ""
echo "建议下一步："
echo "  1. 运行验证脚本: python gen_video/check_models_and_modules.py"
echo "  2. 测试核心功能是否正常"
echo "  3. 检查 config.yaml 中的配置是否与保留的模型一致"
echo ""

