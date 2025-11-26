#!/bin/bash
# 复制完整的 CosyVoice 模型到正确位置

set -e

SOURCE_DIR="/root/.cache/modelscope/hub/models/iic/CosyVoice2-0___5B"
TARGET_DIR="/vepfs-dev/shawn/vid/fanren/CosyVoice/pretrained_models/CosyVoice2-0.5B"

echo "============================================================"
echo "复制 CosyVoice 模型"
echo "============================================================"
echo ""

# 检查源目录
if [ ! -d "$SOURCE_DIR" ]; then
    echo "✗ 源目录不存在: $SOURCE_DIR"
    exit 1
fi

echo "源目录: $SOURCE_DIR"
echo "目标目录: $TARGET_DIR"
echo ""

# 检查源目录大小
source_size=$(du -sh "$SOURCE_DIR" | cut -f1)
echo "源目录大小: $source_size"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 删除旧的损坏文件
echo ""
echo "删除旧的损坏文件..."
rm -f "$TARGET_DIR"/*.pt 2>/dev/null || true
rm -f "$TARGET_DIR"/*.safetensors 2>/dev/null || true
echo "✓ 已删除旧文件"

# 复制模型文件
echo ""
echo "复制模型文件（这可能需要几分钟）..."
rsync -av --progress "$SOURCE_DIR/" "$TARGET_DIR/" || {
    echo "⚠ rsync 失败，尝试使用 cp..."
    cp -r "$SOURCE_DIR"/* "$TARGET_DIR/"
}

echo ""
echo "验证复制结果..."
target_size=$(du -sh "$TARGET_DIR" | cut -f1)
echo "目标目录大小: $target_size"

# 检查关键文件
key_files=("llm.pt" "flow.pt" "hift.pt" "cosyvoice2.yaml")
all_ok=true
for f in "${key_files[@]}"; do
    if [ -f "$TARGET_DIR/$f" ]; then
        size=$(du -h "$TARGET_DIR/$f" | cut -f1)
        echo "  ✓ $f: $size"
    else
        echo "  ✗ $f: 不存在"
        all_ok=false
    fi
done

if [ "$all_ok" = true ]; then
    echo ""
    echo "============================================================"
    echo "✓ 模型复制完成！"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "⚠ 部分文件缺失，请检查"
    echo "============================================================"
    exit 1
fi

