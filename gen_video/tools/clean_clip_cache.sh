#!/bin/bash
# 清理可能损坏的 CLIP 模型缓存

CACHE_PATH="/vepfs-dev/shawn/.cache/huggingface/hub/models--openai--clip-vit-large-patch14"

echo "🗑️  清理 CLIP 模型缓存..."
echo "   路径: $CACHE_PATH"

if [ -d "$CACHE_PATH" ]; then
    echo "   删除缓存目录..."
    rm -rf "$CACHE_PATH"
    echo "   ✅ 缓存已清理"
else
    echo "   ℹ️  缓存目录不存在，无需清理"
fi

echo ""
echo "💡 现在可以重新运行下载脚本："
echo "   ./tools/download_clip_model.sh"

