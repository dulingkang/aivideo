#!/bin/bash
# 清理所有位置的 CLIP 模型缓存（包括 /root/.cache）

echo "🗑️  清理所有位置的 CLIP 模型缓存..."

# 定义所有可能的缓存基础路径
CACHE_BASES=(
    "/root/.cache/huggingface"
    "/vepfs-dev/shawn/.cache/huggingface"
    "$HOME/.cache/huggingface"
)

MODEL_NAME="models--openai--clip-vit-large-patch14"

for CACHE_BASE in "${CACHE_BASES[@]}"; do
    if [ ! -d "$CACHE_BASE" ]; then
        continue
    fi
    
    # 检查两种可能的路径结构
    CACHE_PATHS=(
        "$CACHE_BASE/hub/$MODEL_NAME"
        "$CACHE_BASE/$MODEL_NAME"
    )
    
    for CACHE_PATH in "${CACHE_PATHS[@]}"; do
        if [ -d "$CACHE_PATH" ]; then
            echo "   删除: $CACHE_PATH"
            rm -rf "$CACHE_PATH"
            echo "   ✅ 已清理"
        fi
    done
    
    # 也清理 locks
    LOCK_PATHS=(
        "$CACHE_BASE/hub/.locks/$MODEL_NAME"
        "$CACHE_BASE/.locks/$MODEL_NAME"
    )
    
    for LOCK_PATH in "${LOCK_PATHS[@]}"; do
        if [ -d "$LOCK_PATH" ]; then
            echo "   删除锁文件: $LOCK_PATH"
            rm -rf "$LOCK_PATH"
            echo "   ✅ 锁文件已清理"
        fi
    done
done

echo ""
echo "✅ 所有缓存已清理"
echo ""
echo "💡 现在可以重新运行下载脚本："
echo "   proxychains4 python3 tools/download_clip_model.py"

