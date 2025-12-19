#!/bin/bash
# 清理损坏的 CLIP 模型缓存

model_id="openai/clip-vit-large-patch14"
cache_name="models--${model_id//\//--}"

echo "🗑️  清理损坏的 CLIP 模型缓存: $cache_name"
echo ""

# 所有可能的缓存位置
cache_locations=(
    "/vepfs-dev/shawn/.cache/huggingface"
    "$HOME/.cache/huggingface"
    "/root/.cache/huggingface"
)

cleaned=0

for cache_base in "${cache_locations[@]}"; do
    if [ ! -d "$cache_base" ]; then
        continue
    fi
    
    # 检查两种可能的路径结构
    for subpath in "hub" ""; do
        if [ -n "$subpath" ]; then
            cache_path="$cache_base/$subpath/$cache_name"
        else
            cache_path="$cache_base/$cache_name"
        fi
        
        if [ -d "$cache_path" ]; then
            echo "🗑️  删除: $cache_path"
            rm -rf "$cache_path" 2>/dev/null && echo "   ✅ 已清理" || echo "   ⚠️  清理失败"
            cleaned=1
        fi
    done
    
    # 清理 locks
    for lock_subpath in "hub/.locks" ".locks"; do
        lock_path="$cache_base/$lock_subpath/$cache_name"
        if [ -d "$lock_path" ]; then
            echo "🗑️  删除锁文件: $lock_path"
            rm -rf "$lock_path" 2>/dev/null && echo "   ✅ 已清理" || echo "   ⚠️  清理失败"
        fi
    done
done

if [ $cleaned -eq 0 ]; then
    echo "ℹ️  未找到缓存目录"
else
    echo ""
    echo "✅ 清理完成！现在可以重新下载了。"
fi

