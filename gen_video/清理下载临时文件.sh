#!/bin/bash
# 清理下载过程中的临时文件（可选，不影响已下载的模型）

MODEL_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev"
CACHE_DIR="$MODEL_DIR/.cache/huggingface/download"

echo "=========================================="
echo "清理下载临时文件"
echo "=========================================="
echo ""

# 检查模型是否完整
if [ ! -f "$MODEL_DIR/model_index.json" ]; then
    echo "❌ 警告: 未找到 model_index.json，模型可能不完整"
    echo "   不建议清理临时文件"
    exit 1
fi

echo "✅ 模型已确认完整（有 model_index.json）"
echo ""

# 检查临时文件
incomplete_files=$(find "$CACHE_DIR" -name "*.incomplete" 2>/dev/null)
lock_files=$(find "$CACHE_DIR" -name "*.lock" 2>/dev/null)

if [ -z "$incomplete_files" ] && [ -z "$lock_files" ]; then
    echo "✓ 没有临时文件需要清理"
    exit 0
fi

# 显示临时文件
if [ -n "$incomplete_files" ]; then
    incomplete_count=$(echo "$incomplete_files" | wc -l)
    incomplete_size=$(du -sh "$CACHE_DIR"/*.incomplete 2>/dev/null | awk '{sum+=$1} END {print sum}' || echo "0")
    echo "发现 $incomplete_count 个 .incomplete 文件"
    for inc_file in $incomplete_files; do
        size=$(du -h "$inc_file" 2>/dev/null | cut -f1)
        echo "  - $size: $(basename $inc_file)"
    done
    echo ""
fi

if [ -n "$lock_files" ]; then
    lock_count=$(echo "$lock_files" | wc -l)
    echo "发现 $lock_count 个 .lock 文件"
    echo ""
fi

# 询问是否清理
read -p "是否清理这些临时文件？(y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "正在清理..."
    
    if [ -n "$incomplete_files" ]; then
        for inc_file in $incomplete_files; do
            rm -f "$inc_file"
            echo "  ✓ 已删除: $(basename $inc_file)"
        done
    fi
    
    if [ -n "$lock_files" ]; then
        for lock_file in $lock_files; do
            rm -f "$lock_file"
        done
        echo "  ✓ 已删除 $lock_count 个锁定文件"
    fi
    
    echo ""
    echo "✅ 临时文件已清理"
    echo ""
    echo "模型目录大小:"
    du -sh "$MODEL_DIR"
else
    echo ""
    echo "保留临时文件"
fi

