#!/bin/bash
# 安全停止卡住的下载进程

echo "=========================================="
echo "安全停止卡住的下载进程"
echo "=========================================="
echo ""

MODEL_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev"

# 1. 检查模型是否已完整
echo "1. 检查模型完整性..."
if [ -f "$MODEL_DIR/model_index.json" ]; then
    file_count=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
    total_size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    echo "   ✅ 模型已下载完整！"
    echo "   - 文件数量: $file_count 个 .safetensors 文件"
    echo "   - 总大小: $total_size"
    echo "   - 配置文件: model_index.json 存在"
    echo ""
    echo "   💡 模型已经可以使用了！"
else
    echo "   ⚠️  模型可能不完整"
fi
echo ""

# 2. 查找下载进程
echo "2. 查找下载进程..."
PROCESSES=$(ps aux | grep -E "huggingface-cli.*FLUX.2-dev" | grep -v grep | awk '{print $2}')

if [ -z "$PROCESSES" ]; then
    echo "   ✓ 没有正在运行的下载进程"
    exit 0
else
    echo "   发现以下进程:"
    ps aux | grep -E "huggingface-cli.*FLUX.2-dev" | grep -v grep
    echo ""
    
    for pid in $PROCESSES; do
        # 检查进程运行时间
        runtime=$(ps -o etime= -p $pid 2>/dev/null | tr -d ' ')
        echo "   进程 $pid 已运行: $runtime"
    done
    echo ""
fi

# 3. 检查 .incomplete 文件
echo "3. 检查未完成文件..."
INCOMPLETE_FILES=$(find "$MODEL_DIR/.cache/huggingface/download" -name "*.incomplete" 2>/dev/null)
if [ -n "$INCOMPLETE_FILES" ]; then
    incomplete_count=$(echo "$INCOMPLETE_FILES" | wc -l)
    echo "   发现 $incomplete_count 个 .incomplete 文件:"
    for inc_file in $INCOMPLETE_FILES; do
        size=$(du -h "$inc_file" 2>/dev/null | cut -f1)
        mtime=$(stat -c %y "$inc_file" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
        echo "   - $size (最后修改: $mtime)"
    done
    echo ""
    echo "   💡 这些是临时文件，如果模型已完整，可以安全删除"
else
    echo "   ✓ 没有未完成文件"
fi
echo ""

# 4. 询问是否停止
echo "=========================================="
echo "操作建议"
echo "=========================================="
echo ""
echo "由于模型已经下载完整，建议："
echo ""
echo "1. 停止卡住的下载进程（推荐）"
echo "2. 清理 .incomplete 临时文件（可选）"
echo ""
read -p "是否停止下载进程？(y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "正在停止进程..."
    for pid in $PROCESSES; do
        echo "   停止进程 $pid..."
        kill $pid 2>/dev/null
        sleep 2
        
        # 检查是否还在运行
        if kill -0 $pid 2>/dev/null; then
            echo "   进程仍在运行，强制停止..."
            kill -9 $pid 2>/dev/null
            sleep 1
        fi
        
        if ! kill -0 $pid 2>/dev/null; then
            echo "   ✓ 进程 $pid 已停止"
        else
            echo "   ⚠️  无法停止进程 $pid"
        fi
    done
    echo ""
    echo "✅ 进程已停止"
    echo ""
    
    # 询问是否清理临时文件
    if [ -n "$INCOMPLETE_FILES" ]; then
        read -p "是否清理 .incomplete 临时文件？(y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            echo "正在清理临时文件..."
            for inc_file in $INCOMPLETE_FILES; do
                rm -f "$inc_file"
                echo "   ✓ 已删除: $(basename $inc_file)"
            done
            echo ""
            echo "✅ 临时文件已清理"
        fi
    fi
    
    echo ""
    echo "=========================================="
    echo "✅ 完成"
    echo "=========================================="
    echo ""
    echo "模型位置: $MODEL_DIR"
    echo "模型状态: ✅ 已下载完整，可以使用"
    echo ""
    echo "验证命令:"
    echo "  cd /vepfs-dev/shawn/vid/fanren/gen_video"
    echo "  /vepfs-dev/shawn/venv/py312/bin/python check_download_status.py --model-dir models/flux2-dev"
    echo ""
else
    echo ""
    echo "保留当前进程"
    echo ""
    echo "💡 提示: 如果进程一直卡住，可以稍后运行此脚本停止"
fi

