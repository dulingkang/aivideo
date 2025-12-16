#!/bin/bash
# 确保断点续传正常工作的辅助脚本

MODEL_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev"
VENV_PYTHON="/vepfs-dev/shawn/venv/py312/bin/python"

echo "=========================================="
echo "检查断点续传配置"
echo "=========================================="
echo ""

# 1. 检查已下载的文件
echo "1. 检查已下载的文件..."
if [ -d "$MODEL_DIR" ]; then
    existing_size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    file_count=$(find "$MODEL_DIR" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | wc -l)
    echo "   ✓ 已下载: $existing_size ($file_count 个文件)"
    
    if [ "$file_count" -gt 0 ]; then
        echo "   ℹ 这些文件将被保留，不会重新下载"
    fi
else
    echo "   ℹ 模型目录不存在，将从头开始下载"
fi
echo ""

# 2. 检查缓存目录
echo "2. 检查缓存目录..."
CACHE_DIR="$MODEL_DIR/.cache/huggingface/download"
if [ -d "$CACHE_DIR" ]; then
    incomplete_count=$(find "$CACHE_DIR" -name "*.incomplete" 2>/dev/null | wc -l)
    lock_count=$(find "$CACHE_DIR" -name "*.lock" 2>/dev/null | wc -l)
    echo "   ✓ 缓存目录存在"
    echo "   - 未完成文件: $incomplete_count"
    echo "   - 锁定文件: $lock_count"
    
    if [ "$lock_count" -gt 0 ]; then
        echo "   ⚠️  发现锁定文件，可能有其他进程正在下载"
        echo "   建议: 检查是否有其他下载进程运行"
        ps aux | grep -E "(huggingface|download)" | grep -v grep
    fi
else
    echo "   ℹ 缓存目录不存在（首次下载）"
fi
echo ""

# 3. 检查 Python 环境
echo "3. 检查 Python 环境..."
if [ -f "$VENV_PYTHON" ]; then
    echo "   ✓ Python 路径: $VENV_PYTHON"
    
    # 检查 huggingface_hub
    if $VENV_PYTHON -c "import huggingface_hub" 2>/dev/null; then
        version=$($VENV_PYTHON -c "from huggingface_hub import __version__; print(__version__)" 2>/dev/null)
        echo "   ✓ huggingface_hub 已安装 (版本: $version)"
    else
        echo "   ❌ huggingface_hub 未安装"
        echo "   安装命令: $VENV_PYTHON -m pip install huggingface_hub"
        exit 1
    fi
else
    echo "   ❌ Python 路径不存在: $VENV_PYTHON"
    exit 1
fi
echo ""

# 4. 检查磁盘空间
echo "4. 检查磁盘空间..."
available=$(df -h "$MODEL_DIR" 2>/dev/null | tail -1 | awk '{print $4}')
echo "   ✓ 可用空间: $available"
echo ""

# 5. 提供下载建议
echo "=========================================="
echo "下载建议"
echo "=========================================="
echo ""
echo "推荐使用可靠的 Python 下载脚本:"
echo ""
echo "  $VENV_PYTHON download_flux2_robust.py"
echo ""
echo "或者使用现有的下载脚本:"
echo ""
echo "  $VENV_PYTHON download_models.py --model flux2"
echo ""
echo "如果下载中断，重新运行相同命令即可继续下载"
echo "（已下载的文件会被保留，不会重新下载）"
echo ""

