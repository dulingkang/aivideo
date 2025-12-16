#!/bin/bash
# FLUX.2-dev 模型下载脚本（使用 proxychains4，解决100%卡住问题）

set -e

MODEL_DIR="/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev"
VENV_PYTHON="/vepfs-dev/shawn/venv/py312/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "FLUX.2-dev 模型下载（使用 proxychains4）"
echo "=========================================="
echo ""

# 1. 检查 proxychains4
if ! command -v proxychains4 &> /dev/null; then
    echo "❌ 错误: 未找到 proxychains4"
    echo "   请安装: sudo apt install proxychains4"
    exit 1
fi
echo "✅ proxychains4 已安装"
echo ""

# 2. 检查 Python 环境
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ 错误: Python 路径不存在: $VENV_PYTHON"
    exit 1
fi
echo "✅ Python 环境: $VENV_PYTHON"
echo ""

# 3. 检查已下载的文件
if [ -d "$MODEL_DIR" ]; then
    existing_size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
    file_count=$(find "$MODEL_DIR" -name "*.safetensors" 2>/dev/null | wc -l)
    echo "📦 已下载: $existing_size ($file_count 个文件)"
    echo "   ℹ 将自动续传，不会重新下载已存在的文件"
    echo ""
fi

# 4. 停止可能正在运行的下载进程
echo "检查是否有正在运行的下载进程..."
FLUX_PROCESSES=$(ps aux | grep -E "huggingface.*FLUX.2-dev" | grep -v grep | awk '{print $2}' || true)
if [ -n "$FLUX_PROCESSES" ]; then
    echo "   发现正在运行的下载进程，正在停止..."
    for pid in $FLUX_PROCESSES; do
        kill $pid 2>/dev/null || true
        sleep 2
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi
    done
    echo "   ✓ 已停止旧进程"
    sleep 2
else
    echo "   ✓ 没有正在运行的下载进程"
fi
echo ""

# 5. 清理锁定文件（可选）
echo "清理锁定文件..."
LOCK_FILES=$(find "$MODEL_DIR/.cache/huggingface/download" -name "*.lock" 2>/dev/null | wc -l)
if [ "$LOCK_FILES" -gt 0 ]; then
    echo "   发现 $LOCK_FILES 个锁定文件，正在清理..."
    find "$MODEL_DIR/.cache/huggingface/download" -name "*.lock" -delete 2>/dev/null || true
    echo "   ✓ 锁定文件已清理"
else
    echo "   ✓ 没有锁定文件"
fi
echo ""

# 6. 开始下载
echo "=========================================="
echo "开始下载"
echo "=========================================="
echo ""
echo "使用 proxychains4 运行 Python 下载脚本..."
echo ""

# 切换到脚本目录
cd "$SCRIPT_DIR"

# 使用 proxychains4 运行 Python 脚本
# -q 参数：静默模式，减少输出
proxychains4 -q "$VENV_PYTHON" download_flux2_with_proxy.py --model-dir "$MODEL_DIR" || {
    exit_code=$?
    echo ""
    echo "⚠️  下载过程中出现错误（退出码: $exit_code）"
    echo ""
    echo "💡 提示:"
    echo "   1. 已下载的文件已保存，可以重新运行此脚本继续下载"
    echo "   2. 如果下载到100%后卡住，可以按 Ctrl+C 中断"
    echo "   3. 模型的主要文件可能已经下载完成，可以检查模型是否可用"
    echo ""
    echo "检查模型状态:"
    echo "   $VENV_PYTHON check_download_status.py --model-dir $MODEL_DIR"
    exit $exit_code
}

echo ""
echo "=========================================="
echo "✅ 下载完成！"
echo "=========================================="
echo ""
echo "验证模型完整性..."
"$VENV_PYTHON" check_download_status.py --model-dir "$MODEL_DIR" || true
echo ""
echo "模型位置: $MODEL_DIR"

