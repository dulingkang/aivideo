#!/bin/bash
# 同步venv到持久化云存储的脚本

set -e

# 配置
VENV_PATH="${1:-/vepfs-dev/shawn/venv/py312}"
CLOUD_PATH="${2:-cloud:/path/to/venv}"

echo "=========================================="
echo "同步venv到云存储"
echo "=========================================="
echo ""
echo "源路径: $VENV_PATH"
echo "目标路径: $CLOUD_PATH"
echo ""

# 检查源路径
if [ ! -d "$VENV_PATH" ]; then
    echo "✗ 源路径不存在: $VENV_PATH"
    exit 1
fi

# 计算源大小
SOURCE_SIZE=$(du -sh "$VENV_PATH" | cut -f1)
echo "源环境大小: $SOURCE_SIZE"
echo ""

# 清理缓存文件（可选）
read -p "是否清理缓存文件？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "清理缓存文件..."
    find "$VENV_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$VENV_PATH" -name "*.pyc" -delete 2>/dev/null || true
    find "$VENV_PATH" -name "*.pyo" -delete 2>/dev/null || true
    echo "✓ 缓存文件已清理"
    echo ""
fi

# 同步
echo "开始同步..."
rsync -av --progress \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.pytest_cache' \
  --exclude='pip' \
  --exclude='setuptools' \
  --exclude='wheel' \
  --exclude='*.dist-info' \
  --exclude='.cache' \
  "$VENV_PATH/" "$CLOUD_PATH/"

echo ""
echo "=========================================="
echo "✓ 同步完成！"
echo "=========================================="
echo ""
echo "提示："
echo "  1. 在目标系统上激活环境: source $CLOUD_PATH/bin/activate"
echo "  2. 如果缺少某些包，运行: pip install -r requirements.txt"
echo ""

