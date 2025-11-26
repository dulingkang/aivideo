#!/bin/bash
# 将虚拟环境移动到挂载盘
# 释放系统盘空间

set -e

VENV_PATH="/vepfs-dev/shawn/venv/py312"
MOUNT_PATH="/vepfs-dev/shawn"
NEW_VENV_PATH="${MOUNT_PATH}/venv/py312"

echo "=========================================="
echo "虚拟环境迁移脚本"
echo "=========================================="
echo ""

# 检查当前虚拟环境
if [ ! -d "$VENV_PATH" ]; then
    echo "错误: 虚拟环境不存在: $VENV_PATH"
    exit 1
fi

echo "当前虚拟环境: $VENV_PATH"
VENV_SIZE=$(du -sh "$VENV_PATH" 2>/dev/null | cut -f1)
echo "虚拟环境大小: $VENV_SIZE"
echo ""

# 检查挂载盘空间
echo "挂载盘空间:"
df -h "$MOUNT_PATH" | tail -1
echo ""

# 确认操作
read -p "是否要将虚拟环境移动到挂载盘? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消操作"
    exit 1
fi

# 创建目标目录
echo "创建目标目录..."
mkdir -p "$(dirname "$NEW_VENV_PATH")"

# 移动虚拟环境
echo "移动虚拟环境..."
echo "  从: $VENV_PATH"
echo "  到: $NEW_VENV_PATH"

# 使用 rsync 复制（更安全）
if command -v rsync &> /dev/null; then
    echo "  使用 rsync 复制..."
    rsync -av --progress "$VENV_PATH/" "$NEW_VENV_PATH/"
    echo "  ✓ 复制完成"
    
    # 验证复制
    if [ -d "$NEW_VENV_PATH/bin" ]; then
        echo "  ✓ 验证成功"
        
        # 备份原虚拟环境
        echo "  备份原虚拟环境..."
        mv "$VENV_PATH" "${VENV_PATH}.backup"
        echo "  ✓ 原虚拟环境已备份到: ${VENV_PATH}.backup"
        
        # 创建符号链接
        echo "  创建符号链接..."
        ln -s "$NEW_VENV_PATH" "$VENV_PATH"
        echo "  ✓ 符号链接已创建"
        
        # 测试虚拟环境
        echo "  测试虚拟环境..."
        source "$VENV_PATH/bin/activate"
        python --version
        deactivate
        echo "  ✓ 虚拟环境工作正常"
        
        # 删除备份（可选）
        read -p "是否删除原虚拟环境备份? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "${VENV_PATH}.backup"
            echo "  ✓ 备份已删除"
        else
            echo "  备份保留在: ${VENV_PATH}.backup"
        fi
    else
        echo "  ✗ 验证失败，保留原虚拟环境"
        exit 1
    fi
else
    echo "  使用 cp 复制..."
    cp -r "$VENV_PATH" "$NEW_VENV_PATH"
    echo "  ✓ 复制完成"
    
    # 备份并创建符号链接（同上）
    mv "$VENV_PATH" "${VENV_PATH}.backup"
    ln -s "$NEW_VENV_PATH" "$VENV_PATH"
    echo "  ✓ 迁移完成"
fi

echo ""
echo "=========================================="
echo "迁移完成！"
echo "=========================================="
echo "新的虚拟环境路径: $NEW_VENV_PATH"
echo "符号链接: $VENV_PATH -> $NEW_VENV_PATH"
echo ""
echo "使用方法:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "清理后的系统盘使用情况:"
df -h / | tail -1


