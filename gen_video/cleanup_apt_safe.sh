#!/bin/bash
# APT 安全清理脚本（无需交互）
# 只清理安全的项目，不会删除重要的包

set -e

echo "=========================================="
echo "APT 安全清理脚本"
echo "=========================================="
echo ""

# 检查当前磁盘使用情况
echo "清理前磁盘使用情况:"
df -h / | tail -1
echo ""

TOTAL_FREED=0

# 1. 清理 apt 缓存（安全）
echo "1. 清理 apt 缓存..."
if [ -d "/var/cache/apt" ]; then
    APT_CACHE_SIZE_BEFORE=$(du -sm /var/cache/apt 2>/dev/null | cut -f1 || echo "0")
    echo "  Apt 缓存大小: ${APT_CACHE_SIZE_BEFORE}MB"
    
    apt-get clean 2>/dev/null || sudo apt-get clean 2>/dev/null || true
    apt-get autoclean 2>/dev/null || sudo apt-get autoclean 2>/dev/null || true
    
    APT_CACHE_SIZE_AFTER=$(du -sm /var/cache/apt 2>/dev/null | cut -f1 || echo "0")
    FREED=$((APT_CACHE_SIZE_BEFORE - APT_CACHE_SIZE_AFTER))
    TOTAL_FREED=$((TOTAL_FREED + FREED))
    echo "  ✓ 释放空间: ${FREED}MB"
else
    echo "  Apt 缓存不存在，跳过"
fi
echo ""

# 2. 清理 apt 列表缓存（安全）
echo "2. 清理 apt 列表缓存..."
if [ -d "/var/lib/apt/lists" ]; then
    LISTS_SIZE_BEFORE=$(du -sm /var/lib/apt/lists 2>/dev/null | cut -f1 || echo "0")
    echo "  Apt 列表大小: ${LISTS_SIZE_BEFORE}MB"
    
    # 清理部分列表，保留基本的
    apt-get update 2>/dev/null || sudo apt-get update 2>/dev/null || true
    
    LISTS_SIZE_AFTER=$(du -sm /var/lib/apt/lists 2>/dev/null | cut -f1 || echo "0")
    # 这个通常不会减少太多，因为会重新下载
    echo "  ✓ Apt 列表已更新"
else
    echo "  Apt 列表不存在，跳过"
fi
echo ""

# 3. 清理自动安装但不再需要的依赖（安全，但需要确认）
echo "3. 检查可以自动删除的包..."
AUTOREMOVE_PACKAGES=$(apt-get --dry-run autoremove 2>/dev/null | \
    grep "The following packages will be REMOVED" -A 1000 | \
    grep "^  " | grep -v "The following" | wc -l || echo "0")

if [ "$AUTOREMOVE_PACKAGES" -gt 0 ]; then
    echo "  找到 $AUTOREMOVE_PACKAGES 个可以自动删除的包"
    echo "  这些是自动安装但不再需要的依赖包"
    
    # 显示前10个
    apt-get --dry-run autoremove 2>/dev/null | \
        grep "The following packages will be REMOVED" -A 20 | \
        grep "^  " | head -10 | sed 's/^/    /'
    
    if [ "$AUTOREMOVE_PACKAGES" -lt 20 ]; then
        echo "  执行自动清理..."
        apt-get autoremove -y 2>/dev/null || sudo apt-get autoremove -y 2>/dev/null || true
        echo "  ✓ 自动清理完成"
    else
        echo "  包数量较多，建议手动运行: apt-get autoremove -y"
    fi
else
    echo "  没有需要自动删除的包"
fi
echo ""

# 4. 清理不需要的包配置（rc 状态）- 安全
echo "4. 清理不需要的包配置..."
RC_COUNT=$(dpkg -l | grep -E "^rc" | wc -l)
if [ "$RC_COUNT" -gt 0 ]; then
    echo "  找到 $RC_COUNT 个已卸载但配置保留的包"
    
    if [ "$RC_COUNT" -lt 50 ]; then
        # 获取包名列表
        PACKAGES=$(dpkg -l | grep -E "^rc" | awk '{print $2}')
        
        if [ -n "$PACKAGES" ]; then
            echo "  正在清理..."
            echo "$PACKAGES" | xargs -r dpkg --purge 2>/dev/null || \
            echo "$PACKAGES" | xargs -r sudo dpkg --purge 2>/dev/null || \
            echo "  警告: 需要 sudo 权限"
            echo "  ✓ 已清理 $RC_COUNT 个包配置"
        fi
    else
        echo "  包数量较多，建议手动运行: dpkg --purge \$(dpkg -l | grep '^rc' | awk '{print \$2}')"
    fi
else
    echo "  没有需要清理的包配置"
fi
echo ""

# 5. 清理编译缓存（如果有）
echo "5. 清理编译缓存..."
if [ -d "/tmp" ]; then
    # 清理一些常见的编译临时文件
    find /tmp -name "*.o" -type f -mtime +7 -delete 2>/dev/null || true
    find /tmp -name "*.a" -type f -mtime +7 -delete 2>/dev/null || true
    find /tmp -name "*.so" -type f -mtime +7 -delete 2>/dev/null || true
    echo "  ✓ 编译缓存已清理"
else
    echo "  /tmp 不存在，跳过"
fi
echo ""

# 显示清理后的磁盘使用情况
echo "=========================================="
echo "安全清理完成！"
echo "=========================================="
echo "清理后的磁盘使用情况:"
df -h / | tail -1
echo ""

# 显示可以进一步清理的项目
echo "可以进一步清理的项目（需要手动确认）:"
echo "1. 旧内核: 运行 'dpkg -l | grep linux-image' 查看"
echo "2. 文档和手册页: /usr/share/doc, /usr/share/man"
echo "3. 语言包: /usr/share/locale (保留 en 和 zh)"
echo ""
echo "运行完整清理脚本（包含交互式选项）:"
echo "  bash cleanup_apt.sh"


