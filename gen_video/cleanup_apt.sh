#!/bin/bash
# APT 包清理脚本
# 清理不需要的 apt 包、缓存、旧内核等

set -e

echo "=========================================="
echo "APT 包清理脚本"
echo "=========================================="
echo ""

# 检查当前磁盘使用情况
echo "清理前磁盘使用情况:"
df -h / | tail -1
echo ""

# 1. 清理 apt 缓存
echo "1. 清理 apt 缓存..."
if [ -d "/var/cache/apt" ]; then
    APT_CACHE_SIZE=$(du -sh /var/cache/apt 2>/dev/null | cut -f1 || echo "0")
    echo "  Apt 缓存大小: $APT_CACHE_SIZE"
    
    apt-get clean 2>/dev/null || sudo apt-get clean
    apt-get autoclean 2>/dev/null || sudo apt-get autoclean
    
    APT_CACHE_SIZE_AFTER=$(du -sh /var/cache/apt 2>/dev/null | cut -f1 || echo "0")
    echo "  ✓ Apt 缓存已清理: $APT_CACHE_SIZE -> $APT_CACHE_SIZE_AFTER"
else
    echo "  Apt 缓存不存在，跳过"
fi
echo ""

# 2. 清理不需要的包（rc 状态）
echo "2. 清理不需要的包..."
RC_COUNT=$(dpkg -l | grep -E "^rc" | wc -l)
if [ "$RC_COUNT" -gt 0 ]; then
    echo "  找到 $RC_COUNT 个已卸载但配置保留的包"
    echo "  正在清理..."
    
    # 获取包名列表
    PACKAGES=$(dpkg -l | grep -E "^rc" | awk '{print $2}')
    
    if [ -n "$PACKAGES" ]; then
        echo "$PACKAGES" | xargs -r apt-get purge -y 2>/dev/null || \
        echo "$PACKAGES" | xargs -r sudo apt-get purge -y || \
        echo "  警告: 需要 sudo 权限来清理包"
    fi
    
    echo "  ✓ 已清理 $RC_COUNT 个包"
else
    echo "  没有需要清理的包"
fi
echo ""

# 3. 清理自动安装但不再需要的依赖
echo "3. 清理自动安装但不再需要的依赖..."
apt-get autoremove -y 2>/dev/null || sudo apt-get autoremove -y || echo "  警告: 需要 sudo 权限"
echo "  ✓ 自动清理完成"
echo ""

# 4. 清理旧内核（保留当前内核和最新的一个）
echo "4. 清理旧内核..."
CURRENT_KERNEL=$(uname -r | sed 's/-generic//')
echo "  当前内核: $CURRENT_KERNEL"

# 获取所有已安装的内核
INSTALLED_KERNELS=$(dpkg -l | grep linux-image | grep -v "linux-image-generic" | awk '{print $2}' | sort -V)

# 获取最新的内核（除了当前内核）
LATEST_KERNEL=$(echo "$INSTALLED_KERNELS" | tail -1)

# 需要保留的内核
KEEP_KERNELS="$CURRENT_KERNEL $LATEST_KERNEL"
echo "  保留的内核: $KEEP_KERNELS"

# 需要删除的旧内核
OLD_KERNELS=$(dpkg -l | grep linux-image | grep -v "linux-image-generic" | \
    awk '{print $2}' | grep -vE "$(echo $KEEP_KERNELS | tr ' ' '|')" || true)

if [ -n "$OLD_KERNELS" ]; then
    OLD_KERNEL_COUNT=$(echo "$OLD_KERNELS" | wc -l)
    echo "  找到 $OLD_KERNEL_COUNT 个旧内核可以删除:"
    echo "$OLD_KERNELS" | sed 's/^/    - /'
    
    read -p "  是否删除这些旧内核? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$OLD_KERNELS" | xargs -r apt-get purge -y 2>/dev/null || \
        echo "$OLD_KERNELS" | xargs -r sudo apt-get purge -y || \
        echo "  警告: 需要 sudo 权限来删除内核"
        echo "  ✓ 旧内核已删除"
    else
        echo "  跳过删除旧内核"
    fi
else
    echo "  没有需要删除的旧内核"
fi
echo ""

# 5. 清理旧内核头文件
echo "5. 清理旧内核头文件..."
OLD_HEADERS=$(dpkg -l | grep linux-headers | \
    grep -vE "$(echo $KEEP_KERNELS | tr ' ' '|')" | \
    awk '{print $2}' || true)

if [ -n "$OLD_HEADERS" ]; then
    OLD_HEADER_COUNT=$(echo "$OLD_HEADERS" | wc -l)
    echo "  找到 $OLD_HEADER_COUNT 个旧内核头文件可以删除"
    
    read -p "  是否删除这些旧内核头文件? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$OLD_HEADERS" | xargs -r apt-get purge -y 2>/dev/null || \
        echo "$OLD_HEADERS" | xargs -r sudo apt-get purge -y || \
        echo "  警告: 需要 sudo 权限"
        echo "  ✓ 旧内核头文件已删除"
    else
        echo "  跳过删除旧内核头文件"
    fi
else
    echo "  没有需要删除的旧内核头文件"
fi
echo ""

# 6. 清理文档和手册页（可选，谨慎使用）
echo "6. 清理文档和手册页（可选）..."
DOC_SIZE=$(du -sh /usr/share/doc 2>/dev/null | cut -f1 || echo "0")
MAN_SIZE=$(du -sh /usr/share/man 2>/dev/null | cut -f1 || echo "0")

echo "  /usr/share/doc 大小: $DOC_SIZE"
echo "  /usr/share/man 大小: $MAN_SIZE"

read -p "  是否清理文档和手册页? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 清理压缩的文档
    find /usr/share/doc -name "*.gz" -delete 2>/dev/null || true
    find /usr/share/man -name "*.gz" -delete 2>/dev/null || true
    
    # 清理非英语语言包文档（可选）
    # find /usr/share/doc -mindepth 1 -maxdepth 1 ! -name "*en*" -type d -exec rm -rf {} + 2>/dev/null || true
    
    echo "  ✓ 文档和手册页已清理"
else
    echo "  跳过清理文档和手册页"
fi
echo ""

# 7. 清理语言包（保留英语和中文）
echo "7. 清理语言包..."
LOCALE_SIZE=$(du -sh /usr/share/locale 2>/dev/null | cut -f1 || echo "0")
echo "  /usr/share/locale 大小: $LOCALE_SIZE"

read -p "  是否清理不需要的语言包（保留 en 和 zh）? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 保留英语和中文
    find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name "en*" ! -name "zh*" ! -name "locale-archive" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "  ✓ 语言包已清理（保留 en 和 zh）"
else
    echo "  跳过清理语言包"
fi
echo ""

# 显示清理后的磁盘使用情况
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo "清理后的磁盘使用情况:"
df -h / | tail -1
echo ""

# 计算释放的空间
echo "建议："
echo "1. 定期运行 'apt-get autoremove' 清理不需要的依赖"
echo "2. 定期运行 'apt-get autoclean' 清理旧的缓存"
echo "3. 使用 'apt-get clean' 清理所有缓存"


