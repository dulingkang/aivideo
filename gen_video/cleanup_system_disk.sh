#!/bin/bash
# 系统盘清理脚本
# 清理临时文件、缓存等，释放系统盘空间

set -e

echo "=========================================="
echo "系统盘清理脚本"
echo "=========================================="
echo ""

# 检查当前磁盘使用情况
echo "当前磁盘使用情况:"
df -h / | tail -1
echo ""

# 1. 清理 pip 缓存
echo "1. 清理 pip 缓存..."
if [ -d "/root/.cache/pip" ]; then
    PIP_CACHE_SIZE=$(du -sh /root/.cache/pip 2>/dev/null | cut -f1)
    echo "  Pip 缓存大小: $PIP_CACHE_SIZE"
    pip cache purge 2>/dev/null || echo "  警告: pip cache purge 失败"
    echo "  ✓ Pip 缓存已清理"
else
    echo "  Pip 缓存不存在，跳过"
fi
echo ""

# 2. 清理 /tmp 目录（保留最近7天的文件）
echo "2. 清理 /tmp 目录..."
TMP_SIZE_BEFORE=$(du -sh /tmp 2>/dev/null | cut -f1 || echo "0")
echo "  /tmp 当前大小: $TMP_SIZE_BEFORE"

# 清理 /tmp 中7天前的文件
find /tmp -type f -atime +7 -delete 2>/dev/null || true
find /tmp -type d -empty -delete 2>/dev/null || true

# 清理 reservoirpy-temp（如果存在，这是最大的临时目录）
if [ -d "/tmp/reservoirpy-temp" ]; then
    RESERVOIR_SIZE=$(du -sh /tmp/reservoirpy-temp 2>/dev/null | cut -f1 || echo "0")
    echo "  清理 /tmp/reservoirpy-temp (大小: $RESERVOIR_SIZE)..."
    rm -rf /tmp/reservoirpy-temp/* 2>/dev/null || true
    echo "  ✓ reservoirpy-temp 已清理"
fi

# 清理其他临时目录
for dir in /tmp/flaxmodels /tmp/loss-of-plasticity /tmp/cifar10; do
    if [ -d "$dir" ]; then
        DIR_SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "0")
        echo "  清理 $dir (大小: $DIR_SIZE)..."
        rm -rf "$dir"/* 2>/dev/null || true
    fi
done

TMP_SIZE_AFTER=$(du -sh /tmp 2>/dev/null | cut -f1 || echo "0")
echo "  /tmp 清理后大小: $TMP_SIZE_AFTER"
echo ""

# 3. 清理 /root/.cache
echo "3. 清理 /root/.cache..."
if [ -d "/root/.cache" ]; then
    CACHE_SIZE_BEFORE=$(du -sh /root/.cache 2>/dev/null | cut -f1 || echo "0")
    echo "  /root/.cache 当前大小: $CACHE_SIZE_BEFORE"
    
    # 清理各种缓存
    rm -rf /root/.cache/pip/* 2>/dev/null || true
    rm -rf /root/.cache/torch/* 2>/dev/null || true
    rm -rf /root/.cache/huggingface/* 2>/dev/null || true
    
    CACHE_SIZE_AFTER=$(du -sh /root/.cache 2>/dev/null | cut -f1 || echo "0")
    echo "  /root/.cache 清理后大小: $CACHE_SIZE_AFTER"
else
    echo "  /root/.cache 不存在，跳过"
fi
echo ""

# 4. 清理系统日志（保留最近7天）
echo "4. 清理系统日志..."
if command -v journalctl &> /dev/null; then
    JOURNAL_SIZE_BEFORE=$(journalctl --disk-usage 2>/dev/null | awk '{print $7 $8}' || echo "0")
    echo "  日志当前大小: $JOURNAL_SIZE_BEFORE"
    
    # 保留最近7天的日志
    journalctl --vacuum-time=7d 2>/dev/null || echo "  警告: journalctl 清理失败"
    
    JOURNAL_SIZE_AFTER=$(journalctl --disk-usage 2>/dev/null | awk '{print $7 $8}' || echo "0")
    echo "  日志清理后大小: $JOURNAL_SIZE_AFTER"
else
    echo "  journalctl 不可用，跳过"
fi
echo ""

# 5. 清理 /var/log（保留最近7天）
echo "5. 清理 /var/log..."
if [ -d "/var/log" ]; then
    LOG_SIZE_BEFORE=$(du -sh /var/log 2>/dev/null | cut -f1 || echo "0")
    echo "  /var/log 当前大小: $LOG_SIZE_BEFORE"
    
    # 清理7天前的日志文件
    find /var/log -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
    find /var/log -type f -name "*.gz" -mtime +7 -delete 2>/dev/null || true
    
    LOG_SIZE_AFTER=$(du -sh /var/log 2>/dev/null | cut -f1 || echo "0")
    echo "  /var/log 清理后大小: $LOG_SIZE_AFTER"
else
    echo "  /var/log 不存在，跳过"
fi
echo ""

# 6. 清理 apt 缓存（如果存在）
echo "6. 清理 apt 缓存..."
if [ -d "/var/cache/apt" ]; then
    APT_CACHE_SIZE=$(du -sh /var/cache/apt 2>/dev/null | cut -f1 || echo "0")
    echo "  Apt 缓存大小: $APT_CACHE_SIZE"
    apt-get clean 2>/dev/null || echo "  警告: apt-get clean 失败（可能需要sudo）"
    echo "  ✓ Apt 缓存已清理"
else
    echo "  Apt 缓存不存在，跳过"
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
AVAIL_BEFORE=$(df / | tail -1 | awk '{print $4}')
echo "可用空间: $(df -h / | tail -1 | awk '{print $4}')"
echo ""

echo "建议："
echo "1. 如果虚拟环境占用空间大，考虑将其移动到挂载盘"
echo "2. 定期运行此脚本清理系统盘"
echo "3. 安装新包时使用 --no-cache-dir 选项"

