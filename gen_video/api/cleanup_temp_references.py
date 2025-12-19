#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理临时参考图脚本
自动清理超过24小时的临时参考图文件
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# 配置
TEMP_REF_DIR = Path(__file__).parent.parent.parent / "outputs" / "api" / "temp_references"
CLEANUP_AGE_HOURS = 24  # 24小时后清理


def cleanup_temp_references():
    """清理超过指定时间的临时参考图"""
    if not TEMP_REF_DIR.exists():
        print(f"临时参考图目录不存在: {TEMP_REF_DIR}")
        return
    
    print(f"开始清理临时参考图...")
    print(f"目录: {TEMP_REF_DIR}")
    print(f"清理时间阈值: {CLEANUP_AGE_HOURS} 小时")
    print()
    
    cutoff_time = time.time() - (CLEANUP_AGE_HOURS * 3600)
    deleted_count = 0
    total_size = 0
    
    for file_path in TEMP_REF_DIR.iterdir():
        if file_path.is_file():
            # 检查文件修改时间
            file_mtime = file_path.stat().st_mtime
            
            if file_mtime < cutoff_time:
                file_size = file_path.stat().st_size
                file_age_hours = (time.time() - file_mtime) / 3600
                
                try:
                    file_path.unlink()
                    deleted_count += 1
                    total_size += file_size
                    print(f"  ✓ 已删除: {file_path.name} (年龄: {file_age_hours:.1f} 小时, 大小: {file_size / 1024:.1f} KB)")
                except Exception as e:
                    print(f"  ✗ 删除失败: {file_path.name} - {e}")
    
    print()
    print(f"清理完成:")
    print(f"  删除文件数: {deleted_count}")
    print(f"  释放空间: {total_size / 1024 / 1024:.2f} MB")
    
    return deleted_count, total_size


if __name__ == "__main__":
    cleanup_temp_references()

