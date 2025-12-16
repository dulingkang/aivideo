#!/usr/bin/env python3
"""测试脚本路径解析"""
from pathlib import Path
import sys

def test_script_path(script_input):
    """测试脚本路径解析"""
    script_path = Path(script_input)
    cwd = Path.cwd()
    
    print(f"输入: {script_input}")
    print(f"当前目录: {cwd}")
    
    if script_path.is_absolute():
        final_path = script_path
        print(f"绝对路径: {final_path}")
    else:
        # 处理 .. 路径
        if '..' in str(script_path):
            # Path 会自动解析 ..，所以我们直接使用
            candidate = (cwd / script_path).resolve()
            print(f"包含 ..，解析后: {candidate}")
            final_path = candidate
        else:
            # 1. 先尝试当前目录
            candidate = cwd / script_path
            print(f"尝试1（当前目录）: {candidate} (exists: {candidate.exists()})")
            if candidate.exists():
                final_path = candidate.resolve()
            else:
                # 2. 尝试父目录
                candidate = cwd.parent / script_path
                print(f"尝试2（父目录）: {candidate} (exists: {candidate.exists()})")
                if candidate.exists():
                    final_path = candidate.resolve()
                else:
                    final_path = candidate  # 即使不存在也返回
    
    print(f"最终路径: {final_path}")
    print(f"文件存在: {final_path.exists()}")
    print("-" * 60)
    return final_path

# 测试两种方式
test_script_path('lingjie/episode/1.json')
test_script_path('../lingjie/episode/1.json')
