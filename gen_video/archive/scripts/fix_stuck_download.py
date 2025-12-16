#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复卡住的下载：检查并完成未完成的下载
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi
import hashlib

def get_file_hash(file_path: Path, chunk_size: int = 8192):
    """计算文件的 SHA256 哈希值"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def check_and_fix_incomplete_file(incomplete_path: Path, model_dir: Path):
    """检查并修复未完成的文件"""
    print(f"🔍 检查未完成文件: {incomplete_path.name}")
    
    if not incomplete_path.exists():
        print("   ✓ 文件不存在，可能已处理")
        return True
    
    file_size = incomplete_path.stat().st_size
    size_gb = file_size / (1024 ** 3)
    print(f"   大小: {size_gb:.2f} GB")
    
    # 检查文件是否还在被使用
    try:
        # 尝试以追加模式打开，如果文件正在被写入，这可能会失败
        with open(incomplete_path, 'ab') as f:
            pass
    except Exception as e:
        print(f"   ⚠️  文件可能正在被写入: {e}")
        print("   💡 建议: 等待当前下载进程完成，或先停止下载进程")
        return False
    
    # 检查文件是否完整（通过检查文件大小是否稳定）
    print("   ⏳ 检查文件完整性...")
    time.sleep(2)
    new_size = incomplete_path.stat().st_size
    
    if new_size != file_size:
        print(f"   ⚠️  文件大小仍在变化 ({file_size} -> {new_size})")
        print("   💡 文件可能仍在下载中，建议等待")
        return False
    
    print(f"   ✓ 文件大小稳定: {size_gb:.2f} GB")
    
    # 检查文件是否应该对应某个已存在的文件
    # 通常 .incomplete 文件会在验证后重命名为最终文件名
    # 但如果是卡住了，可能需要手动处理
    
    print("   💡 建议操作:")
    print("      1. 如果下载进程还在运行，等待它完成验证（可能需要几分钟到几十分钟）")
    print("      2. 如果下载进程已经停止，可以:")
    print("         - 重新运行下载脚本（支持断点续传）")
    print("         - 或者手动检查文件完整性")
    
    return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="修复卡住的下载")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev",
        help="模型目录路径"
    )
    parser.add_argument(
        "--force-check",
        action="store_true",
        help="强制检查所有未完成文件"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    cache_dir = model_dir / ".cache" / "huggingface" / "download"
    
    print("=" * 60)
    print("🔧 修复卡住的下载")
    print("=" * 60)
    print(f"模型目录: {model_dir}")
    print()
    
    if not cache_dir.exists():
        print("✓ 没有缓存目录，下载可能已完成")
        return
    
    # 查找所有 .incomplete 文件
    incomplete_files = list(cache_dir.glob("*.incomplete"))
    
    if not incomplete_files:
        print("✓ 没有未完成的文件")
        return
    
    print(f"发现 {len(incomplete_files)} 个未完成的文件:")
    print()
    
    for inc_file in incomplete_files:
        check_and_fix_incomplete_file(inc_file, model_dir)
        print()
    
    print("=" * 60)
    print("💡 解决方案:")
    print("=" * 60)
    print("1. 如果下载进程还在运行:")
    print("   - 等待验证完成（大文件验证可能需要很长时间）")
    print("   - 检查进程是否还在活动: ps aux | grep huggingface")
    print()
    print("2. 如果下载进程已停止:")
    print("   - 重新运行下载脚本（支持断点续传）")
    print("   - 命令: bash download_model.sh")
    print("   或: python download_models.py --model flux2")
    print()
    print("3. 如果确定文件已下载完成但卡在验证:")
    print("   - 可以尝试停止下载进程，然后重新运行")
    print("   - huggingface-cli 会自动检测已下载的文件")
    print()
    print("4. 检查模型是否可用:")
    print("   - 运行: python check_download_status.py --model-dir models/flux2-dev")

if __name__ == "__main__":
    import time
    main()

