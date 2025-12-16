#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型下载状态和完整性
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
import time

def check_file_integrity(file_path: Path, expected_size: int = None):
    """检查文件完整性"""
    if not file_path.exists():
        return False, "文件不存在"
    
    actual_size = file_path.stat().st_size
    if expected_size and actual_size != expected_size:
        return False, f"文件大小不匹配: 期望 {expected_size}, 实际 {actual_size}"
    
    # 检查文件是否可读
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, 2)  # 移动到文件末尾
            if f.tell() != actual_size:
                return False, "文件可能正在写入中"
    except Exception as e:
        return False, f"文件读取错误: {e}"
    
    return True, f"文件完整 ({actual_size / (1024**3):.2f} GB)"

def check_model_download_status(model_dir: Path, model_id: str = None):
    """检查模型下载状态"""
    print("=" * 60)
    print("🔍 检查模型下载状态")
    print("=" * 60)
    print(f"模型目录: {model_dir}")
    print()
    
    if not model_dir.exists():
        print("❌ 模型目录不存在")
        return False
    
    # 检查所有 safetensors 文件
    safetensors_files = list(model_dir.rglob("*.safetensors"))
    print(f"📦 找到 {len(safetensors_files)} 个 .safetensors 文件:")
    print()
    
    total_size = 0
    all_complete = True
    
    for sf_file in sorted(safetensors_files):
        size = sf_file.stat().st_size
        size_gb = size / (1024 ** 3)
        total_size += size
        
        # 检查文件完整性
        is_complete, msg = check_file_integrity(sf_file)
        status = "✅" if is_complete else "⚠️"
        
        rel_path = sf_file.relative_to(model_dir)
        print(f"{status} {rel_path}")
        print(f"   大小: {size_gb:.2f} GB")
        print(f"   状态: {msg}")
        print()
        
        if not is_complete:
            all_complete = False
    
    print(f"📊 总大小: {total_size / (1024 ** 3):.2f} GB")
    print()
    
    # 检查是否有临时文件
    temp_files = list(model_dir.rglob("*.tmp")) + list(model_dir.rglob("*.part"))
    if temp_files:
        print(f"⚠️  发现 {len(temp_files)} 个临时文件:")
        for tf in temp_files:
            print(f"   - {tf.relative_to(model_dir)}")
        print()
    
    # 检查配置文件
    config_files = ["model_index.json", "config.json"]
    has_config = False
    for cfg in config_files:
        if (model_dir / cfg).exists():
            has_config = True
            print(f"✅ 找到配置文件: {cfg}")
    
    if not has_config:
        print("⚠️  未找到配置文件（model_index.json 或 config.json）")
        print("   这可能是单文件格式模型，或下载未完成")
    
    print()
    
    if all_complete and has_config:
        print("✅ 模型下载完整，可以使用")
        return True
    elif all_complete:
        print("⚠️  文件已下载，但可能缺少配置文件")
        print("   建议: 重新运行下载脚本以获取完整模型")
        return False
    else:
        print("❌ 模型下载不完整")
        print("   建议: 重新运行下载脚本继续下载")
        return False

def force_complete_download(model_dir: Path):
    """强制完成下载（清理临时文件，验证完整性）"""
    print("=" * 60)
    print("🔧 强制完成下载")
    print("=" * 60)
    
    # 查找并清理临时文件
    temp_files = list(model_dir.rglob("*.tmp")) + list(model_dir.rglob("*.part"))
    if temp_files:
        print(f"发现 {len(temp_files)} 个临时文件，正在清理...")
        for tf in temp_files:
            try:
                tf.unlink()
                print(f"   ✓ 已删除: {tf.relative_to(model_dir)}")
            except Exception as e:
                print(f"   ✗ 删除失败: {tf.relative_to(model_dir)} - {e}")
        print()
    else:
        print("✓ 没有临时文件需要清理")
        print()
    
    # 验证所有文件
    safetensors_files = list(model_dir.rglob("*.safetensors"))
    print(f"验证 {len(safetensors_files)} 个文件...")
    
    all_valid = True
    for sf_file in safetensors_files:
        is_complete, msg = check_file_integrity(sf_file)
        if not is_complete:
            all_valid = False
            print(f"   ✗ {sf_file.relative_to(model_dir)}: {msg}")
        else:
            print(f"   ✓ {sf_file.relative_to(model_dir)}")
    
    print()
    if all_valid:
        print("✅ 所有文件验证通过，下载已完成")
    else:
        print("⚠️  部分文件可能有问题，建议重新下载")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检查模型下载状态")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev",
        help="模型目录路径"
    )
    parser.add_argument(
        "--force-complete",
        action="store_true",
        help="强制完成下载（清理临时文件）"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if args.force_complete:
        force_complete_download(model_dir)
    else:
        check_model_download_status(model_dir)

