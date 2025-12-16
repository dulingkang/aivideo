#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX.2-dev 模型下载脚本（可靠的断点续传版本）
使用 Python API 确保断点续传正常工作
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
import signal

# 全局变量，用于优雅退出
download_interrupted = False

def signal_handler(sig, frame):
    """处理中断信号，确保文件保存"""
    global download_interrupted
    print("\n⚠️  收到中断信号，正在安全退出...")
    print("   ℹ 已下载的文件已保存，可以重新运行脚本继续下载")
    download_interrupted = True
    sys.exit(0)

def check_existing_files(model_dir: Path):
    """检查已存在的文件"""
    if not model_dir.exists():
        return 0, []
    
    # 查找所有已下载的文件
    safetensors_files = list(model_dir.rglob("*.safetensors"))
    bin_files = list(model_dir.rglob("*.bin"))
    pt_files = list(model_dir.rglob("*.pt"))
    all_files = safetensors_files + bin_files + pt_files
    
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())
    size_gb = total_size / (1024 ** 3)
    
    return size_gb, all_files

def download_with_progress(model_id: str, local_dir: Path, max_retries: int = 3):
    """使用 snapshot_download 下载，支持断点续传"""
    global download_interrupted
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("📥 FLUX.2-dev 模型下载（可靠断点续传版本）")
    print("=" * 60)
    print(f"模型ID: {model_id}")
    print(f"保存目录: {local_dir}")
    print()
    
    # 检查已存在的文件
    existing_size, existing_files = check_existing_files(local_dir)
    if existing_size > 0:
        print(f"✅ 发现已下载的文件: {existing_size:.2f} GB ({len(existing_files)} 个文件)")
        print("   ℹ 将自动续传，不会重新下载已存在的文件")
        print()
    
    # 创建目录
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载配置
    download_kwargs = {
        "repo_id": model_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
        "resume_download": True,  # 关键：启用断点续传
        "max_workers": 2,  # 减少并发，避免连接问题
    }
    
    # 重试下载
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"⏳ 重试下载 ({attempt}/{max_retries})...")
                time.sleep(5)
            else:
                print("⏳ 开始下载...")
                print("   💡 提示: 按 Ctrl+C 可以安全中断，已下载的文件会保留")
                print()
            
            # 开始下载
            snapshot_download(**download_kwargs)
            
            if download_interrupted:
                print("\n⚠️  下载被中断，但已下载的文件已保存")
                return False
            
            print()
            print("✅ 下载完成！")
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  下载被用户中断")
            print("   ℹ 已下载的文件已保存，可以重新运行脚本继续下载")
            return False
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ 下载失败: {error_msg}")
            
            # 检查是否是网络错误
            if any(keyword in error_msg.lower() for keyword in ["timeout", "connection", "network", "socket"]):
                if attempt < max_retries:
                    print(f"   ⏸️  网络错误，{5}秒后重试...")
                    time.sleep(5)
                    continue
                else:
                    print(f"   ❌ 已重试 {max_retries} 次，仍然失败")
                    print("   💡 建议: 检查网络连接后重新运行脚本")
                    return False
            else:
                # 其他错误，也重试一次
                if attempt < max_retries:
                    print(f"   ⏸️  {5}秒后重试...")
                    time.sleep(5)
                    continue
                else:
                    raise
    
    return False

def verify_download(model_dir: Path):
    """验证下载是否完整"""
    print()
    print("=" * 60)
    print("🔍 验证下载完整性")
    print("=" * 60)
    
    # 检查关键文件
    required_files = [
        "model_index.json",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    
    all_exist = True
    for req_file in required_files:
        file_path = model_dir / req_file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 ** 3)
            print(f"✅ {req_file} ({size:.2f} GB)")
        else:
            print(f"❌ {req_file} (缺失)")
            all_exist = False
    
    # 统计总大小
    total_size = sum(
        f.stat().st_size 
        for f in model_dir.rglob("*") 
        if f.is_file()
    ) / (1024 ** 3)
    
    print()
    print(f"📊 总大小: {total_size:.2f} GB")
    
    if all_exist and total_size > 50:
        print("✅ 模型下载完整，可以使用")
        return True
    elif total_size > 50:
        print("⚠️  模型文件较大，但可能缺少部分文件")
        print("   💡 建议: 重新运行下载脚本确保完整性")
        return False
    else:
        print("❌ 模型下载不完整")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 FLUX.2-dev 模型（可靠断点续传版本）")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev",
        help="模型保存目录"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="最大重试次数（默认: 5）"
    )
    
    args = parser.parse_args()
    
    model_id = "black-forest-labs/FLUX.2-dev"
    local_dir = Path(args.model_dir)
    
    # 下载模型
    success = download_with_progress(
        model_id=model_id,
        local_dir=local_dir,
        max_retries=args.max_retries
    )
    
    # 验证下载
    if success:
        verify_download(local_dir)
    else:
        print()
        print("=" * 60)
        print("💡 下载未完成，但已下载的文件已保存")
        print("=" * 60)
        print("重新运行此脚本可以继续下载（自动断点续传）")
        print(f"命令: python {sys.argv[0]} --model-dir {local_dir}")
        print()
        
        # 显示当前进度
        existing_size, existing_files = check_existing_files(local_dir)
        if existing_size > 0:
            print(f"当前已下载: {existing_size:.2f} GB ({len(existing_files)} 个文件)")

if __name__ == "__main__":
    main()

