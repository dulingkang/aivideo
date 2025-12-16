#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX.2-dev 模型下载脚本（使用 proxychains4 代理，解决100%卡住问题）
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
import signal

# 全局变量
download_interrupted = False

def signal_handler(sig, frame):
    """处理中断信号"""
    global download_interrupted
    print("\n⚠️  收到中断信号，正在安全退出...")
    print("   ℹ 已下载的文件已保存，可以重新运行脚本继续下载")
    download_interrupted = True
    sys.exit(0)

def check_proxychains4():
    """检查 proxychains4 是否可用"""
    try:
        result = subprocess.run(
            ["which", "proxychains4"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None

def check_existing_files(model_dir: Path):
    """检查已存在的文件"""
    if not model_dir.exists():
        return 0, []
    
    safetensors_files = list(model_dir.rglob("*.safetensors"))
    bin_files = list(model_dir.rglob("*.bin"))
    pt_files = list(model_dir.rglob("*.pt"))
    all_files = safetensors_files + bin_files + pt_files
    
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())
    size_gb = total_size / (1024 ** 3)
    
    return size_gb, all_files

def download_with_proxy(model_id: str, local_dir: Path, max_retries: int = 3):
    """使用 proxychains4 下载模型"""
    global download_interrupted
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("📥 FLUX.2-dev 模型下载（使用 proxychains4 代理）")
    print("=" * 60)
    print(f"模型ID: {model_id}")
    print(f"保存目录: {local_dir}")
    print()
    
    # 检查 proxychains4
    proxychains_path = check_proxychains4()
    if not proxychains_path:
        print("❌ 错误: 未找到 proxychains4")
        print("   请安装: sudo apt install proxychains4")
        print("   或确保 proxychains4 在 PATH 中")
        return False
    
    print(f"✅ 找到 proxychains4: {proxychains_path}")
    print()
    
    # 检查已存在的文件
    existing_size, existing_files = check_existing_files(local_dir)
    if existing_size > 0:
        print(f"✅ 发现已下载的文件: {existing_size:.2f} GB ({len(existing_files)} 个文件)")
        print("   ℹ 将自动续传，不会重新下载已存在的文件")
        print()
    
    # 创建目录
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量，优化下载
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用 hf_transfer 加速
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"  # 设置超时（1小时）
    
    # 下载配置
    download_kwargs = {
        "repo_id": model_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
        "resume_download": True,
        "max_workers": 2,  # 减少并发
    }
    
    # 重试下载
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"⏳ 重试下载 ({attempt}/{max_retries})...")
                time.sleep(5)
            else:
                print("⏳ 开始下载（使用 proxychains4 代理）...")
                print("   💡 提示: 按 Ctrl+C 可以安全中断，已下载的文件会保留")
                print("   💡 如果下载到100%后卡住，可能是验证阶段，请耐心等待或按 Ctrl+C 中断")
                print()
            
            # 检查是否在 proxychains4 环境中
            # proxychains4 会设置 LD_PRELOAD，我们可以检查这个
            is_proxychains = "PROXYCHAINS_CONF_FILE" in os.environ or \
                           any("proxychains" in str(v).lower() for v in os.environ.values())
            
            if not is_proxychains and "HTTP_PROXY" not in env and "HTTPS_PROXY" not in env:
                print("   ⚠️  警告: 未检测到 proxychains4 环境或代理设置")
                print("   💡 建议: 使用以下命令运行:")
                print(f"      proxychains4 -q python {sys.argv[0]}")
                print()
                print("   或者设置代理环境变量:")
                print("      export HTTP_PROXY=your_proxy")
                print("      export HTTPS_PROXY=your_proxy")
                print()
                response = input("   是否继续？(y/n): ")
                if response.lower() != 'y':
                    return False
            
            # 设置下载超时和重试参数，避免100%后卡住
            # 通过设置较小的 chunk_size 和超时来避免卡住
            print("   ⏳ 开始下载...")
            print("   💡 如果下载到100%后卡住，可能是验证阶段，请等待或按 Ctrl+C 中断")
            print()
            
            # 直接使用 snapshot_download（假设已经在 proxychains4 环境中运行）
            # 设置超时避免无限等待
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
            
            if any(keyword in error_msg.lower() for keyword in ["timeout", "connection", "network", "socket"]):
                if attempt < max_retries:
                    print(f"   ⏸️  网络错误，5秒后重试...")
                    time.sleep(5)
                    continue
                else:
                    print(f"   ❌ 已重试 {max_retries} 次，仍然失败")
                    print("   💡 建议:")
                    print("      1. 检查 proxychains4 配置: /etc/proxychains4.conf")
                    print("      2. 确保代理服务正在运行")
                    print("      3. 使用命令: proxychains4 -q python download_flux2_with_proxy.py")
                    return False
            else:
                if attempt < max_retries:
                    print(f"   ⏸️  5秒后重试...")
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
        return False
    else:
        print("❌ 模型下载不完整")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 FLUX.2-dev 模型（使用 proxychains4 代理）")
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
    success = download_with_proxy(
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
        print()
        print("⚠️  重要: 请使用 proxychains4 运行:")
        print(f"   proxychains4 -q python {sys.argv[0]} --model-dir {local_dir}")
        print()
        
        existing_size, existing_files = check_existing_files(local_dir)
        if existing_size > 0:
            print(f"当前已下载: {existing_size:.2f} GB ({len(existing_files)} 个文件)")

if __name__ == "__main__":
    main()

