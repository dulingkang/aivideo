#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复模型缺失文件的下载脚本
支持 proxychains4 代理和 mirror
"""

import sys
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def download_kolors_missing_file(use_proxy: bool = False, use_mirror: bool = False):
    """下载 Kolors-base 缺失的文件"""
    print("="*80)
    print("下载 Kolors-base 缺失的文件")
    print("="*80)
    
    # 正确的模型 ID 是 Kwai-Kolors/Kolors，不是 Kwai-Kolors/Kolors-base
    repo_id = "Kwai-Kolors/Kolors"
    filename = "text_encoder/pytorch_model-00002-of-00007.bin"
    local_dir = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/kolors-base")
    
    if use_mirror:
        # 使用镜像
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("使用 HuggingFace 镜像: https://hf-mirror.com")
    
    try:
        print(f"下载: {filename}")
        print(f"保存到: {local_dir}")
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ 下载成功: {downloaded_path}")
        
        # 验证文件
        if Path(downloaded_path).exists():
            size_mb = Path(downloaded_path).stat().st_size / (1024**2)
            print(f"   文件大小: {size_mb:.2f} MB")
            print("✅ 文件验证通过")
            return True
        else:
            print("❌ 文件不存在")
            return False
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_sd3_turbo_standard(use_proxy: bool = False, use_mirror: bool = False):
    """尝试下载标准格式的 SD3.5 Large Turbo"""
    print("="*80)
    print("下载 SD3.5 Large Turbo（标准 diffusers 格式）")
    print("="*80)
    
    # 可能的模型 ID（需要确认）
    model_ids = [
        "stabilityai/stable-diffusion-3.5-large-turbo",
        "stabilityai/sd3.5-large-turbo",
        "calcuis/sd3.5-large-turbo",  # 当前使用的，但可能不是标准格式
    ]
    
    local_dir = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo")
    
    if use_mirror:
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("使用 HuggingFace 镜像: https://hf-mirror.com")
    
    # 备份当前文件
    if local_dir.exists():
        backup_dir = local_dir.parent / f"sd3-turbo-backup-{Path(__file__).stat().st_mtime}"
        print(f"备份当前文件到: {backup_dir}")
        import shutil
        if not backup_dir.exists():
            shutil.move(str(local_dir), str(backup_dir))
    
    for model_id in model_ids:
        try:
            print(f"\n尝试模型: {model_id}")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            # 检查是否有 model_index.json
            if (local_dir / "model_index.json").exists():
                print(f"✅ {model_id} 下载成功，包含 model_index.json")
                return True
            else:
                print(f"⚠️  {model_id} 下载完成，但缺少 model_index.json")
                
        except Exception as e:
            print(f"❌ {model_id} 下载失败: {e}")
            continue
    
    return False

def download_hunyuan_dit_standard(use_proxy: bool = False, use_mirror: bool = False):
    """下载标准格式的 Hunyuan-DiT"""
    print("="*80)
    print("下载 Hunyuan-DiT（标准 diffusers 格式）")
    print("="*80)
    
    repo_id = "Tencent-Hunyuan/HunyuanDiT"
    local_dir = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-dit")
    
    if use_mirror:
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("使用 HuggingFace 镜像: https://hf-mirror.com")
    
    try:
        # 下载 t2i 子目录
        print(f"下载: {repo_id}/t2i")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns="t2i/**",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        # 检查是否有 model_index.json
        t2i_dir = local_dir / "t2i"
        if (t2i_dir / "model_index.json").exists():
            print(f"✅ Hunyuan-DiT 下载成功，包含 model_index.json")
            return True
        else:
            print(f"⚠️  Hunyuan-DiT 下载完成，但缺少 model_index.json")
            return False
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="修复模型缺失文件")
    parser.add_argument("--model", choices=["kolors", "sd3-turbo", "hunyuan-dit", "all"], 
                       default="kolors", help="要修复的模型")
    parser.add_argument("--use-mirror", action="store_true", help="使用 HuggingFace 镜像")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.model in ["kolors", "all"]:
        results["kolors"] = download_kolors_missing_file(use_mirror=args.use_mirror)
    
    if args.model in ["sd3-turbo", "all"]:
        results["sd3-turbo"] = download_sd3_turbo_standard(use_mirror=args.use_mirror)
    
    if args.model in ["hunyuan-dit", "all"]:
        results["hunyuan-dit"] = download_hunyuan_dit_standard(use_mirror=args.use_mirror)
    
    # 总结
    print("\n" + "="*80)
    print("修复总结")
    print("="*80)
    for model, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{model:20s}: {status}")

if __name__ == "__main__":
    main()

