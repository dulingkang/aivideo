#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型下载脚本（带重试机制，解决 proxychains 连接问题）
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional
import argparse


def download_model_with_retry(
    model_id: str,
    local_dir: Path,
    max_retries: int = 3,
    retry_delay: int = 5
) -> bool:
    """
    下载模型（带重试机制）
    
    Args:
        model_id: HuggingFace 模型ID
        local_dir: 本地保存目录
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        是否下载成功
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"📥 开始下载: {model_id}")
        print(f"   保存到: {local_dir}")
        
        # 如果目录已存在且有内容，跳过下载
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"   ✓ 模型已存在，跳过下载")
            return True
        
        # 创建目录
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # 重试下载
        for attempt in range(1, max_retries + 1):
            try:
                print(f"   ⏳ 尝试 {attempt}/{max_retries}...")
                
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,  # 支持断点续传
                    max_workers=2,  # 减少并发，避免连接问题
                )
                
                print(f"   ✓ 下载完成")
                return True
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ✗ 尝试 {attempt} 失败: {error_msg[:100]}")
                
                # 检查是否是连接超时错误
                if "timeout" in error_msg.lower() or "socket error" in error_msg.lower():
                    if attempt < max_retries:
                        print(f"   ⏸️  等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                
                # 如果是其他错误，也重试
                if attempt < max_retries:
                    print(f"   ⏸️  等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    # 最后一次尝试失败
                    raise
        
        return False
        
    except Exception as e:
        print(f"   ✗ 下载失败（已重试 {max_retries} 次）: {e}")
        print(f"   💡 提示:")
        print(f"      1. 检查 proxychains4 配置和代理服务")
        print(f"      2. 可以重新运行脚本继续下载（支持断点续传）")
        print(f"      3. 或尝试不使用 proxychains4: python download_models_no_proxy.py")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载多模型组合方案所需模型（带重试机制）")
    parser.add_argument(
        "--model",
        choices=["all", "sd3-turbo", "flux", "flux1", "flux2", "hunyuan-dit", "kolors"],
        default="all",
        help="选择要下载的模型（默认: all）"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/vepfs-dev/shawn/vid/fanren/gen_video/models",
        help="模型保存基础目录"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="最大重试次数（默认: 5）"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=10,
        help="重试延迟（秒，默认: 10）"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 多模型下载脚本（带重试机制）")
    print("=" * 60)
    print(f"基础目录: {base_dir}")
    print(f"最大重试次数: {args.max_retries}")
    print(f"重试延迟: {args.retry_delay} 秒")
    print("=" * 60)
    
    # 模型配置
    models = {
        "sd3-turbo": {
            "id": "calcuis/sd3.5-large-turbo",
            "dir": base_dir / "sd3-turbo",
            "description": "SD3.5 Large Turbo（极速批量生成）"
        },
        "flux1": {
            "id": "black-forest-labs/FLUX.1-dev",
            "dir": base_dir / "flux1-dev",
            "description": "Flux.1（主持人脸+FaceID，实验室/医学场景，约24GB）"
        },
        "flux2": {
            "id": "black-forest-labs/FLUX.1-schnell",
            "dir": base_dir / "flux2-dev",
            "description": "Flux.2（科学背景图、太空/粒子/量子类，冲击力强，约24GB）"
        },
        "flux": {
            "id": None,
            "dir": None,
            "description": "Flux（下载 Flux.1 和 Flux.2）",
            "is_alias": True
        },
        "hunyuan-dit": {
            "id": "Tencent-Hunyuan/HunyuanDiT",
            "dir": base_dir / "hunyuan-dit",
            "description": "Hunyuan-DiT（中文场景，可能需要授权）"
        },
        "kolors": {
            "id": "Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus",
            "dir": base_dir / "kolors",
            "description": "Kolors（真实感场景，快手可图团队开发）"
        }
    }
    
    # 选择要下载的模型
    if args.model == "all":
        models_to_download = [k for k in models.keys() if not models[k].get("is_alias", False)]
    elif args.model == "flux":
        models_to_download = ["flux1", "flux2"]
    else:
        models_to_download = [args.model]
    
    # 下载模型
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for i, model_key in enumerate(models_to_download, 1):
        model_info = models[model_key]
        print(f"\n[{i}/{len(models_to_download)}] {model_info['description']}")
        print("-" * 60)
        
        if model_info.get("is_alias", False):
            print(f"   ℹ️  {model_key} 是别名，已处理")
            skipped_count += 1
            continue
        
        if model_info["id"] is None:
            print(f"   ⚠ {model_key} 模型不可用")
            skipped_count += 1
            continue
        
        if download_model_with_retry(
            model_id=model_info["id"],
            local_dir=model_info["dir"],
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        ):
            success_count += 1
        else:
            fail_count += 1
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 下载总结")
    print("=" * 60)
    print(f"成功: {success_count}/{len(models_to_download)}")
    print(f"失败: {fail_count}/{len(models_to_download)}")
    if skipped_count > 0:
        print(f"跳过: {skipped_count}/{len(models_to_download)}")
    
    if fail_count > 0:
        print("\n⚠️  部分模型下载失败，建议：")
        print("  1. 检查 proxychains4 配置和代理服务")
        print("  2. 重新运行脚本继续下载（支持断点续传）")
        print("  3. 或尝试不使用 proxychains4: python download_models_no_proxy.py")
    
    print("\n✅ 下载完成！")


if __name__ == "__main__":
    main()



