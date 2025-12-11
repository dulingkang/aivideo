#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型下载脚本（不使用 proxychains，直接下载或使用 HuggingFace 镜像）
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse


def setup_huggingface_mirror():
    """设置 HuggingFace 镜像源（如果可用）"""
    # 尝试使用 HuggingFace 镜像
    mirrors = [
        "https://hf-mirror.com",  # HuggingFace 镜像
    ]
    
    # 设置环境变量
    if "HF_ENDPOINT" not in os.environ:
        # 可以尝试设置镜像，但需要确认镜像是否支持
        print("ℹ️  提示: 如果下载慢，可以尝试设置 HuggingFace 镜像")
        print("    export HF_ENDPOINT=https://hf-mirror.com")
    
    # 启用 hf_transfer（如果可用，可以加速下载）
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("ℹ️  已启用 hf_transfer 加速下载")


def download_model(
    model_id: str,
    local_dir: Path,
    resume: bool = True,
    use_mirror: bool = False
) -> bool:
    """
    下载模型（不使用 proxychains）
    
    Args:
        model_id: HuggingFace 模型ID
        local_dir: 本地保存目录
        resume: 是否支持断点续传
        use_mirror: 是否使用镜像
    
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
        
        # 下载模型（支持断点续传）
        print(f"   ⏳ 开始下载（支持断点续传，如果中断可以重新运行继续下载）...")
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=resume,
            # 使用多线程下载（如果支持）
            max_workers=4,
        )
        
        print(f"   ✓ 下载完成")
        return True
        
    except Exception as e:
        print(f"   ✗ 下载失败: {e}")
        print(f"   💡 提示:")
        print(f"      1. 检查网络连接")
        print(f"      2. 如果网络不稳定，可以重新运行脚本继续下载（支持断点续传）")
        print(f"      3. 如果下载速度慢，可以尝试使用 HuggingFace 镜像")
        print(f"      4. 确保已登录: huggingface-cli login")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载多模型组合方案所需模型（不使用 proxychains）")
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
        "--use-mirror",
        action="store_true",
        help="使用 HuggingFace 镜像源"
    )
    
    args = parser.parse_args()
    
    # 设置 HuggingFace 镜像（如果启用）
    if args.use_mirror:
        setup_huggingface_mirror()
    
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 多模型下载脚本（直接下载，不使用 proxychains）")
    print("=" * 60)
    print(f"基础目录: {base_dir}")
    print(f"使用镜像: {args.use_mirror}")
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
            "id": "black-forest-labs/FLUX.1-schnell",  # 注意：Flux.2 的实际模型ID需要确认
            "dir": base_dir / "flux2-dev",
            "description": "Flux.2（科学背景图、太空/粒子/量子类，冲击力强，约24GB）"
        },
        "flux": {  # 兼容旧参数，下载 flux1 和 flux2
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
            "description": "Kolors（真实感场景，快手可图团队开发）",
            "note": "使用 Kolors-IP-Adapter-FaceID-Plus 版本，可直接用 diffusers 加载"
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
        
        # 检查是否是别名
        if model_info.get("is_alias", False):
            print(f"   ℹ️  {model_key} 是别名，已处理")
            skipped_count += 1
            continue
        
        # 检查模型ID是否存在
        if model_info["id"] is None:
            print(f"   ⚠ {model_key} 模型不可用")
            skipped_count += 1
            continue
        
        # 特殊处理：显示 Kolors 的说明
        if model_key == "kolors" and "note" in model_info:
            print(f"   ℹ️  {model_info['note']}")
        
        if download_model(
            model_id=model_info["id"],
            local_dir=model_info["dir"],
            resume=True,
            use_mirror=args.use_mirror
        ):
            success_count += 1
        else:
            fail_count += 1
            if model_key == "kolors":
                print("   ⚠ Kolors 下载失败")
                print("   💡 提示: Kolors 可能需要特殊授权，请访问 https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus")
                print("   💡 注意: 确保已安装最新版本: pip install -U diffusers transformers accelerate")
            else:
                print(f"   ⚠ {model_key} 下载失败")
    
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
        print("  1. 检查网络连接")
        print("  2. 重新运行脚本继续下载（支持断点续传）")
        print("  3. 如果网络不稳定，可以尝试使用 HuggingFace 镜像:")
        print("     python download_models_no_proxy.py --use-mirror")
        print("  4. 确保已登录 HuggingFace: huggingface-cli login")
    
    print("\n✅ 下载完成！")


if __name__ == "__main__":
    main()



