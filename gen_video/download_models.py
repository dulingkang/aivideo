#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型下载脚本（Python版本）
支持使用 proxychains4 和虚拟环境
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse


def download_model(
    model_id: str,
    local_dir: Path,
    resume: bool = True,
    use_proxy: bool = False,
    max_retries: int = 3,
    retry_delay: int = 10
) -> bool:
    """
    下载模型
    
    Args:
        model_id: HuggingFace 模型ID
        local_dir: 本地保存目录
        resume: 是否支持断点续传
        use_proxy: 是否使用代理（通过环境变量设置）
    
    Returns:
        是否下载成功
    """
    try:
        from huggingface_hub import snapshot_download
        
        print(f"📥 开始下载: {model_id}")
        print(f"   保存到: {local_dir}")
        
        # 检查模型是否真的完整（不只是目录存在）
        if local_dir.exists():
            # 检查是否有权重文件（.safetensors, .bin, .pt）
            weight_files = list(local_dir.rglob("*.safetensors")) + \
                         list(local_dir.rglob("*.bin")) + \
                         list(local_dir.rglob("*.pt"))
            
            # 检查是否有 model_index.json 或 config.json
            has_config = (local_dir / "model_index.json").exists() or \
                        any(local_dir.rglob("config.json"))
            
            # 检查是否有大文件（至少 > 100MB）
            large_files = [f for f in local_dir.rglob("*") 
                          if f.is_file() and f.stat().st_size > 100 * 1024 * 1024]
            
            # 如果有权重文件或大文件，且大小合理，认为已下载
            if (weight_files or large_files) and has_config:
                total_size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
                size_gb = total_size / (1024 * 1024 * 1024)
                if size_gb > 0.5:  # 至少 500MB
                    print(f"   ✓ 模型已存在（{size_gb:.2f} GB），跳过下载")
                    return True
                else:
                    print(f"   ⚠️  模型目录存在但文件过小（{size_gb:.2f} GB），重新下载...")
            elif weight_files or large_files:
                # 有权重文件但缺少配置文件，可能是部分下载
                total_size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
                size_gb = total_size / (1024 * 1024 * 1024)
                if size_gb > 1.0:  # 至少 1GB
                    print(f"   ⚠️  模型部分存在（{size_gb:.2f} GB），但缺少配置文件，继续下载...")
                else:
                    print(f"   ⚠️  模型目录存在但文件不完整（{size_gb:.2f} GB），重新下载...")
            else:
                # 只有元数据文件，没有实际模型
                print(f"   ⚠️  目录存在但无模型文件，重新下载...")
        
        # 创建目录
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # 带重试的下载
        import time
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"   ⏳ 重试 {attempt}/{max_retries}...")
                    time.sleep(retry_delay)
                else:
                    print(f"   ⏳ 开始下载（支持断点续传）...")
                
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=resume,
                    max_workers=2,  # 减少并发，避免连接问题
                )
                
                print(f"   ✓ 下载完成")
                return True
                
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "socket error" in error_msg.lower():
                    if attempt < max_retries:
                        print(f"   ⚠️  连接超时，{retry_delay}秒后重试...")
                        continue
                elif attempt < max_retries:
                    print(f"   ⚠️  下载失败，{retry_delay}秒后重试: {error_msg[:80]}")
                    continue
                else:
                    # 最后一次尝试失败
                    raise
        
        print(f"   ✓ 下载完成")
        return True
        
    except Exception as e:
        print(f"   ✗ 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载多模型组合方案所需模型")
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="使用 proxychains4（通过环境变量设置）"
    )
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
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 多模型下载脚本")
    print("=" * 60)
    print(f"基础目录: {base_dir}")
    print(f"使用代理: {args.use_proxy}")
    print("=" * 60)
    
    # 模型配置
    models = {
        "sd3-turbo": {
            "id": "stabilityai/stable-diffusion-3.5-large-turbo",  # 使用标准 diffusers 格式
            "dir": base_dir / "sd3-turbo",
            "description": "SD3.5 Large Turbo（极速批量生成，标准 diffusers 格式）"
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
            "id": "Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus",  # 使用 IP-Adapter FaceID Plus 版本
            "dir": base_dir / "kolors",
            "description": "Kolors（真实感场景，快手可图团队开发，真人质感强，中文 prompt 理解优秀）",
            "note": "使用 Kolors-IP-Adapter-FaceID-Plus 版本，可直接用 diffusers 加载"
        }
    }
    
    # 选择要下载的模型
    if args.model == "all":
        models_to_download = [k for k in models.keys() if not models[k].get("is_alias", False)]
    elif args.model == "flux":
        # flux 是别名，下载 flux1 和 flux2
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
            if "alternative" in model_info:
                print(f"   💡 替代方案: 可以使用 {model_info['alternative']} 或其他真实感模型")
                print(f"   💡 建议: 使用 SDXL 或 Flux 配合真实感 LoRA 实现类似效果")
            skipped_count += 1
            continue
        
        # 特殊处理：显示 Kolors 的说明
        if model_key == "kolors" and "note" in model_info:
            print(f"   ℹ️  {model_info['note']}")
        
        if download_model(
            model_id=model_info["id"],
            local_dir=model_info["dir"],
            resume=True,
            use_proxy=args.use_proxy,
            max_retries=5,  # 默认重试5次
            retry_delay=10  # 默认延迟10秒
        ):
            success_count += 1
        else:
            fail_count += 1
            # 对于某些模型，失败不阻止继续下载
            if model_key == "kolors":
                print("   ⚠ Kolors 下载失败")
                print("   💡 提示: Kolors 可能需要特殊授权，请访问 https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus")
                print("   💡 注意: 确保已安装最新版本的 diffusers: pip install -U diffusers transformers accelerate")
            else:
                print(f"   ⚠ {model_key} 下载失败")
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 下载总结")
    print("=" * 60)
    print(f"成功: {success_count}/{len(models_to_download)}")
    print(f"失败: {fail_count}/{len(models_to_download)}")
    
    if fail_count > 0:
        print("\n⚠️  部分模型下载失败，请检查：")
        print("  1. HuggingFace 访问权限")
        print("  2. 网络连接和代理配置")
        print("  3. 模型是否需要特殊授权")
        print("  4. 存储空间是否充足（约 50-60GB）")
    
    print("\n✅ 下载完成！")


if __name__ == "__main__":
    main()

