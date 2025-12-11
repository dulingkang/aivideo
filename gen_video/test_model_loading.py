#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有模型的加载情况
"""

import sys
from pathlib import Path
import torch
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def check_model_structure(model_path: Path, model_name: str) -> dict:
    """检查模型目录结构"""
    result = {
        "name": model_name,
        "path": str(model_path),
        "exists": model_path.exists(),
        "has_model_index": False,
        "has_weights": False,
        "size_gb": 0.0,
        "files": []
    }
    
    if not model_path.exists():
        return result
    
    # 计算大小
    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    result["size_gb"] = total_size / (1024**3)
    
    # 检查 model_index.json
    model_index = model_path / "model_index.json"
    result["has_model_index"] = model_index.exists()
    
    # 检查权重文件
    weight_files = list(model_path.rglob("*.safetensors")) + list(model_path.rglob("*.bin"))
    result["has_weights"] = len(weight_files) > 0
    result["files"] = [f.name for f in weight_files[:5]]  # 只显示前5个
    
    return result

def test_load_flux1():
    """测试 Flux.1 加载"""
    print("\n" + "="*80)
    print("测试 Flux.1 模型加载")
    print("="*80)
    
    model_path = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/flux1-dev")
    info = check_model_structure(model_path, "Flux.1")
    
    print(f"路径: {info['path']}")
    print(f"存在: {'✅' if info['exists'] else '❌'}")
    print(f"大小: {info['size_gb']:.2f} GB")
    print(f"model_index.json: {'✅' if info['has_model_index'] else '❌'}")
    print(f"权重文件: {'✅' if info['has_weights'] else '❌'}")
    
    if not info['exists']:
        print("❌ 模型目录不存在")
        return False
    
    try:
        from diffusers import DiffusionPipeline
        print("\n尝试加载模型...")
        # 使用 balanced 策略（自动分配设备）
        pipeline = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="balanced"  # 使用 balanced 策略
        )
        print("✅ Flux.1 模型可以正常加载")
        del pipeline
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ Flux.1 模型加载失败: {e}")
        return False

def test_load_flux2():
    """测试 Flux.2 加载"""
    print("\n" + "="*80)
    print("测试 Flux.2 模型加载")
    print("="*80)
    
    model_path = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev")
    info = check_model_structure(model_path, "Flux.2")
    
    print(f"路径: {info['path']}")
    print(f"存在: {'✅' if info['exists'] else '❌'}")
    print(f"大小: {info['size_gb']:.2f} GB")
    print(f"model_index.json: {'✅' if info['has_model_index'] else '❌'}")
    print(f"权重文件: {'✅' if info['has_weights'] else '❌'}")
    
    if not info['exists']:
        print("❌ 模型目录不存在")
        return False
    
    try:
        from diffusers import DiffusionPipeline
        print("\n尝试加载模型...")
        pipeline = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        print("✅ Flux.2 模型可以正常加载")
        del pipeline
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ Flux.2 模型加载失败: {e}")
        return False

def test_load_kolors():
    """测试 Kolors 加载"""
    print("\n" + "="*80)
    print("测试 Kolors 模型加载")
    print("="*80)
    
    model_path = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/kolors")
    info = check_model_structure(model_path, "Kolors")
    
    print(f"路径: {info['path']}")
    print(f"存在: {'✅' if info['exists'] else '❌'}")
    print(f"大小: {info['size_gb']:.2f} GB")
    print(f"model_index.json: {'✅' if info['has_model_index'] else '❌'}")
    print(f"权重文件: {'✅' if info['has_weights'] else '❌'}")
    
    if not info['exists']:
        print("❌ 模型目录不存在")
        return False
    
    try:
        from diffusers import DiffusionPipeline
        print("\n尝试加载模型...")
        # Kolors 可能不是标准 diffusers 格式，先检查
        if not info['has_model_index']:
            print("⚠️  缺少 model_index.json，可能是 IP-Adapter 权重")
            print("   需要配合基础模型使用")
            return False
        
        pipeline = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="balanced"
        )
        print("✅ Kolors 模型可以正常加载")
        del pipeline
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ Kolors 模型加载失败: {e}")
        return False

def test_load_sd3_turbo():
    """测试 SD3 Turbo 加载"""
    print("\n" + "="*80)
    print("测试 SD3.5 Large Turbo 模型加载")
    print("="*80)
    
    model_path = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/sd3-turbo")
    info = check_model_structure(model_path, "SD3.5 Large Turbo")
    
    print(f"路径: {info['path']}")
    print(f"存在: {'✅' if info['exists'] else '❌'}")
    print(f"大小: {info['size_gb']:.2f} GB")
    print(f"model_index.json: {'✅' if info['has_model_index'] else '❌'}")
    print(f"权重文件: {'✅' if info['has_weights'] else '❌'}")
    
    if not info['exists']:
        print("❌ 模型目录不存在")
        return False
    
    if not info['has_model_index']:
        print("❌ 缺少 model_index.json，不是标准 diffusers 格式")
        return False
    
    try:
        from diffusers import StableDiffusion3Pipeline
        print("\n尝试加载模型...")
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        print("✅ SD3.5 Large Turbo 模型可以正常加载")
        del pipeline
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ SD3.5 Large Turbo 模型加载失败: {e}")
        return False

def test_load_hunyuan_dit():
    """测试 Hunyuan-DiT 加载"""
    print("\n" + "="*80)
    print("测试 Hunyuan-DiT 模型加载")
    print("="*80)
    
    model_path = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-dit/t2i")
    info = check_model_structure(model_path, "Hunyuan-DiT")
    
    print(f"路径: {info['path']}")
    print(f"存在: {'✅' if info['exists'] else '❌'}")
    print(f"大小: {info['size_gb']:.2f} GB")
    print(f"model_index.json: {'✅' if info['has_model_index'] else '❌'}")
    print(f"权重文件: {'✅' if info['has_weights'] else '❌'}")
    
    if not info['exists']:
        print("❌ 模型目录不存在")
        return False
    
    if not info['has_model_index']:
        print("❌ 缺少 model_index.json")
        return False
    
    try:
        from diffusers import DiffusionPipeline
        print("\n尝试加载模型...")
        pipeline = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        print("✅ Hunyuan-DiT 模型可以正常加载")
        del pipeline
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"❌ Hunyuan-DiT 模型加载失败: {e}")
        return False

def main():
    """主函数"""
    print("="*80)
    print("模型加载测试")
    print("="*80)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {'✅' if torch.cuda.is_available() else '❌'}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    # 测试各个模型
    results["Flux.1"] = test_load_flux1()
    results["Flux.2"] = test_load_flux2()
    results["Kolors"] = test_load_kolors()
    results["SD3.5 Large Turbo"] = test_load_sd3_turbo()
    results["Hunyuan-DiT"] = test_load_hunyuan_dit()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    for model_name, success in results.items():
        status = "✅ 可以加载" if success else "❌ 无法加载"
        print(f"{model_name:20s}: {status}")
    
    all_success = all(results.values())
    print(f"\n总体状态: {'✅ 所有模型都可以正常加载' if all_success else '❌ 部分模型无法加载'}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

