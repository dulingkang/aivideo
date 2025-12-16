#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Flux模型ID的脚本
帮助确认本地模型和HuggingFace模型ID
"""

import json
import os
from pathlib import Path
import sys

def check_local_model(model_path: str):
    """检查本地模型的类型"""
    print("=" * 60)
    print("检查本地模型")
    print("=" * 60)
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return None
    
    # 检查 model_index.json
    model_index = model_path / "model_index.json"
    if model_index.exists():
        with open(model_index, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        class_name = config.get("_class_name", "未知")
        version = config.get("_diffusers_version", "未知")
        
        print(f"模型类名: {class_name}")
        print(f"Diffusers版本: {version}")
        print(f"模型路径: {model_path}")
        
        return {
            "class_name": class_name,
            "path": str(model_path),
            "is_flux1": "FLUX.1" in str(model_path) or "flux1" in str(model_path).lower(),
            "is_flux2": "FLUX.2" in str(model_path) or "flux2" in str(model_path).lower()
        }
    
    # 检查 README.md
    readme = model_path / "README.md"
    if readme.exists():
        with open(readme, 'r', encoding='utf-8') as f:
            content = f.read()
            if "FLUX.1 [schnell]" in content:
                print("📄 README.md 显示: FLUX.1 [schnell]")
                return {"is_flux1": True, "version": "schnell"}
            elif "FLUX.2" in content:
                print("📄 README.md 显示: FLUX.2")
                return {"is_flux2": True}
    
    print("⚠️  无法确定模型类型")
    return None


def get_huggingface_model_ids():
    """获取Flux模型的HuggingFace ID列表"""
    print("\n" + "=" * 60)
    print("Flux 模型的 HuggingFace ID（参考）")
    print("=" * 60)
    
    flux_models = {
        "FLUX.1 [dev]": "black-forest-labs/FLUX.1-dev",
        "FLUX.1 [schnell]": "black-forest-labs/FLUX.1-schnell",
        "FLUX.2 [dev]": "black-forest-labs/FLUX.2-dev",  # 需要确认
        "FLUX.2 [schnell]": "black-forest-labs/FLUX.2-schnell",  # 需要确认
        "FLUX.2 [pro]": "black-forest-labs/FLUX.2-pro",  # 需要确认（可能不公开）
    }
    
    print("\n已知的Flux模型ID:")
    for name, model_id in flux_models.items():
        print(f"  - {name}: {model_id}")
    
    print("\n⚠️  注意:")
    print("  - FLUX.2 的模型ID可能需要确认，建议访问 HuggingFace 确认")
    print("  - 访问: https://huggingface.co/black-forest-labs")
    print("  - 或使用: huggingface-cli search black-forest-labs")
    
    return flux_models


def recommend_model_id(local_info):
    """根据本地模型信息推荐正确的模型ID"""
    print("\n" + "=" * 60)
    print("推荐配置")
    print("=" * 60)
    
    if local_info and local_info.get("is_flux1"):
        print("✅ 本地模型是 FLUX.1，使用:")
        print("   base_model: black-forest-labs/FLUX.1-schnell")
        return "black-forest-labs/FLUX.1-schnell"
    elif local_info and local_info.get("is_flux2"):
        print("✅ 本地模型是 FLUX.2，应该使用:")
        print("   base_model: black-forest-labs/FLUX.2-dev 或 black-forest-labs/FLUX.2-schnell")
        print("   ⚠️  需要确认实际的Flux.2模型ID")
        print("\n   验证方法:")
        print("   1. 访问 https://huggingface.co/black-forest-labs")
        print("   2. 查找 FLUX.2 相关的仓库")
        print("   3. 或运行: huggingface-cli search black-forest-labs | grep -i flux.2")
        return None
    else:
        print("⚠️  无法确定模型类型，建议:")
        print("   1. 检查 model_index.json 和 README.md")
        print("   2. 确认模型是从哪个HuggingFace仓库下载的")
        return None


def main():
    """主函数"""
    print("Flux 模型ID检查工具")
    print("=" * 60)
    
    # 检查本地模型
    model_path = "/vepfs-dev/shawn/vid/fanren/gen_video/models/flux2-dev"
    local_info = check_local_model(model_path)
    
    # 获取HuggingFace模型ID列表
    get_huggingface_model_ids()
    
    # 推荐配置
    recommended_id = recommend_model_id(local_info)
    
    if recommended_id:
        print(f"\n📝 配置文件建议:")
        print(f"   base_model: {recommended_id}")
    
    print("\n" + "=" * 60)
    print("如何确认FLUX.2模型ID:")
    print("=" * 60)
    print("""
方法1: 使用 HuggingFace CLI
  huggingface-cli search black-forest-labs | grep -i flux

方法2: 访问网页
  https://huggingface.co/black-forest-labs
  查找包含 FLUX.2 的仓库

方法3: 检查模型下载历史
  查看模型是从哪个仓库下载的
  git log 或下载脚本的历史记录

方法4: 测试加载模型
  from diffusers import FluxPipeline
  pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.2-dev")
  # 如果能成功加载，说明ID正确
    """)


if __name__ == "__main__":
    main()

