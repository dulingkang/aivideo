#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Flux 图像生成流程
验证加载、LoRA、IP-Adapter 等功能是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_manager import ModelManager
from PIL import Image
import yaml

def load_config():
    """加载配置文件"""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_flux_generation():
    """测试 Flux 图像生成"""
    print("="*70)
    print("Flux 图像生成测试")
    print("="*70)
    print()
    
    # 加载配置
    config = load_config()
    models_root = config.get('models', {}).get('root', 'models')
    models_root = project_root / models_root
    
    print(f"📁 模型根目录: {models_root}")
    print()
    
    # 创建 ModelManager
    print("1️⃣  创建 ModelManager...")
    manager = ModelManager(
        models_root=str(models_root),
        lazy_load=True,
        config_path=str(project_root / "config.yaml")
    )
    print("   ✅ ModelManager 创建成功")
    print()
    
    # 测试场景1: 科普主持人（使用 LoRA）
    print("2️⃣  测试场景1: 科普主持人（使用 LoRA）")
    print("   - 任务: host_face_instantid")
    print("   - 使用 LoRA: host_person_v2")
    print("   - 人脸参考图: kupu_gege.png")
    print()
    
    # 加载人脸参考图
    face_image_path = project_root / "reference_image" / "kupu_gege.png"
    if not face_image_path.exists():
        print(f"   ⚠️  人脸参考图不存在: {face_image_path}")
        print("   ℹ️  跳过人脸测试，使用纯场景测试")
        face_image = None
    else:
        face_image = Image.open(face_image_path)
        print(f"   ✅ 已加载人脸参考图: {face_image_path.name}")
        print(f"      尺寸: {face_image.size}")
    
    # 测试提示词
    prompt = "科普哥哥, (neat modern short hair:1.5), (modern science presenter outfit:1.6), (young friendly face, clear bright eyes:1.3), photorealistic, professional photography, scientific style, high quality, detailed, realistic, Chinese, Asian, in a modern science laboratory, soft lighting"
    negative_prompt = "cartoon, anime, animation, illustration, drawing, sketch, 插画, 绘画, 手绘, 2d, stylized, artistic style, comic style, manga style, female, woman, girl, 女性, 女人, 女孩"
    
    print()
    print("   📝 提示词:")
    print(f"      {prompt[:100]}...")
    print()
    
    try:
        # 生成图像
        print("   🎨 开始生成图像...")
        image = manager.generate(
            task="host_face",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=28,
            guidance_scale=7.5,
            face_image=face_image,
            face_strength=0.8
        )
        
        # 保存结果
        output_dir = project_root / "outputs" / "test_flux"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_host_face.png"
        image.save(output_path)
        
        print()
        print(f"   ✅ 生成成功！")
        print(f"   📁 保存路径: {output_path}")
        print(f"   📐 图像尺寸: {image.size}")
        print()
        
    except Exception as e:
        print()
        print(f"   ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False
    
    # 测试场景2: 纯场景（不使用人脸）
    print("3️⃣  测试场景2: 纯场景（不使用人脸）")
    print("   - 任务: scene")
    print("   - 不使用 LoRA 和人脸")
    print()
    
    scene_prompt = "modern science laboratory, high-tech equipment, clean and bright, professional photography, scientific style, photorealistic, detailed"
    scene_negative = "cartoon, anime, illustration, drawing, sketch, 插画, 绘画"
    
    print("   📝 提示词:")
    print(f"      {scene_prompt}")
    print()
    
    try:
        print("   🎨 开始生成图像...")
        scene_image = manager.generate(
            task="scene",
            prompt=scene_prompt,
            negative_prompt=scene_negative,
            width=1024,
            height=1024,
            num_inference_steps=28,
            guidance_scale=7.5
        )
        
        # 保存结果
        scene_output_path = output_dir / "test_scene.png"
        scene_image.save(scene_output_path)
        
        print()
        print(f"   ✅ 生成成功！")
        print(f"   📁 保存路径: {scene_output_path}")
        print(f"   📐 图像尺寸: {scene_image.size}")
        print()
        
    except Exception as e:
        print()
        print(f"   ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False
    
    print("="*70)
    print("✅ 测试完成！")
    print("="*70)
    print()
    print("📊 测试结果:")
    print("   ✅ 场景1: 科普主持人（LoRA + 人脸）")
    print("   ✅ 场景2: 纯场景")
    print()
    print(f"📁 输出目录: {output_dir}")
    print()
    
    return True

if __name__ == "__main__":
    success = test_flux_generation()
    sys.exit(0 if success else 1)

