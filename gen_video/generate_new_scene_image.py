#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成一张新的场景图片用于测试
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_manager import ModelManager
import yaml

def generate_new_scene():
    """生成一张新的场景图片"""
    print("=" * 60)
    print("生成新的场景图片")
    print("=" * 60)
    
    # 加载配置
    config_path = project_root / "gen_video" / "config.yaml"
    print(f"\n1. 加载配置: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    models_root = config.get('models', {}).get('root', 'models')
    models_root = project_root / "gen_video" / models_root if not Path(models_root).is_absolute() else Path(models_root)
    
    # 创建 ModelManager
    print(f"   模型根目录: {models_root}")
    manager = ModelManager(
        models_root=str(models_root),
        lazy_load=True,
        config_path=str(config_path)
    )
    print("   ✓ ModelManager 初始化成功")
    
    # 准备不同的场景提示词
    scene_prompts = [
        "a futuristic space station interior, high-tech control panels, holographic displays, astronauts working, cinematic lighting, photorealistic, detailed, 8k",
        "a beautiful underwater coral reef, colorful fish swimming, sunlight filtering through water, serene and peaceful, photorealistic, detailed, professional photography",
        "a modern city skyline at sunset, glass skyscrapers reflecting golden light, urban landscape, dramatic clouds, photorealistic, cinematic, 8k",
        "a peaceful mountain landscape, snow-capped peaks, alpine meadow with wildflowers, clear blue sky, photorealistic, detailed, professional photography",
        "a scientific laboratory with advanced equipment, scientists in white coats, glowing screens, modern technology, clean and bright, photorealistic, detailed"
    ]
    
    import random
    selected_prompt = random.choice(scene_prompts)
    negative_prompt = "cartoon, anime, illustration, drawing, sketch, 插画, 绘画, low quality, blurry, distorted"
    
    print(f"\n2. 场景提示词:")
    print(f"   {selected_prompt}")
    print(f"\n   负面提示词:")
    print(f"   {negative_prompt}")
    
    # 准备输出路径
    output_dir = project_root / "gen_video" / "outputs" / "test_flux"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_scene_new.png"
    
    print(f"\n3. 生成图像...")
    print(f"   输出路径: {output_path}")
    
    try:
        # 使用 ModelManager 生成图像（场景任务）
        print("   🎨 开始生成图像...")
        image = manager.generate(
            task="scene",  # 场景生成任务
            prompt=selected_prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=28,
            guidance_scale=7.5
        )
        
        # 保存图像
        image.save(output_path)
        
        print(f"\n   ✓ 图像生成成功!")
        print(f"   输出路径: {output_path}")
        print(f"   图像尺寸: {image.size}")
        return output_path
        
    except Exception as e:
        print(f"\n   ✗ 图像生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = generate_new_scene()
    sys.exit(0 if result else 1)

