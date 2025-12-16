#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成新的测试场景图像（用于HunyuanVideo测试）
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gen_video.image_generator import ImageGenerator

def generate_test_scene():
    """生成新的测试场景图像"""
    print("=" * 60)
    print("生成新的测试场景图像")
    print("=" * 60)
    
    # 初始化图像生成器
    config_path = project_root / "gen_video" / "config.yaml"
    print(f"\n1. 加载配置: {config_path}")
    
    generator = ImageGenerator(str(config_path))
    
    # 准备输出路径
    output_dir = project_root / "gen_video" / "outputs" / "test_flux"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_scene_v2.png"
    
    # 生成一个更详细的科普场景（黑洞）
    prompt = """A realistic scientific visualization of a black hole in space. 
    The black hole has a glowing accretion disk with intense light around the event horizon. 
    The gravitational field creates a bright ring of light. 
    Distant stars and cosmic dust form the background. 
    The scene captures the mysterious and powerful nature of black holes. 
    Wide shot, cinematic composition, high quality, detailed, professional scientific visualization"""
    
    negative_prompt = "cartoon, anime, illustration, fantasy, unrealistic, low quality, blurry"
    
    print(f"\n2. 生成图像...")
    print(f"   Prompt: {prompt[:100]}...")
    print(f"   输出路径: {output_path}")
    
    try:
        result_path = generator.generate_image(
            prompt=prompt,
            output_path=str(output_path),
            negative_prompt=negative_prompt,
            width=960,  # 匹配HunyuanVideo推荐分辨率
            height=544,
            num_inference_steps=28,
            guidance_scale=7.5,
            model_engine="flux2",  # 使用Flux生成高质量场景
            task_type="scene"
        )
        
        print(f"\n   ✓ 图像生成成功!")
        print(f"   输出路径: {result_path}")
        return result_path
        
    except Exception as e:
        print(f"\n   ✗ 图像生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = generate_test_scene()
    if result:
        print(f"\n✓ 场景图像已生成: {result}")
        print(f"  现在可以使用此图像测试HunyuanVideo生成")
    else:
        print(f"\n✗ 场景图像生成失败")
        sys.exit(1)

