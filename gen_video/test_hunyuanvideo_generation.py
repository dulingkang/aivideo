#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HunyuanVideo视频生成功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gen_video.video_generator import VideoGenerator

def test_hunyuanvideo_generation():
    """测试HunyuanVideo视频生成"""
    print("=" * 60)
    print("HunyuanVideo 1.5 视频生成测试")
    print("=" * 60)
    
    # 初始化VideoGenerator
    config_path = project_root / "gen_video" / "config.yaml"
    print(f"\n1. 加载配置: {config_path}")
    
    generator = VideoGenerator(str(config_path))
    
    # 确保使用HunyuanVideo
    generator.video_config['model_type'] = 'hunyuanvideo'
    print(f"   ✓ 模型类型: {generator.video_config['model_type']}")
    
    # 检查配置
    hunyuan_config = generator.video_config.get('hunyuanvideo', {})
    print(f"\n2. HunyuanVideo配置:")
    print(f"   - use_v15: {hunyuan_config.get('use_v15', True)}")
    print(f"   - model_path: {hunyuan_config.get('model_path')}")
    print(f"   - transformer_subfolder: {hunyuan_config.get('transformer_subfolder')}")
    
    # 加载模型
    print(f"\n3. 加载模型...")
    try:
        generator.load_model()
        print(f"   ✓ 模型加载成功")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 准备测试图像
    print(f"\n4. 准备测试图像...")
    # 查找一个测试图像（优先使用新生成的场景图片）
    test_image_path = None
    possible_paths = [
        # 优先使用新生成的场景图片
        project_root / "gen_video" / "outputs" / "test_flux" / "test_scene_new.png",
        # 备选：其他场景图片
        project_root / "gen_video" / "outputs" / "images" / "什么是黑洞_20251209_235807" / "scene_002.png",
        project_root / "gen_video" / "outputs" / "images" / "什么是黑洞_20251209_234504" / "scene_001.png",
        project_root / "gen_video" / "outputs" / "images" / "什么是黑洞_20251209_233523" / "scene_001.png",
        project_root / "gen_video" / "outputs" / "images" / "什么是黑洞_20251209_232411" / "scene_001.png",
        # 备选：test_flux图片
        project_root / "gen_video" / "outputs" / "test_flux" / "test_scene.png",
        project_root / "gen_video" / "outputs" / "test_flux" / "test_host_face.png",
    ]
    
    # 根据图像内容创建场景配置
    scene_config = None
    
    for path in possible_paths:
        if path.exists():
            test_image_path = str(path)
            print(f"   ✓ 找到测试图像: {test_image_path}")
            break
    
    if not test_image_path:
        print(f"   ⚠ 未找到测试图像，请手动指定图像路径")
        print(f"   可能的路径:")
        for path in possible_paths:
            print(f"     - {path}")
        return False
    
    # 根据图像文件名创建场景配置（优化prompt）
    image_name = Path(test_image_path).stem
    if "scene_new" in image_name or "scene" in image_name:
        # 创建一个详细的场景配置，用于生成更好的prompt
        # 根据实际生成的图像内容，创建更详细的描述
        scene_config = {
            "description": "a beautiful and detailed scene with rich visual elements, excellent composition, professional photography quality",
            "motion_intensity": "moderate",  # 适中的运动
            "camera_motion": {
                "type": "slow_zoom",  # 缓慢缩放
                "params": {"zoom_speed": 0.1}
            },
            "visual": {
                "style": "realistic",  # 使用写实风格
                "composition": "wide shot with cinematic composition, excellent framing and visual balance",
                "lighting": "natural lighting with good contrast",
                "mood": "serene and peaceful atmosphere"
            }
        }
        print(f"   ✓ 已创建详细场景配置（优化prompt）")
    
    # 准备输出路径
    output_dir = project_root / "gen_video" / "outputs" / "test_hunyuanvideo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_video_v2.mp4"  # 使用新文件名
    
    print(f"\n5. 生成视频...")
    print(f"   输入图像: {test_image_path}")
    print(f"   输出视频: {output_path}")
    if scene_config:
        print(f"   场景配置: {scene_config}")
    
    try:
        # 生成视频（传入场景配置以优化prompt）
        result_path = generator.generate_video(
            image_path=test_image_path,
            output_path=str(output_path),
            scene=scene_config,  # 传入场景配置，系统会自动构建详细prompt
            # num_frames和fps会从config.yaml读取，这里不指定使用默认值
        )
        
        print(f"\n   ✓ 视频生成成功!")
        print(f"   输出路径: {result_path}")
        return True
        
    except Exception as e:
        print(f"\n   ✗ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hunyuanvideo_generation()
    sys.exit(0 if success else 1)

