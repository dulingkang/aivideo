#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CogVideoX测试脚本
测试CogVideoX基础生成功能、Prompt Engine效果、模型路由
"""

import sys
from pathlib import Path
import time

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_generator import VideoGenerator
from utils.model_router import ModelRouter


def test_cogvideox_basic():
    """测试CogVideoX基础生成功能"""
    print("=" * 60)
    print("测试1: CogVideoX基础生成功能")
    print("=" * 60)
    
    # 创建视频生成器
    generator = VideoGenerator(config_path="config.yaml")
    
    # 测试图像路径（使用一个示例图像，如果没有则创建）
    test_image_dir = Path("outputs/test_images")
    test_image_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否有测试图像
    test_images = list(test_image_dir.glob("*.png")) + list(test_image_dir.glob("*.jpg"))
    if not test_images:
        print("  ⚠ 未找到测试图像，请先准备一张测试图像")
        print(f"    放在目录: {test_image_dir}")
        return
    
    test_image = test_images[0]
    print(f"  使用测试图像: {test_image}")
    
    # 输出路径
    output_dir = Path("outputs/test_cogvideox")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_cogvideox_basic.mp4"
    
    # 简单场景配置
    scene = {
        "type": "novel",
        "description": "a character walking in a fantasy landscape",
        "motion_intensity": "moderate",
        "camera_motion": {"type": "pan"},
        "visual": {
            "composition": "wide shot",
            "lighting": "soft",
            "style": "cinematic"
        }
    }
    
    print(f"\n  场景配置:")
    print(f"    类型: {scene['type']}")
    print(f"    描述: {scene['description']}")
    print(f"    运动强度: {scene['motion_intensity']}")
    
    try:
        print(f"\n  开始生成视频...")
        start_time = time.time()
        
        # 强制使用CogVideoX
        generator.video_config['model_type'] = 'cogvideox'
        
        result_path = generator.generate_video(
            image_path=str(test_image),
            output_path=str(output_path),
            num_frames=81,  # CogVideoX推荐帧数
            fps=16,  # CogVideoX推荐帧率
            scene=scene
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n  ✓ 视频生成成功!")
        print(f"    输出路径: {result_path}")
        print(f"    生成时间: {elapsed_time:.2f}秒")
        
        if Path(result_path).exists():
            file_size = Path(result_path).stat().st_size / (1024 * 1024)
            print(f"    文件大小: {file_size:.2f}MB")
        
        return result_path
        
    except Exception as e:
        print(f"\n  ✗ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cogvideox_with_prompt_engine():
    """测试CogVideoX + Prompt Engine效果"""
    print("\n" + "=" * 60)
    print("测试2: CogVideoX + Prompt Engine")
    print("=" * 60)
    
    generator = VideoGenerator(config_path="config.yaml")
    
    # 确保Prompt Engine已启用
    prompt_engine_config = generator.video_config.get('prompt_engine', {})
    if not prompt_engine_config.get('enabled', True):
        print("  ⚠ Prompt Engine未启用，启用中...")
        generator.video_config.setdefault('prompt_engine', {})['enabled'] = True
    
    # 测试图像
    test_image_dir = Path("outputs/test_images")
    test_images = list(test_image_dir.glob("*.png")) + list(test_image_dir.glob("*.jpg"))
    if not test_images:
        print("  ⚠ 未找到测试图像")
        return
    
    test_image = test_images[0]
    print(f"  使用测试图像: {test_image}")
    
    # 输出路径
    output_dir = Path("outputs/test_cogvideox")
    output_path = output_dir / "test_cogvideox_prompt_engine.mp4"
    
    # 详细场景配置（用于Prompt Engine）
    scene = {
        "type": "novel",
        "description": "a young warrior walking through a mystical forest",
        "motion_intensity": "gentle",
        "camera_motion": {
            "type": "dolly",
            "speed": "slow"
        },
        "visual": {
            "composition": "wide shot, establishing",
            "lighting": "dramatic rim light, soft ambient",
            "style": "cinematic",
            "mood": "mysterious and peaceful"
        }
    }
    
    print(f"\n  场景配置（详细）:")
    print(f"    类型: {scene['type']}")
    print(f"    描述: {scene['description']}")
    print(f"    构图: {scene['visual']['composition']}")
    print(f"    光线: {scene['visual']['lighting']}")
    print(f"    风格: {scene['visual']['style']}")
    
    try:
        print(f"\n  开始生成视频（使用Prompt Engine）...")
        start_time = time.time()
        
        # 强制使用CogVideoX
        generator.video_config['model_type'] = 'cogvideox'
        
        result_path = generator.generate_video(
            image_path=str(test_image),
            output_path=str(output_path),
            num_frames=81,
            fps=16,
            scene=scene
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n  ✓ 视频生成成功!")
        print(f"    输出路径: {result_path}")
        print(f"    生成时间: {elapsed_time:.2f}秒")
        
        return result_path
        
    except Exception as e:
        print(f"\n  ✗ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_router():
    """测试模型路由自动选择"""
    print("\n" + "=" * 60)
    print("测试3: 模型路由自动选择")
    print("=" * 60)
    
    generator = VideoGenerator(config_path="config.yaml")
    
    # 创建模型路由器
    router = ModelRouter(generator.video_config)
    
    # 测试不同场景类型
    test_scenes = [
        {
            "name": "小说场景",
            "scene": {"type": "novel"},
            "user_tier": "basic",
            "expected": "cogvideox"
        },
        {
            "name": "科普场景",
            "scene": {"type": "scientific"},
            "user_tier": "basic",
            "expected": "hunyuanvideo"  # 科普场景应该使用高端产线
        },
        {
            "name": "政府场景",
            "scene": {"type": "government"},
            "user_tier": "basic",
            "expected": "hunyuanvideo"
        },
        {
            "name": "企业用户-通用场景",
            "scene": {"type": "general"},
            "user_tier": "enterprise",
            "expected": "hunyuanvideo"  # 企业用户应该使用高端产线
        },
    ]
    
    print("\n  测试场景路由:")
    for test_case in test_scenes:
        selected = router.select_model(
            scene=test_case["scene"],
            user_tier=test_case["user_tier"],
            force_model=None
        )
        
        status = "✓" if selected == test_case["expected"] else "✗"
        print(f"    {status} {test_case['name']}: {selected} (期望: {test_case['expected']})")
    
    # 测试显存限制
    print("\n  测试显存限制:")
    import torch
    if torch.cuda.is_available():
        available_memory = router._get_available_memory()
        print(f"    可用显存: {available_memory:.2f}GB")
        
        # 测试低显存情况（模拟）
        low_memory_scene = {"type": "scientific"}
        selected = router.select_model(
            scene=low_memory_scene,
            user_tier="basic",
            available_memory=10.0  # 模拟10GB显存
        )
        print(f"    低显存场景(10GB): {selected} (应该选择cogvideox)")
    else:
        print("    ⚠ CUDA不可用，跳过显存测试")


def test_cogvideox_different_scenes():
    """测试CogVideoX在不同场景类型下的表现"""
    print("\n" + "=" * 60)
    print("测试4: CogVideoX不同场景类型")
    print("=" * 60)
    
    generator = VideoGenerator(config_path="config.yaml")
    generator.video_config['model_type'] = 'cogvideox'
    
    # 测试图像
    test_image_dir = Path("outputs/test_images")
    test_images = list(test_image_dir.glob("*.png")) + list(test_image_dir.glob("*.jpg"))
    if not test_images:
        print("  ⚠ 未找到测试图像")
        return
    
    test_image = test_images[0]
    output_dir = Path("outputs/test_cogvideox")
    
    # 不同场景类型
    scene_types = [
        {
            "type": "novel",
            "name": "小说风格",
            "description": "a fantasy character in a mystical world"
        },
        {
            "type": "drama",
            "name": "短剧风格",
            "description": "a dramatic scene with emotional depth"
        },
        {
            "type": "daily",
            "name": "日常风格",
            "description": "a casual everyday scene"
        },
    ]
    
    results = []
    for scene_config in scene_types:
        print(f"\n  测试场景: {scene_config['name']} ({scene_config['type']})")
        
        scene = {
            "type": scene_config["type"],
            "description": scene_config["description"],
            "motion_intensity": "moderate",
            "visual": {
                "style": scene_config["type"]
            }
        }
        
        output_path = output_dir / f"test_cogvideox_{scene_config['type']}.mp4"
        
        try:
            start_time = time.time()
            result_path = generator.generate_video(
                image_path=str(test_image),
                output_path=str(output_path),
                num_frames=81,
                fps=16,
                scene=scene
            )
            elapsed_time = time.time() - start_time
            
            print(f"    ✓ 生成成功: {elapsed_time:.2f}秒")
            results.append({
                "scene": scene_config["name"],
                "path": result_path,
                "time": elapsed_time
            })
        except Exception as e:
            print(f"    ✗ 生成失败: {e}")
    
    print(f"\n  总结:")
    for result in results:
        print(f"    - {result['scene']}: {result['time']:.2f}秒")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CogVideoX 完整测试")
    print("=" * 60)
    
    # 运行测试
    print("\n提示: 请确保已准备好测试图像（放在 outputs/test_images/ 目录）")
    print("如果没有测试图像，某些测试将跳过\n")
    
    # 测试1: 基础生成
    test_cogvideox_basic()
    
    # 测试2: Prompt Engine
    test_cogvideox_with_prompt_engine()
    
    # 测试3: 模型路由
    test_model_router()
    
    # 测试4: 不同场景类型
    # test_cogvideox_different_scenes()  # 可选，需要较长时间
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n生成的视频保存在: outputs/test_cogvideox/")
    print("请查看生成的视频，检查质量是否符合预期")

