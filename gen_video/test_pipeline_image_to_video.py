#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流水线测试：图像生成 → CogVideoX视频生成
测试从图像生成到视频生成的完整流程
"""

import sys
from pathlib import Path
import time

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from image_generator import ImageGenerator
from video_generator import VideoGenerator


def test_complete_pipeline():
    """测试完整流水线：图像生成 → CogVideoX视频生成"""
    print("=" * 80)
    print("完整流水线测试：图像生成 → CogVideoX视频生成")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path("outputs/test_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 第一步：生成图像 ==========
    print("\n" + "=" * 80)
    print("步骤1: 生成测试图像")
    print("=" * 80)
    
    # 使用相对于当前脚本的配置文件路径
    config_path = Path(__file__).parent / "config.yaml"
    image_generator = ImageGenerator(config_path=str(config_path))
    
    # 测试场景配置
    test_scenes = [
        {
            "name": "小说场景",
            "prompt": "a young warrior walking through a mystical forest, fantasy style, cinematic lighting, wide shot",
            "scene_type": "novel",
            "width": 1360,  # 匹配CogVideoX推荐分辨率
            "height": 768,
        },
        {
            "name": "日常场景",
            "prompt": "a person walking on a city street, realistic style, natural lighting, medium shot",
            "scene_type": "daily",
            "width": 1360,
            "height": 768,
        },
        {
            "name": "短剧场景",
            "prompt": "a dramatic scene with emotional depth, cinematic style, dramatic lighting, close-up",
            "scene_type": "drama",
            "width": 1360,
            "height": 768,
        },
    ]
    
    generated_images = []
    
    for i, scene_config in enumerate(test_scenes, 1):
        print(f"\n【场景 {i}/{len(test_scenes)}】{scene_config['name']}")
        print(f"  Prompt: {scene_config['prompt']}")
        print(f"  分辨率: {scene_config['width']}x{scene_config['height']}")
        
        # 输出路径
        image_output_path = output_dir / f"scene_{i:02d}_{scene_config['scene_type']}.png"
        
        # 场景配置
        scene = {
            "prompt": scene_config["prompt"],
            "width": scene_config["width"],
            "height": scene_config["height"],
            "type": scene_config["scene_type"],
            "visual": {
                "style": scene_config["scene_type"],
                "composition": "wide shot" if "wide" in scene_config["prompt"].lower() else "medium shot"
            }
        }
        
        try:
            print(f"  开始生成图像...")
            start_time = time.time()
            
            generated_image_path = image_generator.generate_image(
                prompt=scene_config["prompt"],
                output_path=image_output_path,
                negative_prompt=None,  # 使用默认
                num_inference_steps=None,  # 使用配置默认值
                guidance_scale=None,  # 使用配置默认值
                seed=None,
                scene=scene,
                model_engine="auto",  # 自动选择模型
                task_type="scene"  # 场景生成
            )
            
            elapsed_time = time.time() - start_time
            
            if generated_image_path and Path(generated_image_path).exists():
                file_size = Path(generated_image_path).stat().st_size / (1024 * 1024)
                print(f"  ✓ 图像生成成功!")
                print(f"    路径: {generated_image_path}")
                print(f"    大小: {file_size:.2f}MB")
                print(f"    耗时: {elapsed_time:.2f}秒")
                
                generated_images.append({
                    "path": generated_image_path,
                    "scene_config": scene_config,
                    "scene": scene
                })
            else:
                print(f"  ✗ 图像生成失败: 文件不存在")
                
        except Exception as e:
            print(f"  ✗ 图像生成失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not generated_images:
        print("\n  ⚠ 没有成功生成任何图像，无法继续测试视频生成")
        return
    
    print(f"\n  ✓ 成功生成 {len(generated_images)} 张图像")
    
    # ========== 清理图像生成器的模型和显存 ==========
    print("\n" + "=" * 80)
    print("步骤1.5: 清理图像生成器模型，释放显存")
    print("=" * 80)
    
    try:
        # 清理图像生成器的所有模型
        import torch
        import gc
        
        # 删除图像生成器的pipeline引用
        if hasattr(image_generator, 'pipeline') and image_generator.pipeline is not None:
            del image_generator.pipeline
            image_generator.pipeline = None
            print("  ✓ 已清理图像生成器pipeline")
        
        if hasattr(image_generator, 'flux_pipeline') and image_generator.flux_pipeline is not None:
            del image_generator.flux_pipeline
            image_generator.flux_pipeline = None
            print("  ✓ 已清理Flux pipeline")
        
        if hasattr(image_generator, 'sdxl_pipeline') and image_generator.sdxl_pipeline is not None:
            del image_generator.sdxl_pipeline
            image_generator.sdxl_pipeline = None
            print("  ✓ 已清理SDXL pipeline")
        
        # 清理ModelManager（如果使用）
        if hasattr(image_generator, 'model_manager') and image_generator.model_manager is not None:
            if hasattr(image_generator.model_manager, 'unload'):
                image_generator.model_manager.unload()
                print("  ✓ 已卸载ModelManager所有模型")
        
        # 彻底清理显存
        if torch.cuda.is_available():
            for _ in range(5):  # 多次清理
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  ℹ 清理后显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
        
        # 删除图像生成器对象
        del image_generator
        gc.collect()
        print("  ✓ 图像生成器对象已删除")
        
    except Exception as e:
        print(f"  ⚠ 清理显存时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 第二步：使用生成的图像测试CogVideoX ==========
    print("\n" + "=" * 80)
    print("步骤2: 使用生成的图像测试CogVideoX视频生成")
    print("=" * 80)
    
    # 使用相对于当前脚本的配置文件路径
    config_path = Path(__file__).parent / "config.yaml"
    video_generator = VideoGenerator(config_path=str(config_path))
    
    # 强制使用CogVideoX
    video_generator.video_config['model_type'] = 'cogvideox'
    
    # 降低显存限制，允许使用更多显存（因为已经清理了图像生成器）
    cogvideox_config = video_generator.video_config.get('cogvideox', {})
    cogvideox_config['max_memory_fraction'] = 0.8  # 允许使用80%显存
    print("  ℹ 已调整显存限制到80%（因为已清理图像生成器模型）")
    
    video_output_dir = output_dir / "videos"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_videos = []
    
    for i, image_info in enumerate(generated_images, 1):
        image_path = image_info["path"]
        scene_config = image_info["scene_config"]
        scene = image_info["scene"]
        
        print(f"\n【视频 {i}/{len(generated_images)}】{scene_config['name']}")
        print(f"  输入图像: {Path(image_path).name}")
        
        # 视频输出路径
        video_output_path = video_output_dir / f"video_{i:02d}_{scene_config['scene_type']}.mp4"
        
        # 扩展场景配置（用于视频生成）
        video_scene = {
            **scene,
            "type": scene_config["scene_type"],
            "description": scene_config["prompt"],
            "motion_intensity": "moderate",
            "camera_motion": {
                "type": "pan",
                "speed": "slow"
            },
            "visual": {
                **scene.get("visual", {}),
                "lighting": "natural" if "natural" in scene_config["prompt"].lower() else "cinematic",
                "mood": "peaceful" if scene_config["scene_type"] == "daily" else "dramatic"
            }
        }
        
        try:
            print(f"  开始生成视频（CogVideoX）...")
            start_time = time.time()
            
            result_path = video_generator.generate_video(
                image_path=str(image_path),
                output_path=str(video_output_path),
                num_frames=24,  # CogVideoX推荐帧数
                fps=16,  # CogVideoX推荐帧率
                scene=video_scene
            )
            
            elapsed_time = time.time() - start_time
            
            if result_path and Path(result_path).exists():
                file_size = Path(result_path).stat().st_size / (1024 * 1024)
                print(f"  ✓ 视频生成成功!")
                print(f"    路径: {result_path}")
                print(f"    大小: {file_size:.2f}MB")
                print(f"    耗时: {elapsed_time:.2f}秒")
                
                generated_videos.append({
                    "path": result_path,
                    "image_path": image_path,
                    "scene_config": scene_config,
                    "time": elapsed_time
                })
            else:
                print(f"  ✗ 视频生成失败: 文件不存在")
                
        except Exception as e:
            print(f"  ✗ 视频生成失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    print(f"\n图像生成:")
    print(f"  成功: {len(generated_images)}/{len(test_scenes)}")
    for img_info in generated_images:
        print(f"    - {img_info['scene_config']['name']}: {Path(img_info['path']).name}")
    
    print(f"\n视频生成:")
    print(f"  成功: {len(generated_videos)}/{len(generated_images)}")
    total_video_time = sum(v["time"] for v in generated_videos)
    for vid_info in generated_videos:
        print(f"    - {vid_info['scene_config']['name']}: {vid_info['time']:.2f}秒")
    
    if generated_videos:
        avg_time = total_video_time / len(generated_videos)
        print(f"\n  平均生成时间: {avg_time:.2f}秒/视频")
    
    print(f"\n输出目录:")
    print(f"  图像: {output_dir}")
    print(f"  视频: {video_output_dir}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def test_single_scene_pipeline():
    """测试单个场景的完整流水线（快速测试）"""
    print("=" * 80)
    print("快速测试：单个场景完整流水线")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path("outputs/test_pipeline_single")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 生成图像 ==========
    print("\n步骤1: 生成测试图像")
    print("-" * 80)
    
    # 使用相对于当前脚本的配置文件路径
    config_path = Path(__file__).parent / "config.yaml"
    image_generator = ImageGenerator(config_path=str(config_path))
    
    prompt = "a beautiful landscape with mountains and a lake, cinematic style, wide shot, natural lighting"
    image_output_path = output_dir / "test_scene.png"
    
    scene = {
        "prompt": prompt,
        "width": 1360,
        "height": 768,
        "type": "general",
        "visual": {
            "style": "cinematic",
            "composition": "wide shot"
        }
    }
    
    try:
        print(f"  Prompt: {prompt}")
        print(f"  分辨率: 1360x768")
        print(f"  开始生成...")
        
        start_time = time.time()
        generated_image_path = image_generator.generate_image(
            prompt=prompt,
            output_path=image_output_path,
            scene=scene,
            model_engine="auto",
            task_type="scene"
        )
        image_time = time.time() - start_time
        
        if generated_image_path and Path(generated_image_path).exists():
            print(f"  ✓ 图像生成成功 ({image_time:.2f}秒)")
            print(f"    路径: {generated_image_path}")
        else:
            print(f"  ✗ 图像生成失败")
            return
    except Exception as e:
        print(f"  ✗ 图像生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 清理图像生成器的模型和显存 ==========
    print("\n步骤1.5: 清理图像生成器模型，释放显存")
    print("-" * 80)
    
    try:
        # 清理图像生成器的所有模型
        import torch
        import gc
        
        # 删除图像生成器的pipeline引用
        if hasattr(image_generator, 'pipeline') and image_generator.pipeline is not None:
            del image_generator.pipeline
            image_generator.pipeline = None
            print("  ✓ 已清理图像生成器pipeline")
        
        if hasattr(image_generator, 'flux_pipeline') and image_generator.flux_pipeline is not None:
            del image_generator.flux_pipeline
            image_generator.flux_pipeline = None
            print("  ✓ 已清理Flux pipeline")
        
        if hasattr(image_generator, 'sdxl_pipeline') and image_generator.sdxl_pipeline is not None:
            del image_generator.sdxl_pipeline
            image_generator.sdxl_pipeline = None
            print("  ✓ 已清理SDXL pipeline")
        
        # 清理ModelManager（如果使用）
        if hasattr(image_generator, 'model_manager') and image_generator.model_manager is not None:
            if hasattr(image_generator.model_manager, 'unload'):
                image_generator.model_manager.unload()
                print("  ✓ 已卸载ModelManager所有模型")
        
        # 彻底清理显存
        if torch.cuda.is_available():
            for _ in range(5):  # 多次清理
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  ℹ 清理后显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
        
        # 删除图像生成器对象
        del image_generator
        gc.collect()
        print("  ✓ 图像生成器对象已删除")
        
    except Exception as e:
        print(f"  ⚠ 清理显存时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 生成视频 ==========
    print("\n步骤2: 生成视频（CogVideoX）")
    print("-" * 80)
    
    # 使用相对于当前脚本的配置文件路径
    config_path = Path(__file__).parent / "config.yaml"
    video_generator = VideoGenerator(config_path=str(config_path))
    video_generator.video_config['model_type'] = 'cogvideox'
    
    # 降低显存限制，允许使用更多显存（因为已经清理了图像生成器）
    cogvideox_config = video_generator.video_config.get('cogvideox', {})
    cogvideox_config['max_memory_fraction'] = 0.8  # 允许使用80%显存
    print("  ℹ 已调整显存限制到80%（因为已清理图像生成器模型）")
    
    video_output_path = output_dir / "test_video.mp4"
    
    video_scene = {
        "type": "general",
        "description": prompt,
        "motion_intensity": "gentle",
        "camera_motion": {"type": "pan"},
        "visual": {
            "composition": "wide shot",
            "lighting": "natural",
            "style": "cinematic"
        }
    }
    
    try:
        print(f"  输入图像: {Path(generated_image_path).name}")
        print(f"  开始生成...")
        
        start_time = time.time()
        result_path = video_generator.generate_video(
            image_path=str(generated_image_path),
            output_path=str(video_output_path),
            num_frames=48,
            fps=16,
            scene=video_scene
        )
        video_time = time.time() - start_time
        
        if result_path and Path(result_path).exists():
            file_size = Path(result_path).stat().st_size / (1024 * 1024)
            print(f"  ✓ 视频生成成功 ({video_time:.2f}秒)")
            print(f"    路径: {result_path}")
            print(f"    大小: {file_size:.2f}MB")
        else:
            print(f"  ✗ 视频生成失败")
    except Exception as e:
        print(f"  ✗ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    print(f"  图像生成: {image_time:.2f}秒")
    print(f"  视频生成: {video_time:.2f}秒")
    print(f"  总耗时: {image_time + video_time:.2f}秒")
    print(f"\n  输出目录: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试完整流水线：图像生成 → CogVideoX视频生成")
    parser.add_argument(
        "--mode",
        choices=["full", "single"],
        default="single",
        help="测试模式：full（完整测试多个场景）或 single（快速测试单个场景）"
    )
    
    args = parser.parse_args()
    
    if args.mode == "full":
        test_complete_pipeline()
    else:
        test_single_scene_pipeline()

