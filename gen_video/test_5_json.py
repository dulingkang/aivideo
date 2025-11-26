#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 lingjie/5.json 脚本
测试新字段集成（duration, visual, face_style_auto）
支持测试 AnimateDiff-SDXL 视频生成
"""

import os
import sys
import json
import yaml
from pathlib import Path

# 设置PyTorch CUDA内存分配配置，避免内存碎片
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_generator import ImageGenerator
from video_generator import VideoGenerator
from main import AIVideoPipeline


def test_stage1_only():
    """只测试阶段1：图像生成（测试新字段集成）"""
    print("=" * 60)
    print("测试 lingjie/5.json - 阶段1：图像生成")
    print("=" * 60)
    
    script_path = Path(__file__).parent.parent / "lingjie" / "5.json"
    
    if not script_path.exists():
        print(f"✗ 脚本文件不存在: {script_path}")
        return False
    
    print(f"\n脚本路径: {script_path}")
    
    try:
        # 初始化图像生成器
        print("\n[1] 初始化 ImageGenerator...")
        image_gen = ImageGenerator("config.yaml")
        print("✓ ImageGenerator 初始化成功")
        
        # 生成图像（会自动生成 face_style_auto 并应用新字段）
        print("\n[2] 开始生成场景图像...")
        print("   这将自动：")
        print("   - 生成 face_style_auto（如果不存在）")
        print("   - 使用 visual 字段构建 prompt")
        print("   - 应用 face_style_auto 调整参数")
        print("   - 使用 duration 字段（在视频生成时）")
        
        output_name = "lingjie_ep5_test"
        saved_paths = image_gen.generate_from_script(
            str(script_path),
            output_dir=f"outputs/images/{output_name}",
            overwrite=False,  # 不覆盖已有图像
            update_script=True,  # 更新脚本中的 image_path
        )
        
        print(f"\n✓ 图像生成完成，共生成 {len(saved_paths)} 张图像")
        for path in saved_paths:
            print(f"  - {path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试完整流程：图像 + 视频"""
    print("=" * 60)
    print("测试 lingjie/5.json - 完整流程")
    print("=" * 60)
    
    script_path = Path(__file__).parent.parent / "lingjie" / "5.json"
    
    if not script_path.exists():
        print(f"✗ 脚本文件不存在: {script_path}")
        return False
    
    print(f"\n脚本路径: {script_path}")
    
    try:
        # 初始化完整流水线
        print("\n[1] 初始化 AIVideoPipeline...")
        pipeline = AIVideoPipeline(
            "config.yaml",
            load_image=True,
            load_video=True,
            load_tts=True,  # 启用 TTS
            load_subtitle=True,  # 启用字幕
            load_composer=True,  # 启用合成
        )
        print("✓ AIVideoPipeline 初始化成功")
        
        # 处理脚本
        print("\n[2] 开始处理脚本...")
        output_name = "lingjie_ep5_full_test"
        pipeline.process_script(str(script_path), output_name)
        
        print(f"\n✓ 处理完成，输出目录: outputs/{output_name}")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_animatediff_sdxl():
    """测试 AnimateDiff-SDXL 视频生成"""
    print("=" * 60)
    print("测试 AnimateDiff-SDXL 视频生成")
    print("=" * 60)
    
    script_path = Path(__file__).parent.parent / "lingjie" / "5.json"
    
    if not script_path.exists():
        print(f"✗ 脚本文件不存在: {script_path}")
        return False
    
    print(f"\n脚本路径: {script_path}")
    
    try:
        # 加载脚本
        with open(script_path, 'r', encoding='utf-8') as f:
            script = json.load(f)
        
        scenes = script.get('scenes', [])
        if not scenes:
            print("✗ 脚本中没有场景")
            return False
        
        # 使用第一个场景的图像进行测试
        first_scene = scenes[0]
        image_path = first_scene.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            print(f"✗ 场景图像不存在: {image_path}")
            print("  提示: 请先运行 --stage 1 生成图像")
            return False
        
        print(f"\n使用测试图像: {image_path}")
        
        # 创建临时配置文件用于 AnimateDiff
        print("\n[1] 创建 AnimateDiff 配置...")
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 备份原始配置
        original_model_type = config['video']['model_type']
        original_model_path = config['video']['model_path']
        original_num_frames = config['video']['num_frames']
        original_fps = config['video']['fps']
        original_width = config['video']['width']
        original_height = config['video']['height']
        
        # 更新为 AnimateDiff 配置
        config['video']['model_type'] = 'animatediff-sdxl'
        config['video']['model_path'] = str(Path(__file__).parent / 'models' / 'animatediff-sdxl-1080p')
        config['video']['num_frames'] = 64
        config['video']['fps'] = 16
        config['video']['width'] = 1920
        config['video']['height'] = 1080
        config['video']['num_inference_steps'] = 50
        config['video']['guidance_scale'] = 7.5
        
        # 添加 AnimateDiff 特定配置
        if 'animatediff' not in config['video']:
            config['video']['animatediff'] = {}
        config['video']['animatediff']['use_freeinit'] = False  # 先测试基础功能
        config['video']['animatediff']['motion_module'] = str(Path(__file__).parent / 'models' / 'animatediff-sdxl-1080p' / 'mm_sdxl_v10_beta.ckpt')
        
        # 保存临时配置
        temp_config_path = Path(__file__).parent / "config_animatediff_test.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        print(f"✓ 临时配置文件已创建: {temp_config_path}")
        
        # 初始化视频生成器（使用 AnimateDiff）
        print("\n[2] 初始化 AnimateDiff-SDXL 视频生成器...")
        try:
            video_gen = VideoGenerator(str(temp_config_path))
            print("✓ VideoGenerator 初始化成功")
        except Exception as e:
            print(f"✗ VideoGenerator 初始化失败: {e}")
            print("  提示: 可能需要先实现 AnimateDiff 支持代码")
            import traceback
            traceback.print_exc()
            return False
        
        # 生成测试视频
        print("\n[3] 开始生成 AnimateDiff-SDXL 视频...")
        output_dir = Path(__file__).parent / "outputs" / "animatediff_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_animatediff.mp4"
        
        try:
            video_gen.generate_video(
                image_path,
                str(output_path),
                num_frames=64,
                fps=16,
                scene=first_scene,
            )
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"\n✓ AnimateDiff-SDXL 视频生成成功!")
                print(f"  输出路径: {output_path}")
                print(f"  文件大小: {file_size:.2f} MB")
                return True
            else:
                print(f"\n✗ 视频文件未生成: {output_path}")
                return False
                
        except Exception as e:
            print(f"\n✗ 视频生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理临时配置文件
            if temp_config_path.exists():
                temp_config_path.unlink()
                print(f"\n✓ 已清理临时配置文件")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 lingjie/5.json")
    parser.add_argument(
        "--stage",
        choices=["1", "full", "animatediff"],
        default="1",
        help="测试阶段：1=只生成图像, full=完整流程（图像+视频）, animatediff=测试AnimateDiff-SDXL"
    )
    
    args = parser.parse_args()
    
    if args.stage == "1":
        success = test_stage1_only()
    elif args.stage == "full":
        success = test_full_pipeline()
    elif args.stage == "animatediff":
        success = test_animatediff_sdxl()
    else:
        print(f"✗ 未知的测试阶段: {args.stage}")
        success = False
    
    if success:
        print("\n" + "=" * 60)
        print("✓ 测试完成")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ 测试失败")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

