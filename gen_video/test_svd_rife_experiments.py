#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVD + RIFE 对比实验脚本
直接测试视频生成部分，跳过TTS和完整流程
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from image_generator import ImageGenerator
from video_generator import VideoGenerator

def run_experiment(config_path: str, experiment_json: str, output_dir: str, rife_scale: float = 2.0):
    """运行单个实验"""
    print("=" * 60)
    print(f"SVD + RIFE 实验: interpolation_scale = {rife_scale}")
    print("=" * 60)
    
    # 加载实验JSON
    with open(experiment_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get('scenes', [])
    print(f"共 {len(scenes)} 个测试场景\n")
    
    # 加载生成器
    print("加载图像和视频生成器...")
    image_generator = ImageGenerator(config_path)
    video_generator = VideoGenerator(config_path)
    
    # 修改RIFE配置（临时）
    if hasattr(video_generator, 'video_config'):
        video_generator.video_config['rife'] = {
            'enabled': rife_scale > 0,
            'interpolation_scale': rife_scale if rife_scale > 0 else 2.0
        }
        if rife_scale > 0:
            video_generator.rife_enabled = True
            video_generator.rife_interpolation_scale = rife_scale
        else:
            video_generator.rife_enabled = False
    
    # 预加载pipeline（避免首次生成时加载）
    print("预加载图像生成pipeline...")
    try:
        image_generator.load_pipeline()
        print("✓ 图像生成pipeline已加载")
    except Exception as e:
        print(f"⚠ 图像生成pipeline加载失败: {e}")
    
    print("✓ 生成器加载完成\n")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    videos_dir = output_path / "videos"
    images_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    
    # 处理每个场景
    for i, scene in enumerate(scenes):
        scene_id = scene.get('id', i)
        test_profile = scene.get('test_profile', 'unknown')
        description = scene.get('description', '')[:50]
        
        print(f"[{i+1}/{len(scenes)}] 场景 {scene_id} ({test_profile})")
        print(f"  描述: {description}...")
        
        try:
            # 生成图像
            image_path = images_dir / f"scene_{scene_id:03d}.png"
            prompt = scene.get('prompt', '') or scene.get('description', '')
            
            print(f"  1/2 生成图像...")
            image_generator.generate_image(
                prompt=prompt,
                output_path=str(image_path),
                scene=scene
            )
            
            if not image_path.exists():
                print(f"  ✗ 图像生成失败，跳过")
                continue
            
            # 生成视频
            video_path = videos_dir / f"scene_{scene_id:03d}_scale{rife_scale}.mp4"
            duration = scene.get('duration', 4.0)
            
            print(f"  2/2 生成视频 (时长: {duration:.1f}秒, RIFE scale: {rife_scale})...")
            scene_with_duration = scene.copy()
            scene_with_duration['duration'] = duration
            
            # 为了节省显存，降低帧数和decode_chunk_size
            # 临时修改配置以降低显存需求
            original_decode_chunk_size = video_generator.video_config.get('decode_chunk_size', 8)
            
            # 降低帧数：从120降到48（2秒@24fps），减少显存需求
            # 直接传递num_frames参数，避免被场景分析逻辑覆盖
            target_num_frames = 48
            # 降低decode_chunk_size：从8降到4，减少显存需求
            video_generator.video_config['decode_chunk_size'] = 4
            
            try:
                video_generator.generate_video(
                    image_path=str(image_path),
                    output_path=str(video_path),
                    num_frames=target_num_frames,  # 直接传递，避免被覆盖
                    scene=scene_with_duration
                )
            finally:
                # 恢复原始配置
                video_generator.video_config['decode_chunk_size'] = original_decode_chunk_size
            
            if video_path.exists():
                print(f"  ✓ 完成: {video_path.name}\n")
            else:
                print(f"  ✗ 视频生成失败\n")
                
        except Exception as e:
            print(f"  ✗ 错误: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print("=" * 60)
    print(f"实验完成！输出目录: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SVD + RIFE 对比实验')
    parser.add_argument('--config', default='gen_video/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--json', default='gen_video/temp/svd_rife_experiments.json',
                       help='实验JSON文件')
    parser.add_argument('--output', required=True,
                       help='输出目录')
    parser.add_argument('--rife-scale', type=float, default=2.0,
                       help='RIFE插帧倍数 (2.0, 1.5, 或 0 表示关闭)')
    
    args = parser.parse_args()
    
    run_experiment(
        config_path=args.config,
        experiment_json=args.json,
        output_dir=args.output,
        rife_scale=args.rife_scale
    )

