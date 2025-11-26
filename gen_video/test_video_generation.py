#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视频生成优化
使用已有图像测试视频生成，验证优化后的参数
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_video_generation():
    """测试视频生成（使用已有图像）"""
    print("=" * 60)
    print("测试视频生成优化")
    print("=" * 60)
    
    try:
        from video_generator import VideoGenerator
        
        print("\n[1] 初始化 VideoGenerator...")
        gen = VideoGenerator("config.yaml")
        print("✓ VideoGenerator 初始化成功")
        
        print("\n[2] 加载视频生成模型...")
        gen.load_model()
        print("✓ 视频生成模型加载成功")
        
        # 查找一个已有的图像用于测试
        test_image_path = None
        possible_paths = [
            "outputs/images/lingjie_ep11_full/scene_001.png",
            "outputs/images/lingjie_ep11_full/scene_007.png",
            "outputs/images/lingjie_ep11_full/scene_008.png",
            "outputs/images/lingjie_ep1_full/scene_001.png",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                test_image_path = path
                break
        
        if test_image_path is None:
            print("⚠ 未找到测试图像，请先运行图像生成")
            print("  或者手动指定一个图像路径")
            return False
        
        print(f"\n[3] 使用测试图像: {test_image_path}")
        
        # 创建输出目录
        output_dir = Path("outputs/test_video")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_video_optimized.mp4"
        
        print(f"\n[4] 生成视频（使用优化后的参数）...")
        print(f"   输出路径: {output_path}")
        print(f"   优化参数:")
        print(f"     - num_inference_steps: {gen.video_config.get('num_inference_steps', 60)}")
        print(f"     - fps: {gen.video_config.get('fps', 15)}")
        print(f"     - decode_chunk_size: {gen.video_config.get('decode_chunk_size', 10)}")
        print(f"     - motion_bucket_id: {gen.video_config.get('motion_bucket_id', 2)}")
        print(f"     - noise_aug_strength: {gen.video_config.get('noise_aug_strength', 0.0004)}")
        
        # 生成视频
        result_path = gen.generate_video(
            test_image_path,
            str(output_path),
            num_frames=None,  # 使用配置中的默认值
            fps=None,  # 使用配置中的默认值
        )
        
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
            print(f"\n✓ 视频生成成功: {result_path}")
            print(f"  文件大小: {file_size:.2f} MB")
            
            # 获取视频信息
            try:
                import subprocess
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,r_frame_rate", 
                     "-of", "default=noprint_wrappers=1", str(result_path)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"\n  视频信息:")
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            print(f"    {line}")
            except Exception as e:
                print(f"  ⚠ 无法获取视频详细信息: {e}")
            
            return True
        else:
            print(f"✗ 视频文件未生成: {result_path}")
            return False
            
    except Exception as e:
        print(f"✗ 视频生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_video_generation()
    if success:
        print("\n" + "=" * 60)
        print("✓ 视频生成测试完成")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ 视频生成测试失败")
        print("=" * 60)
        sys.exit(1)

