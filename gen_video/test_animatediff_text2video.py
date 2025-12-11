#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AnimateDiff文生视频功能
"""

import os
import sys
from pathlib import Path

# 设置PyTorch CUDA内存分配配置
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_generator import VideoGenerator
import yaml


def test_animatediff_text2video():
    """测试AnimateDiff文生视频
    
    ⚠️ 警告：AnimateDiff在diffusers中有绿色竖条问题
    如果测试失败，建议使用SVD（修改config.yaml中model_type为svd-xt）
    """
    print("=" * 60)
    print("测试 AnimateDiff 文生视频")
    print("⚠️  警告：AnimateDiff在diffusers中可能有绿色竖条问题")
    print("=" * 60)
    
    # 创建临时配置文件
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新为AnimateDiff配置
    # ⚠️ 注意：AnimateDiff在diffusers中有绿色竖条问题，建议使用SVD
    config['video']['model_type'] = 'animatediff'
    config['video']['model_path'] = str(Path(__file__).parent / 'models' / 'animatediff-sdxl-1080p')
    
    # 尝试修复绿色竖条问题的参数（降低分辨率、增加步数）
    config['video']['num_frames'] = 16  # 减少帧数，降低复杂度
    config['video']['fps'] = 16
    config['video']['width'] = 512  # 降低分辨率，可能有助于避免绿色竖条
    config['video']['height'] = 512
    config['video']['num_inference_steps'] = 60  # 增加步数，提高稳定性
    config['video']['guidance_scale'] = 8.0  # 稍微提高guidance_scale
    
    # 关键修复：设置motion_scale（根据专业分析，这是导致绿色的最常见原因）
    # motion_scale应该在0.5-1.0之间，如果还是绿色，尝试降低到0.5-0.8
    if 'animatediff' not in config['video']:
        config['video']['animatediff'] = {}
    config['video']['animatediff']['motion_scale'] = 0.8  # 从1.0降低到0.8，尝试修复绿色问题
    
    # 保存临时配置
    temp_config_path = Path(__file__).parent / "config_animatediff_test.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"\n临时配置文件: {temp_config_path}")
    
    try:
        # 初始化视频生成器
        print("\n[1] 初始化 AnimateDiff 视频生成器...")
        video_gen = VideoGenerator(str(temp_config_path))
        video_gen.load_model()
        print("✓ 初始化成功")
        
        # 测试场景
        test_scene = {
            "description": "韩立站在沙漠中，远处是金色的沙丘，天空中有飘动的云彩",
            "prompt": "xianxia fantasy, Han Li standing in desert, golden sand dunes in distance, clouds drifting in sky",
            "characters": [{"name": "Han Li"}],
            "visual": {
                "composition": "wide shot, Han Li in center, vast desert landscape, golden sunset"
            }
        }
        
        # 生成视频
        print("\n[2] 开始生成视频...")
        output_dir = Path(__file__).parent.parent / "outputs" / "animatediff_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_animatediff_text2video.mp4"
        
        video_gen.generate_video(
            image_path=None,  # AnimateDiff不需要图像
            output_path=str(output_path),
            scene=test_scene,
        )
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"\n✓ 视频文件已生成!")
            print(f"  输出路径: {output_path}")
            print(f"  文件大小: {file_size:.2f} MB")
            print(f"\n⚠️  请检查视频是否有绿色竖条问题")
            print(f"  如果有绿色竖条，建议：")
            print(f"  1. 切换回SVD：修改config.yaml中model_type为svd-xt")
            print(f"  2. 或等待diffusers更新修复AnimateDiff问题")
            return True
        else:
            print(f"\n✗ 视频文件未生成: {output_path}")
            return False
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时配置文件
        if temp_config_path.exists():
            temp_config_path.unlink()
            print(f"\n✓ 已清理临时配置文件")


if __name__ == "__main__":
    success = test_animatediff_text2video()
    if not success:
        sys.exit(1)

