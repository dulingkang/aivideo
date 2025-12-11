#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HunyuanVideo集成测试脚本
测试HunyuanVideo视频生成功能
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gen_video.video_generator import VideoGenerator
from PIL import Image
import numpy as np

def test_hunyuanvideo():
    """测试HunyuanVideo视频生成"""
    print("=" * 60)
    print("HunyuanVideo集成测试")
    print("=" * 60)
    
    # 创建测试图像
    print("\n1. 创建测试图像...")
    test_image_path = project_root / "gen_video" / "outputs" / "test_hunyuanvideo_input.png"
    test_image_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建一个简单的测试图像（如果不存在）
    if not test_image_path.exists():
        test_image = Image.new('RGB', (1280, 768), color=(100, 150, 200))
        test_image.save(test_image_path)
        print(f"  ✓ 创建测试图像: {test_image_path}")
    else:
        print(f"  ✓ 使用现有测试图像: {test_image_path}")
    
    # 初始化VideoGenerator
    print("\n2. 初始化VideoGenerator...")
    config_path = project_root / "gen_video" / "config.yaml"
    
    # 读取并修改配置
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置为hunyuanvideo
    config['video']['model_type'] = 'hunyuanvideo'
    print(f"  ✓ 模型类型: {config['video']['model_type']}")
    
    # 临时保存修改后的配置
    import tempfile
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config, allow_unicode=True)
    temp_config_path = temp_config.name
    temp_config.close()
    
    generator = VideoGenerator(temp_config_path)
    
    # 加载模型
    print("\n4. 加载HunyuanVideo模型...")
    try:
        generator.load_model()
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        print("\n可能的原因:")
        print("  1. 模型未下载（首次使用会从HuggingFace下载）")
        print("  2. 显存不足（HunyuanVideo需要18-24GB显存）")
        print("  3. 依赖未安装")
        import traceback
        traceback.print_exc()
        return False
    
    # 生成视频
    print("\n5. 生成视频...")
    output_path = project_root / "gen_video" / "outputs" / "test_hunyuanvideo_output.mp4"
    
    try:
        result = generator.generate_video(
            image_path=str(test_image_path),
            output_path=str(output_path),
            num_frames=60,  # 测试用较少帧数
            fps=24,
        )
        print(f"  ✓ 视频生成成功: {result}")
        return True
    except Exception as e:
        print(f"  ✗ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hunyuanvideo()
    if success:
        print("\n" + "=" * 60)
        print("✅ HunyuanVideo集成测试通过！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ HunyuanVideo集成测试失败")
        print("=" * 60)
        sys.exit(1)

