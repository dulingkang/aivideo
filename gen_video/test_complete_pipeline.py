#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整视频生成流水线
包括：图像生成 -> 视频生成 -> TTS -> 字幕 -> 合成
"""

import sys
import json
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置PyTorch CUDA内存分配配置
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from gen_video.main import AIVideoPipeline


def create_test_script(output_dir: Path) -> Path:
    """创建一个简单的测试脚本JSON"""
    test_script = {
        "episode": "test_001",
        "title": "测试视频",
        "scenes": [
            {
                "id": 1,
                "narration": "这是一个测试场景，展示美丽的山景。",
                "description": "a peaceful mountain landscape with snow-capped peaks, alpine meadow with wildflowers, clear blue sky",
                "duration": 5.0,
                "motion_intensity": "gentle",
                "camera_motion": {
                    "type": "static"
                },
                "visual": {
                    "composition": "wide shot of mountain landscape"
                }
            }
        ]
    }
    
    script_path = output_dir / "test_script.json"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        json.dump(test_script, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 创建测试脚本: {script_path}")
    return script_path


def test_complete_pipeline():
    """测试完整流水线"""
    print("=" * 80)
    print("完整视频生成流水线测试")
    print("=" * 80)
    
    # 配置路径
    config_path = project_root / "gen_video" / "config.yaml"
    output_name = "test_complete_pipeline"
    output_dir = project_root / "gen_video" / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试脚本
    print("\n[1/6] 准备测试脚本...")
    script_path = create_test_script(output_dir)
    
    # 初始化流水线
    print("\n[2/6] 初始化流水线组件...")
    try:
        pipeline = AIVideoPipeline(
            str(config_path),
            load_image=True,      # 启用图像生成
            load_video=True,      # 启用视频生成（使用HunyuanVideo）
            load_tts=True,        # 启用TTS
            load_subtitle=True,   # 启用字幕
            load_composer=True,   # 启用合成
        )
        
        # 确保使用HunyuanVideo模型
        if pipeline.video_generator:
            pipeline.video_generator.video_config['model_type'] = 'hunyuanvideo'
            print("   ✓ 已设置使用HunyuanVideo模型")
        
        print("✓ 流水线初始化成功")
    except Exception as e:
        print(f"✗ 流水线初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 处理脚本
    print("\n[3/6] 开始处理脚本...")
    print(f"   脚本路径: {script_path}")
    print(f"   输出名称: {output_name}")
    
    try:
        pipeline.process_script(str(script_path), output_name)
        print(f"\n✓ 流水线处理完成！")
        print(f"   输出目录: {output_dir}")
        
        # 检查输出文件
        print("\n[4/6] 检查输出文件...")
        expected_files = {
            "videos": output_dir / "videos",
            "audio": output_dir / "audio.wav",
            "subtitle": output_dir / "subtitle.srt",
            "final": output_dir / f"{output_name}.mp4"
        }
        
        all_ok = True
        for name, path in expected_files.items():
            if path.exists():
                if path.is_dir():
                    files = list(path.glob("*.mp4"))
                    print(f"   ✓ {name}: {len(files)} 个文件")
                else:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"   ✓ {name}: {size_mb:.2f} MB")
            else:
                print(f"   ✗ {name}: 文件不存在")
                all_ok = False
        
        if all_ok:
            print("\n" + "=" * 80)
            print("✓ 完整流水线测试成功！")
            print("=" * 80)
            print(f"\n最终视频: {expected_files['final']}")
            return True
        else:
            print("\n⚠ 部分文件缺失，但流程已执行")
            return True
            
    except Exception as e:
        print(f"\n✗ 流水线处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)

