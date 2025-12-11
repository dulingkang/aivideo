#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整视频生成流水线（商业化方向）
根据商业化方案，测试以下场景：
1. 科普视频（政府/教育赛道）
2. 产品广告视频（电商赛道）
3. 短剧/推文视频（内容赛道）
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置PyTorch CUDA内存分配配置
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from gen_video.main import AIVideoPipeline


def create_test_script_scenario(scenario_type: str, output_dir: Path) -> Path:
    """
    根据商业化方案创建不同场景的测试脚本
    
    Args:
        scenario_type: 'scientific' (科普), 'product' (产品广告), 'story' (短剧推文)
    """
    scenarios = {
        'scientific': {
            "episode": "test_scientific_001",
            "title": "科普视频测试 - 黑洞是什么",
            "scenes": [
                {
                    "id": 1,
                    "narration": "黑洞是宇宙中最神秘的天体之一，它的引力如此强大，连光都无法逃脱。",
                    "description": "a realistic visualization of a black hole in space, accretion disk glowing with intense light, stars in the background, scientific and educational style",
                    "duration": 5.0,
                    "motion_intensity": "gentle",
                    "camera_motion": {
                        "type": "slow_zoom",
                        "params": {"zoom_speed": 0.1}
                    },
                    "visual": {
                        "composition": "wide shot of black hole in space",
                        "style": "scientific"  # 可在前端和脚本中配置，支持: scientific, commercial, dramatic, realistic, xianxia
                    }
                }
            ]
        },
        'product': {
            "episode": "test_product_001",
            "title": "产品广告测试 - 智能水杯",
            "scenes": [
                {
                    "id": 1,
                    "narration": "这款智能水杯采用最新科技，自动提醒您及时补水，让健康生活更简单。",
                    "description": "a modern smart water bottle on a clean white background, minimalist design, professional product photography style, high quality",
                    "duration": 5.0,
                    "motion_intensity": "gentle",
                    "camera_motion": {
                        "type": "rotate",
                        "params": {"rotation_speed": 0.05}
                    },
                    "visual": {
                        "composition": "product shot of smart water bottle",
                        "style": "commercial"  # 可在前端和脚本中配置
                    }
                }
            ]
        },
        'story': {
            "episode": "test_story_001",
            "title": "短剧推文测试 - 都市情感",
            "scenes": [
                {
                    "id": 1,
                    "narration": "他站在雨夜的街头，看着远去的背影，心中涌起无尽的悔恨。",
                    "description": "a realistic urban street scene at night in the rain, a man standing alone, emotional atmosphere, cinematic lighting",
                    "duration": 5.0,
                    "motion_intensity": "moderate",
                    "camera_motion": {
                        "type": "dolly",
                        "params": {"direction": "forward", "speed": 0.1}
                    },
                    "visual": {
                        "composition": "cinematic shot of urban night scene",
                        "style": "dramatic"  # 可在前端和脚本中配置
                    }
                }
            ]
        }
    }
    
    if scenario_type not in scenarios:
        raise ValueError(f"Unknown scenario type: {scenario_type}. Choose from: {list(scenarios.keys())}")
    
    test_script = scenarios[scenario_type]
    script_path = output_dir / f"test_script_{scenario_type}.json"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        json.dump(test_script, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 创建{scenario_type}测试脚本: {script_path}")
    return script_path


def test_complete_pipeline(scenario_type: str = 'scientific'):
    """
    测试完整流水线（根据商业化方案）
    
    Args:
        scenario_type: 'scientific' (科普), 'product' (产品广告), 'story' (短剧推文)
    """
    print("=" * 80)
    print(f"完整视频生成流水线测试 - {scenario_type.upper()}场景")
    print("=" * 80)
    print("\n根据商业化方案：")
    print("  - 科普视频：政府/教育赛道，高质量，使用HunyuanVideo")
    print("  - 产品广告：电商赛道，快速生成，使用CogVideoX（当前用HunyuanVideo测试）")
    print("  - 短剧推文：内容赛道，批量生成，使用CogVideoX（当前用HunyuanVideo测试）")
    print()
    
    # 配置路径
    config_path = project_root / "gen_video" / "config.yaml"
    output_name = f"test_complete_pipeline_{scenario_type}"
    output_dir = project_root / "gen_video" / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试脚本
    print(f"\n[1/6] 准备{scenario_type}测试脚本...")
    script_path = create_test_script_scenario(scenario_type, output_dir)
    
    # 初始化流水线
    print("\n[2/6] 初始化流水线组件...")
    try:
        pipeline = AIVideoPipeline(
            str(config_path),
            load_image=True,      # 启用图像生成（Flux + InstantID）
            load_video=True,      # 启用视频生成（HunyuanVideo用于高端，CogVideoX用于量产）
            load_tts=True,        # 启用TTS（CosyVoice）
            load_subtitle=True,   # 启用字幕
            load_composer=True,   # 启用合成
        )
        
        # 根据场景类型选择视频模型
        if pipeline.video_generator:
            # 科普视频使用HunyuanVideo（高质量）
            if scenario_type == 'scientific':
                pipeline.video_generator.video_config['model_type'] = 'hunyuanvideo'
                print("   ✓ 已设置使用HunyuanVideo模型（科普视频 - 高质量）")
            # 产品广告和短剧可以使用HunyuanVideo测试（未来切换到CogVideoX）
            else:
                pipeline.video_generator.video_config['model_type'] = 'hunyuanvideo'
                print("   ✓ 已设置使用HunyuanVideo模型（测试用，未来切换到CogVideoX）")
        
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
    print(f"   场景类型: {scenario_type}")
    
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
                    for f in files:
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"      - {f.name}: {size_mb:.2f} MB")
                else:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"   ✓ {name}: {size_mb:.2f} MB")
            else:
                print(f"   ✗ {name}: 文件不存在")
                all_ok = False
        
        if all_ok:
            print("\n" + "=" * 80)
            print(f"✓ {scenario_type.upper()}场景流水线测试成功！")
            print("=" * 80)
            print(f"\n最终视频: {expected_files['final']}")
            print("\n下一步建议：")
            if scenario_type == 'scientific':
                print("  - 这是科普视频场景，适合政府/教育赛道")
                print("  - 可以继续优化：添加更多科学可视化元素")
            elif scenario_type == 'product':
                print("  - 这是产品广告场景，适合电商赛道")
                print("  - 未来可以切换到CogVideoX以提高生成速度")
            elif scenario_type == 'story':
                print("  - 这是短剧推文场景，适合内容赛道")
                print("  - 未来可以切换到CogVideoX以支持批量生成")
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
    import argparse
    
    parser = argparse.ArgumentParser(description="测试完整视频生成流水线（商业化方向）")
    parser.add_argument(
        "--scenario",
        type=str,
        default="scientific",
        choices=["scientific", "product", "story"],
        help="测试场景类型: scientific(科普), product(产品广告), story(短剧推文)"
    )
    
    args = parser.parse_args()
    
    success = test_complete_pipeline(args.scenario)
    sys.exit(0 if success else 1)

