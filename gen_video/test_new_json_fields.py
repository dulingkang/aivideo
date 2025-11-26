#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新 JSON 字段集成
"""

import json
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from face_style_auto_generator import (
    auto_face_style_for_scene,
    generate_face_styles_for_episode,
    to_instantid_params
)


def test_face_style_auto():
    """测试 face_style_auto 自动生成"""
    print("=" * 60)
    print("测试 face_style_auto 自动生成")
    print("=" * 60)
    
    # 测试场景
    test_scenes = [
        {
            "id": 1,
            "mood": "serious",
            "lighting": "day",
            "action": "walking_forward",
            "camera": "wide_shot_low_angle"
        },
        {
            "id": 2,
            "mood": "alert",
            "lighting": "day",
            "action": "detect_spiritual_fluctuation",
            "camera": "medium_shot_front"
        },
        {
            "id": 3,
            "mood": "calm",
            "lighting": "night",
            "action": "meditate",
            "camera": "close_up"
        }
    ]
    
    print("\n原始场景:")
    for scene in test_scenes:
        print(f"  场景 {scene['id']}: mood={scene['mood']}, action={scene['action']}")
    
    # 生成 face_style_auto
    styles = generate_face_styles_for_episode(test_scenes, smooth=True, overwrite_existing=False)
    
    print("\n生成的 face_style_auto:")
    for scene, style in zip(test_scenes, styles):
        print(f"  场景 {scene['id']}:")
        print(f"    expression: {style['expression']}")
        print(f"    lighting: {style['lighting']}")
        print(f"    detail: {style['detail']}")
        print(f"    strength: {style['strength']}")
        
        # 转换为 InstantID 参数
        params = to_instantid_params(style)
        print(f"    -> InstantID 参数:")
        print(f"       ip_adapter_scale_multiplier: {params['ip_adapter_scale_multiplier']:.2f}")
        print(f"       face_kps_scale_multiplier: {params['face_kps_scale_multiplier']:.2f}")


def test_visual_fields():
    """测试 visual 字段解析"""
    print("\n" + "=" * 60)
    print("测试 visual 字段解析")
    print("=" * 60)
    
    from image_generator import ImageGenerator
    
    gen = ImageGenerator("config.yaml")
    
    # 测试场景（包含 visual 字段）
    test_scene = {
        "id": 1,
        "description": "韩立踏入青罗沙漠",
        "visual": {
            "composition": "Han Li small silhouette vs vast golden desert",
            "environment": "rolling sand waves, intense sunlight, heat distortion",
            "character_pose": "steady forward walk, robe slightly fluttering",
            "fx": "subtle heat haze, drifting sand",
            "motion": "slow dolly-forward shot"
        },
        "camera": "wide_shot_low_angle"
    }
    
    print("\n测试场景:")
    print(json.dumps(test_scene, ensure_ascii=False, indent=2))
    
    # 构建 prompt
    prompt = gen.build_prompt(test_scene)
    
    print("\n生成的 prompt:")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)


def test_duration_field():
    """测试 duration 字段"""
    print("\n" + "=" * 60)
    print("测试 duration 字段")
    print("=" * 60)
    
    # 测试场景
    test_scene = {
        "id": 1,
        "duration": 5,
        "description": "测试场景"
    }
    
    # 模拟视频生成参数计算
    fps = 12
    if test_scene.get("duration"):
        duration = test_scene["duration"]
        num_frames = int(duration * fps)
        print(f"\n场景 duration: {duration} 秒")
        print(f"fps: {fps}")
        print(f"计算得到的帧数: {num_frames} 帧")
        print(f"视频时长: {num_frames / fps:.2f} 秒")


def test_full_json():
    """测试完整 JSON 文件"""
    print("\n" + "=" * 60)
    print("测试完整 JSON 文件")
    print("=" * 60)
    
    json_path = Path(__file__).parent.parent / "lingjie" / "5.json"
    
    if not json_path.exists():
        print(f"⚠ JSON 文件不存在: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    print(f"\n找到 {len(scenes)} 个场景")
    
    # 检查字段
    for scene in scenes[:3]:  # 只检查前3个
        print(f"\n场景 {scene.get('id')}:")
        print(f"  duration: {scene.get('duration', 'N/A')}")
        print(f"  visual: {bool(scene.get('visual'))}")
        print(f"  face_style_auto: {bool(scene.get('face_style_auto'))}")
        
        if scene.get("visual"):
            visual = scene["visual"]
            print(f"    composition: {visual.get('composition', 'N/A')[:50]}...")
            print(f"    motion: {visual.get('motion', 'N/A')}")


if __name__ == "__main__":
    test_face_style_auto()
    test_visual_fields()
    test_duration_field()
    test_full_json()
    
    print("\n" + "=" * 60)
    print("✓ 测试完成")
    print("=" * 60)

