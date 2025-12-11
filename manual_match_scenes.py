#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动匹配场景和图像对应关系的工具
根据实际图像内容，手动指定每个图像对应的场景ID
"""

import json
from pathlib import Path
from typing import Dict, List

def load_scenes(json_path: Path) -> List[Dict]:
    """加载场景配置"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("scenes", [])

def print_scene_summary(scenes: List[Dict]):
    """打印场景摘要"""
    print("=" * 80)
    print("📋 场景配置摘要")
    print("=" * 80)
    for idx, scene in enumerate(scenes):
        scene_id = scene.get("id", idx)
        description = scene.get("description", "")
        prompt_keywords = []
        if "scroll" in scene.get("prompt", "").lower() or "卷轴" in description:
            prompt_keywords.append("卷轴")
        if "city" in scene.get("prompt", "").lower() or "城市" in description:
            prompt_keywords.append("城市")
        if "forest" in scene.get("prompt", "").lower() or "山林" in description:
            prompt_keywords.append("山林")
        if "sand" in scene.get("prompt", "").lower() or "沙地" in description:
            prompt_keywords.append("沙地")
        if "Han Li" in scene.get("prompt", "") or "韩立" in description:
            prompt_keywords.append("韩立")
        
        print(f"场景ID {scene_id}: {description[:50]}...")
        if prompt_keywords:
            print(f"  关键词: {', '.join(prompt_keywords)}")
        print()

def manual_matching():
    """手动匹配流程"""
    project_root = Path(__file__).parent
    test_json_path = project_root / "renjie" / "episode" / "test_scenes.json"
    images_dir = project_root / "outputs" / "test_scenes_quality" / "images"
    
    scenes = load_scenes(test_json_path)
    image_files = sorted(images_dir.glob("scene_*.png"))
    
    print("=" * 80)
    print("🔍 手动匹配场景和图像对应关系")
    print("=" * 80)
    print()
    print("说明：")
    print("  根据你查看的实际图像内容，为每个图像文件指定对应的场景ID")
    print("  如果图像内容与预期不符，可以手动指定正确的场景")
    print()
    
    print_scene_summary(scenes)
    
    print("=" * 80)
    print("🖼️  请根据实际图像内容，为每个图像指定对应的场景ID")
    print("=" * 80)
    print()
    
    # 默认映射（如果用户不修改）
    default_mapping = {
        "scene_001.png": 0,  # 应该是卷轴
        "scene_002.png": 1,  # 应该是山林，但显示沙漠
        "scene_003.png": 2,  # 应该是城市
        "scene_004.png": 3,  # 应该是近景
        "scene_005.png": 4,  # 应该是法术
        "scene_006.png": 5,  # 应该是山峰
        "scene_007.png": 6,  # 应该是沙地
    }
    
    # 根据用户反馈，scene_002.png显示的是沙漠，可能是场景6
    # scene_007.png如果显示的是山林，可能是场景1
    # 提供推测的映射
    suggested_mapping = {
        "scene_001.png": 0,  # 卷轴
        "scene_002.png": 6,  # 沙地（实际显示的内容）
        "scene_003.png": 2,  # 城市
        "scene_004.png": 3,  # 近景
        "scene_005.png": 4,  # 法术
        "scene_006.png": 5,  # 山峰
        "scene_007.png": 1,  # 山林（如果scene_002是场景6，那么scene_007可能是场景1）
    }
    
    print("📝 推测的映射关系（基于你的反馈）：")
    print("-" * 80)
    mapping = {}
    for img_file in image_files:
        scene_id = suggested_mapping.get(img_file.name, -1)
        if scene_id >= 0 and scene_id < len(scenes):
            scene = scenes[scene_id]
            description = scene.get("description", "")[:50]
            print(f"{img_file.name} -> 场景ID {scene_id}: {description}...")
            mapping[img_file.name] = scene_id
        else:
            print(f"{img_file.name} -> 未匹配")
    
    print()
    print("=" * 80)
    print("💾 保存映射关系到文件")
    print("=" * 80)
    
    # 保存映射
    mapping_file = project_root / "outputs" / "test_scenes_quality" / "scene_image_mapping.json"
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    
    mapping_data = {
        "mapping": mapping,
        "note": "手动匹配的场景和图像对应关系，基于实际图像内容",
        "default_mapping": default_mapping,
        "suggested_mapping": suggested_mapping
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 映射关系已保存到: {mapping_file}")
    print()
    print("📝 请检查映射是否正确：")
    print(f"   - scene_002.png -> 场景ID {mapping.get('scene_002.png')} (应该是场景6-沙地)")
    print(f"   - scene_007.png -> 场景ID {mapping.get('scene_007.png')} (可能是场景1-山林)")
    print()
    print("如果映射不正确，请手动编辑文件修改")
    print()

if __name__ == "__main__":
    manual_matching()

