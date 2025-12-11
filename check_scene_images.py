#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查场景和图像对应关系的工具脚本
通过查看JSON中的场景描述和提示词，帮助确认每个图像对应的场景
"""

import json
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    test_json_path = project_root / "renjie" / "episode" / "test_scenes.json"
    images_dir = project_root / "outputs" / "test_scenes_quality" / "images"
    
    # 加载场景
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    scenes = test_data.get("scenes", [])
    image_files = sorted(images_dir.glob("scene_*.png"))
    
    print("=" * 80)
    print("📋 场景和图像对应关系检查")
    print("=" * 80)
    print()
    
    print("场景配置（从JSON）：")
    print("-" * 80)
    for idx, scene in enumerate(scenes):
        scene_id = scene.get("id", idx)
        description = scene.get("description", "")[:50]
        prompt = scene.get("prompt", "")[:80]
        has_character = "韩立" in description or "Han Li" in prompt
        has_desert = "沙地" in description or "沙漠" in description or "sand" in prompt.lower()
        has_forest = "山林" in description or "forest" in prompt.lower()
        has_scroll = "卷轴" in description or "scroll" in prompt.lower()
        has_city = "城市" in description or "city" in prompt.lower()
        
        print(f"场景ID {scene_id} (数组索引{idx}):")
        print(f"  描述: {description}")
        print(f"  提示词: {prompt}")
        print(f"  特征: ", end="")
        features = []
        if has_character:
            features.append("有人物(韩立)")
        if has_desert:
            features.append("沙漠/沙地")
        if has_forest:
            features.append("山林")
        if has_scroll:
            features.append("卷轴")
        if has_city:
            features.append("城市")
        print(", ".join(features) if features else "无特殊特征")
        print()
    
    print("=" * 80)
    print("图像文件列表：")
    print("-" * 80)
    for img_file in image_files:
        file_num = int(img_file.stem.split('_')[1])
        array_idx = file_num - 1  # 文件名编号 = 数组索引 + 1
        print(f"{img_file.name} -> 应该对应数组索引{array_idx}的场景")
    print()
    
    print("=" * 80)
    print("🔍 需要手动检查的问题：")
    print("=" * 80)
    print()
    print("根据你的反馈：")
    print("  - scene_002.png 显示的是沙漠（但场景1应该是'韩立站在山林中'）")
    print()
    print("可能的原因：")
    print("  1. 图像生成时场景顺序错乱")
    print("  2. 场景ID和数组索引不一致")
    print("  3. JSON中的场景顺序与实际生成顺序不匹配")
    print()
    print("建议：")
    print("  1. 查看每个图像文件，确认实际内容")
    print("  2. 根据图像内容匹配对应的场景描述")
    print("  3. 如果顺序错乱，需要找出正确的对应关系")
    print()
    
    # 根据场景6应该是"韩立躺在沙地"，对应scene_007.png
    print("已知对应关系：")
    print("  - 场景6: '韩立躺在沙地' 应该对应 scene_007.png")
    print("  - 场景1: '韩立站在山林中' 应该对应 scene_002.png（但实际显示沙漠）")
    print()
    print("请检查以下图像的实际内容：")
    print("  - scene_001.png: 应该是卷轴（场景0）")
    print("  - scene_002.png: 应该是韩立站在山林中（场景1），但显示的是沙漠？")
    print("  - scene_007.png: 应该是韩立躺在沙地（场景6）")
    print()
    print("如果scene_002.png显示的是沙地，那么可能是场景6的图像")
    print("如果scene_007.png显示的是山林，那么可能是场景1的图像")
    print("（图像顺序可能完全错乱了）")
    print()

if __name__ == "__main__":
    main()

