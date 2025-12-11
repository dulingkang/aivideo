#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用映射关系重新分析测试场景图像质量
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from test_scenes_quality import analyze_image_quality, generate_quality_report, load_test_scenes


def load_mapping() -> Dict[str, int]:
    """加载场景和图像的映射关系"""
    mapping_file = Path(__file__).parent / "outputs" / "test_scenes_quality" / "scene_image_mapping.json"
    
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("suggested_mapping", data.get("mapping", {}))
    else:
        print(f"⚠️  警告: 映射文件不存在: {mapping_file}")
        print("   将使用默认映射（scene_XXX.png -> 场景ID XXX-1）")
        return {}


def main():
    project_root = Path(__file__).parent
    test_json_path = project_root / "renjie" / "episode" / "test_scenes.json"
    output_dir = project_root / "outputs" / "test_scenes_quality"
    images_dir = output_dir / "images"
    
    print("=" * 80)
    print("🔍 使用映射关系重新分析测试场景图像质量")
    print("=" * 80)
    print()
    
    # 加载场景
    test_data = load_test_scenes(test_json_path)
    scenes = test_data.get("scenes", [])
    
    # 加载映射关系
    mapping = load_mapping()
    print(f"📋 加载映射关系: {len(mapping)} 个映射")
    if mapping:
        print("   映射关系:")
        for img_name, scene_id in sorted(mapping.items()):
            scene = scenes[scene_id] if scene_id < len(scenes) else None
            desc = scene.get("description", "")[:40] if scene else "未知"
            print(f"     {img_name} -> 场景ID {scene_id}: {desc}...")
    print()
    
    # 列出图像文件
    image_files = sorted(images_dir.glob("scene_*.png"))
    print(f"🖼️  找到 {len(image_files)} 个图像文件\n")
    
    # 分析质量
    print("=" * 80)
    print("📊 分析图像质量（使用映射关系）...")
    print("=" * 80)
    
    analysis_results = []
    for img_file in image_files:
        # 使用映射关系找到对应的场景ID
        scene_id = mapping.get(img_file.name, -1)
        
        if scene_id < 0:
            # 如果没有映射，尝试从文件名推断
            file_num = int(img_file.stem.split('_')[1])
            scene_id = file_num - 1
        
        if scene_id < 0 or scene_id >= len(scenes):
            print(f"⚠️  警告: 无法为 {img_file.name} 找到对应的场景")
            continue
        
        scene = scenes[scene_id]
        result = analyze_image_quality(scene, img_file)
        result["image_file"] = img_file.name
        result["mapped_scene_id"] = scene_id
        analysis_results.append(result)
        
        status_icon = "✅" if result["quality_score"] >= 70 else "⚠️" if result["quality_score"] >= 60 else "❌"
        print(f"{status_icon} {img_file.name} -> 场景ID {scene_id}: {result['quality_score']}分")
        print(f"   描述: {result['description'][:40]}...")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"   ⚠️  {issue}")
    
    # 按场景ID排序
    analysis_results.sort(key=lambda x: x["scene_id"])
    
    # 生成报告
    print("\n" + "=" * 80)
    print("📝 生成质量评估报告...")
    print("=" * 80)
    
    report_path = generate_quality_report(analysis_results, output_dir)
    
    # 更新报告，添加映射信息
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    # 在报告开头添加映射说明
    mapping_note = f"""
## ⚠️ 重要说明：场景和图像对应关系

由于图像生成时场景顺序可能出现错乱，已使用手动映射关系：

"""
    for img_name, scene_id in sorted(mapping.items()):
        if scene_id < len(scenes):
            scene = scenes[scene_id]
            mapping_note += f"- **{img_name}** -> 场景ID {scene_id}: {scene.get('description', '')[:50]}...\n"
    
    mapping_note += "\n"
    
    # 插入映射说明到报告开头（在"总体评估"之前）
    if "## 📊 总体评估" in report_content:
        report_content = report_content.replace("## 📊 总体评估", mapping_note + "## 📊 总体评估")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("\n" + "=" * 80)
    print("✅ 重新分析完成！")
    print("=" * 80)
    print(f"📊 评估报告: {report_path}")
    print()
    
    # 显示总体评估
    total_scenes = len(analysis_results)
    character_scenes = [r for r in analysis_results if r["has_character"]]
    front_view_count = sum(1 for r in character_scenes if r["is_front_view"])
    object_correct_count = sum(1 for r in analysis_results if r["object_correct"])
    avg_quality = sum(r["quality_score"] for r in analysis_results) / total_scenes if total_scenes > 0 else 0
    
    print(f"📊 总体评估:")
    print(f"   平均质量分数: {avg_quality:.1f}/100")
    if len(character_scenes) > 0:
        print(f"   正面视角率: {front_view_count}/{len(character_scenes)} ({front_view_count/len(character_scenes)*100:.1f}%)")
    print(f"   物体识别正确率: {object_correct_count}/{total_scenes} ({object_correct_count/total_scenes*100:.1f}%)")
    print()
    
    if avg_quality >= 70:
        print(f"✅ 质量达到可用标准，可以继续开发MVP")
    else:
        print(f"⚠️  质量未达可用标准，建议先优化效果")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

