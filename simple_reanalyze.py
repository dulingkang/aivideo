#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的重新分析脚本（不需要导入图像生成器）
直接基于映射关系分析场景质量
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def analyze_image_quality(scene: Dict, image_path: Path) -> Dict[str, Any]:
    """分析单个场景的图像质量"""
    result = {
        "scene_id": scene.get("id"),
        "description": scene.get("description", ""),
        "image_path": str(image_path),
        "has_character": "韩立" in scene.get("description", "") or "Han Li" in scene.get("prompt", ""),
        "is_front_view": False,
        "object_correct": True,
        "quality_score": 0,
        "issues": []
    }
    
    # 检查是否是正面视角
    prompt = scene.get("prompt", "").lower()
    description = scene.get("description", "").lower()
    
    front_keywords = ["front view", "facing camera", "正面", "面向镜头"]
    back_keywords = ["back", "背影", "from behind", "back view"]
    
    has_front = any(kw in prompt or kw in description for kw in front_keywords)
    has_back = any(kw in prompt or kw in description for kw in back_keywords)
    
    if result["has_character"]:
        result["is_front_view"] = has_front and not has_back
    
    # 检查物体识别是否正确
    scene_id = scene.get("id")
    if scene_id == 0:
        # 卷轴场景
        scroll_keywords = ["scroll", "卷轴"]
        weapon_keywords = ["weapon", "兵器", "sword", "刀"]
        has_scroll = any(kw in prompt or kw in description for kw in scroll_keywords)
        has_weapon = any(kw in prompt or kw in description for kw in weapon_keywords)
        result["object_correct"] = has_scroll and not has_weapon
        if not result["object_correct"]:
            result["issues"].append("卷轴识别错误（可能生成了兵器）")
    elif scene_id == 2:
        # 城市场景
        city_keywords = ["city", "城市", "silhouette"]
        people_keywords = ["people", "人物", "character", "person"]
        has_city = any(kw in prompt or kw in description for kw in city_keywords)
        has_people = any(kw in prompt or kw in description for kw in people_keywords)
        result["object_correct"] = has_city and not has_people
        if not result["object_correct"]:
            result["issues"].append("城市识别错误（可能生成了人物）")
    
    # 计算质量分数
    quality_score = 100
    if result["has_character"] and not result["is_front_view"]:
        quality_score -= 30
        result["issues"].append("人物不是正面视角")
    if not result["object_correct"]:
        quality_score -= 50
    if scene_id == 1 and not result["is_front_view"]:
        quality_score -= 20
        result["issues"].append("中景场景人物不是正面")
    if scene_id == 3 and not result["is_front_view"]:
        quality_score -= 30
        result["issues"].append("近景场景人物不是正面")
    
    result["quality_score"] = max(0, quality_score)
    
    return result


def generate_quality_report(analysis_results: List[Dict], output_dir: Path, mapping: Dict[str, int], scenes: List[Dict]):
    """生成质量评估报告"""
    report_path = output_dir / "quality_report_corrected.md"
    
    # 统计信息
    total_scenes = len(analysis_results)
    character_scenes = [r for r in analysis_results if r["has_character"]]
    front_view_count = sum(1 for r in character_scenes if r["is_front_view"])
    object_correct_count = sum(1 for r in analysis_results if r["object_correct"])
    avg_quality = sum(r["quality_score"] for r in analysis_results) / total_scenes if total_scenes > 0 else 0
    
    # 生成报告
    report = f"""# 测试场景质量评估报告（修正版）

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## ⚠️ 重要说明：场景和图像对应关系

由于图像生成时场景顺序可能出现错乱，已使用手动映射关系：

"""
    for img_name, scene_id in sorted(mapping.items()):
        if scene_id < len(scenes):
            scene = scenes[scene_id]
            report += f"- **{img_name}** -> 场景ID {scene_id}: {scene.get('description', '')[:50]}...\n"
    
    report += f"""

## 📊 总体评估

- **总场景数**: {total_scenes}
- **人物场景数**: {len(character_scenes)}
- **正面视角率**: {front_view_count}/{len(character_scenes)} ({front_view_count/len(character_scenes)*100:.1f}%)" if len(character_scenes) > 0 else "N/A (无人物场景)"
- **物体识别正确率**: {object_correct_count}/{total_scenes} ({object_correct_count/total_scenes*100:.1f}%)
- **平均质量分数**: {avg_quality:.1f}/100

## 🎯 质量等级

"""
    
    if avg_quality >= 80:
        report += "✅ **优秀** (80-100分): 可以用于MVP演示，质量达到可用标准\n"
    elif avg_quality >= 70:
        report += "⚠️ **可用** (70-79分): 基本达到可用标准，需要继续优化\n"
    elif avg_quality >= 60:
        report += "⚠️ **需改进** (60-69分): 需要修复关键问题后再考虑MVP\n"
    else:
        report += "❌ **不可用** (<60分): 必须先优化效果，再考虑MVP\n"
    
    report += f"""

## 📋 详细场景分析

"""
    
    for result in sorted(analysis_results, key=lambda x: x["scene_id"]):
        scene_id = result["scene_id"]
        quality_score = result["quality_score"]
        status_icon = "✅" if quality_score >= 80 else "⚠️" if quality_score >= 70 else "❌"
        
        report += f"""### 场景 {scene_id}: {status_icon} {quality_score}分

- **图像文件**: {result.get('image_file', '未知')}
- **描述**: {result["description"]}
- **图像路径**: `{result["image_path"]}`
- **是否有人物**: {"是" if result["has_character"] else "否"}
- **是否正面视角**: {"是" if result["is_front_view"] else "否"}
- **物体识别正确**: {"是" if result["object_correct"] else "否"}
- **质量分数**: {quality_score}/100

"""
        
        if result["issues"]:
            report += "**问题**:\n"
            for issue in result["issues"]:
                report += f"- ⚠️ {issue}\n"
        
        report += "\n"
    
    # 写入报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 质量评估报告已生成: {report_path}")
    
    return report_path


def main():
    project_root = Path(__file__).parent
    test_json_path = project_root / "renjie" / "episode" / "test_scenes.json"
    output_dir = project_root / "outputs" / "test_scenes_quality"
    images_dir = output_dir / "images"
    mapping_file = output_dir / "scene_image_mapping.json"
    
    print("=" * 80)
    print("🔍 使用映射关系重新分析测试场景图像质量")
    print("=" * 80)
    print()
    
    # 加载场景
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    scenes = test_data.get("scenes", [])
    
    # 加载映射关系
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        mapping = mapping_data.get("suggested_mapping", mapping_data.get("mapping", {}))
    else:
        print(f"⚠️  警告: 映射文件不存在，使用默认映射")
        mapping = {}
    
    print(f"📋 映射关系: {len(mapping)} 个")
    for img_name, scene_id in sorted(mapping.items()):
        if scene_id < len(scenes):
            scene = scenes[scene_id]
            print(f"   {img_name} -> 场景ID {scene_id}: {scene.get('description', '')[:40]}...")
    print()
    
    # 分析质量
    analysis_results = []
    image_files = sorted(images_dir.glob("scene_*.png"))
    
    for img_file in image_files:
        scene_id = mapping.get(img_file.name, -1)
        if scene_id < 0:
            file_num = int(img_file.stem.split('_')[1])
            scene_id = file_num - 1
        
        if scene_id < 0 or scene_id >= len(scenes):
            continue
        
        scene = scenes[scene_id]
        result = analyze_image_quality(scene, img_file)
        result["image_file"] = img_file.name
        analysis_results.append(result)
    
    # 生成报告
    report_path = generate_quality_report(analysis_results, output_dir, mapping, scenes)
    
    # 统计
    avg_quality = sum(r["quality_score"] for r in analysis_results) / len(analysis_results) if analysis_results else 0
    character_scenes = [r for r in analysis_results if r["has_character"]]
    front_view_count = sum(1 for r in character_scenes if r["is_front_view"])
    
    print(f"\n📊 总体评估:")
    print(f"   平均质量分数: {avg_quality:.1f}/100")
    if len(character_scenes) > 0:
        print(f"   正面视角率: {front_view_count}/{len(character_scenes)} ({front_view_count/len(character_scenes)*100:.1f}%)")
    
    if avg_quality >= 70:
        print(f"\n✅ 质量达到可用标准，可以继续开发MVP")
    else:
        print(f"\n⚠️  质量未达可用标准，建议先优化效果")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

