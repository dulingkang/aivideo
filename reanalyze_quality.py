#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新分析已生成的测试场景图像质量
不重新生成图像，只重新分析并生成报告
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 导入分析函数
sys.path.insert(0, str(Path(__file__).parent))
from test_scenes_quality import analyze_image_quality, generate_quality_report, load_test_scenes


def main():
    # 配置路径
    project_root = Path(__file__).parent
    test_json_path = project_root / "renjie" / "episode" / "test_scenes.json"
    output_dir = project_root / "outputs" / "test_scenes_quality"
    images_dir = output_dir / "images"
    
    print("=" * 80)
    print("🔍 重新分析测试场景图像质量")
    print("=" * 80)
    print(f"📄 测试场景JSON: {test_json_path}")
    print(f"🖼️  图像目录: {images_dir}")
    print(f"📁 输出目录: {output_dir}")
    print()
    
    # 检查文件是否存在
    if not test_json_path.exists():
        print(f"❌ 错误: 测试场景JSON文件不存在: {test_json_path}")
        return 1
    
    if not images_dir.exists():
        print(f"❌ 错误: 图像目录不存在: {images_dir}")
        return 1
    
    # 加载测试场景
    print("📖 加载测试场景...")
    test_data = load_test_scenes(test_json_path)
    scenes = test_data.get("scenes", [])
    print(f"✅ 加载了 {len(scenes)} 个测试场景\n")
    
    # 列出所有图像文件
    image_files = sorted(images_dir.glob("scene_*.png"))
    print(f"🖼️  找到 {len(image_files)} 个图像文件:")
    for img_file in image_files:
        print(f"   - {img_file.name}")
    print()
    
    # 分析质量
    print("=" * 80)
    print("📊 分析图像质量...")
    print("=" * 80)
    
    analysis_results = []
    for idx, scene in enumerate(scenes):
        scene_id = scene.get("id", idx)
        # 文件名编号 = 数组索引 + 1
        file_num = idx + 1
        image_path = images_dir / f"scene_{file_num:03d}.png"
        
        if not image_path.exists():
            print(f"⚠️  警告: 场景 {scene_id} 的图像文件不存在: {image_path.name}")
            # 尝试按顺序查找
            if idx < len(image_files):
                image_path = image_files[idx]
                print(f"   → 使用: {image_path.name}")
        
        result = analyze_image_quality(scene, image_path)
        analysis_results.append(result)
        
        status_icon = "✅" if result["quality_score"] >= 70 else "⚠️" if result["quality_score"] >= 60 else "❌"
        print(f"{status_icon} 场景 {scene_id}: {result['quality_score']}分 - {result['description'][:40]}...")
        print(f"   图像: {image_path.name}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"   ⚠️  {issue}")
    
    # 生成报告
    print("\n" + "=" * 80)
    print("📝 生成质量评估报告...")
    print("=" * 80)
    
    report_path = generate_quality_report(analysis_results, output_dir)
    
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
    print(f"   正面视角率: {front_view_count}/{len(character_scenes)} ({front_view_count/len(character_scenes)*100:.1f}%)" if len(character_scenes) > 0 else "   N/A (无人物场景)")
    print(f"   物体识别正确率: {object_correct_count}/{total_scenes} ({object_correct_count/total_scenes*100:.1f}%)")
    print()
    
    if avg_quality >= 70:
        print(f"✅ 质量达到可用标准，可以继续开发MVP")
    else:
        print(f"⚠️  质量未达可用标准，建议先优化效果")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

