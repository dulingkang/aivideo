#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查JSON文件中的镜头运镜信息
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

def check_scene_camera_motion(scene: Dict) -> Dict:
    """检查单个场景的运镜信息"""
    result = {
        "has_camera": False,
        "has_motion": False,
        "camera": "",
        "motion": "",
        "status": "missing"
    }
    
    # 检查 camera 字段
    camera = scene.get("camera", "")
    if camera:
        result["has_camera"] = True
        result["camera"] = camera
    
    # 检查 visual.motion 字段
    visual = scene.get("visual", {})
    if isinstance(visual, dict):
        motion = visual.get("motion", "")
        if motion:
            result["has_motion"] = True
            result["motion"] = motion
    
    # 判断状态
    if result["has_camera"] or result["has_motion"]:
        result["status"] = "ok"
    else:
        result["status"] = "missing"
    
    return result

def check_json_file(json_path: str) -> Dict:
    """检查单个JSON文件的运镜信息"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    results = []
    
    for i, scene in enumerate(scenes):
        scene_id = scene.get("id") or scene.get("scene_number") or (i + 1)
        check_result = check_scene_camera_motion(scene)
        check_result["scene_id"] = scene_id
        check_result["scene_number"] = i + 1
        results.append(check_result)
    
    return {
        "file": os.path.basename(json_path),
        "total_scenes": len(scenes),
        "scenes": results
    }

def check_directory(directory: str) -> List[Dict]:
    """检查目录中所有JSON文件的运镜信息"""
    dir_path = Path(directory)
    json_files = sorted(dir_path.glob("*.json"))
    
    all_results = []
    for json_file in json_files:
        try:
            result = check_json_file(str(json_file))
            all_results.append(result)
        except Exception as e:
            print(f"✗ 检查 {json_file.name} 失败: {e}")
    
    return all_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="检查JSON文件中的镜头运镜信息")
    parser.add_argument("--dir", "-d", default="lingjie/scenes", help="JSON文件目录")
    parser.add_argument("--file", "-f", help="单个JSON文件路径")
    parser.add_argument("--summary", "-s", action="store_true", help="只显示汇总信息")
    
    args = parser.parse_args()
    
    if args.file:
        # 检查单个文件
        result = check_json_file(args.file)
        print(f"\n文件: {result['file']}")
        print(f"总场景数: {result['total_scenes']}")
        print("\n场景运镜信息:")
        print("-" * 80)
        for scene_info in result["scenes"]:
            status_icon = "✓" if scene_info["status"] == "ok" else "✗"
            print(f"{status_icon} 场景 {scene_info['scene_number']} (id={scene_info['scene_id']}):")
            if scene_info["has_camera"]:
                print(f"    camera: {scene_info['camera']}")
            if scene_info["has_motion"]:
                print(f"    motion: {scene_info['motion']}")
            if scene_info["status"] == "missing":
                print(f"    ⚠ 缺少运镜信息（将使用默认运镜）")
    else:
        # 检查目录
        results = check_directory(args.dir)
        
        if args.summary:
            # 只显示汇总
            total_scenes = 0
            scenes_with_motion = 0
            scenes_without_motion = 0
            
            for result in results:
                total_scenes += result["total_scenes"]
                for scene_info in result["scenes"]:
                    if scene_info["status"] == "ok":
                        scenes_with_motion += 1
                    else:
                        scenes_without_motion += 1
            
            print(f"\n汇总信息:")
            print(f"  总文件数: {len(results)}")
            print(f"  总场景数: {total_scenes}")
            print(f"  有运镜的场景: {scenes_with_motion} ({scenes_with_motion/total_scenes*100:.1f}%)")
            print(f"  无运镜的场景: {scenes_without_motion} ({scenes_without_motion/total_scenes*100:.1f}%)")
        else:
            # 显示详细信息
            print("=" * 80)
            print("检查镜头运镜信息")
            print("=" * 80)
            
            total_scenes = 0
            scenes_with_motion = 0
            scenes_without_motion = 0
            
            for result in results:
                print(f"\n文件: {result['file']}")
                print(f"总场景数: {result['total_scenes']}")
                
                for scene_info in result["scenes"]:
                    total_scenes += 1
                    status_icon = "✓" if scene_info["status"] == "ok" else "✗"
                    scene_label = f"场景 {scene_info['scene_number']}"
                    if scene_info['scene_id'] not in [None, scene_info['scene_number']]:
                        scene_label += f" (id={scene_info['scene_id']})"
                    
                    if scene_info["status"] == "ok":
                        scenes_with_motion += 1
                        print(f"  {status_icon} {scene_label}:")
                        if scene_info["has_camera"]:
                            print(f"      camera: {scene_info['camera']}")
                        if scene_info["has_motion"]:
                            print(f"      motion: {scene_info['motion']}")
                    else:
                        scenes_without_motion += 1
                        print(f"  {status_icon} {scene_label}: ⚠ 缺少运镜信息（将使用默认运镜）")
            
            print("\n" + "=" * 80)
            print("汇总:")
            print(f"  总场景数: {total_scenes}")
            print(f"  有运镜的场景: {scenes_with_motion} ({scenes_with_motion/total_scenes*100:.1f}%)")
            print(f"  无运镜的场景: {scenes_without_motion} ({scenes_without_motion/total_scenes*100:.1f}%)")
            print("=" * 80)

if __name__ == "__main__":
    main()

