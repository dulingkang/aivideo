#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证JSON文件中的运镜配置是否符合规范
根据 doc.md 中的规范检查 camera 和 visual.motion 字段
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 根据 doc.md 定义的规范值
VALID_MOTION_TYPES = ["static", "pan", "tilt", "push_in", "pull_out", "orbit", "shake", "follow"]
VALID_MOTION_DIRECTIONS = ["left_to_right", "right_to_left", "up", "down", "forward", "backward", "around"]
VALID_MOTION_SPEEDS = ["slow", "medium", "fast"]

# 代码中期望的 camera 值（从代码中提取）
EXPECTED_CAMERA_TYPES = [
    "wide_shot_low_angle", "medium_shot_front", "medium_low_angle",
    "close_up_hand", "medium_shot_side", "wide_shot_front",
    "low_angle_close", "wide_shot_dynamic", "high_angle_top",
    "close_rune_detail", "wide_shot_overhead", "dynamic_follow",
    "wide_slow_pan", "close_up_side", "over_shoulder_dark_depth",
    "tracking_shot_rear", "fast_push_in", "low_angle_dynamic",
    "front_long_shot", "close_up_shadow", "wide_shot_pull_in",
    "slow_close_up", "wide_pan", "macro_shot",
    "dynamic_shot", "medium_rotating", "wide_pull_back",
    "close_up_forehead", "long_shot_dark", "slow_push_in",
    "orbiting_shot", "wide_pull_up"
]

def check_motion_format(motion_value) -> Dict:
    """检查 motion 字段格式"""
    result = {
        "valid": False,
        "format": "unknown",
        "issues": []
    }
    
    if motion_value is None or motion_value == "":
        result["format"] = "missing"
        result["issues"].append("缺少 motion 字段")
        return result
    
    # 检查是否为字符串格式（旧格式）
    if isinstance(motion_value, str):
        result["format"] = "string"
        result["issues"].append("motion 是字符串格式，建议使用结构化对象格式")
        # 检查字符串中是否包含关键词
        motion_lower = motion_value.lower()
        has_pan = "pan" in motion_lower or "move" in motion_lower or "平移" in motion_value
        has_zoom = "zoom" in motion_lower or "push" in motion_lower or "pull" in motion_lower or "推" in motion_value or "拉" in motion_value
        if has_pan or has_zoom:
            result["valid"] = True  # 虽然格式不对，但内容可能有用
        else:
            result["issues"].append("字符串格式的 motion 可能无法被正确解析")
    
    # 检查是否为对象格式（新格式，符合doc.md规范）
    elif isinstance(motion_value, dict):
        result["format"] = "object"
        motion_type = motion_value.get("type", "")
        motion_direction = motion_value.get("direction", "")
        motion_speed = motion_value.get("speed", "")
        
        if not motion_type:
            result["issues"].append("缺少 motion.type 字段")
        elif motion_type not in VALID_MOTION_TYPES:
            result["issues"].append(f"motion.type 值 '{motion_type}' 不在有效值列表中: {VALID_MOTION_TYPES}")
        else:
            result["valid"] = True
        
        if motion_direction and motion_direction not in VALID_MOTION_DIRECTIONS:
            result["issues"].append(f"motion.direction 值 '{motion_direction}' 不在有效值列表中: {VALID_MOTION_DIRECTIONS}")
        
        if motion_speed and motion_speed not in VALID_MOTION_SPEEDS:
            result["issues"].append(f"motion.speed 值 '{motion_speed}' 不在有效值列表中: {VALID_MOTION_SPEEDS}")
    
    else:
        result["issues"].append(f"motion 字段类型错误: {type(motion_value)}")
    
    return result

def check_camera_format(camera_value) -> Dict:
    """检查 camera 字段格式"""
    result = {
        "valid": False,
        "format": "unknown",
        "issues": []
    }
    
    if camera_value is None or camera_value == "":
        result["format"] = "missing"
        result["issues"].append("缺少 camera 字段")
        return result
    
    if not isinstance(camera_value, str):
        result["issues"].append(f"camera 字段类型错误: {type(camera_value)}")
        return result
    
    result["format"] = "string"
    camera_lower = camera_value.lower()
    
    # 检查是否为代码期望的格式
    if camera_value in EXPECTED_CAMERA_TYPES:
        result["valid"] = True
    else:
        # 检查是否包含关键词（可能可以解析）
        has_wide = "wide" in camera_lower or "远景" in camera_value
        has_close = "close" in camera_lower or "特写" in camera_value or "closeup" in camera_lower
        has_medium = "medium" in camera_lower or "中景" in camera_value
        has_overhead = "overhead" in camera_lower or "俯视" in camera_value
        has_sky = "sky" in camera_lower or "天空" in camera_value
        
        if has_wide or has_close or has_medium or has_overhead or has_sky:
            result["valid"] = True  # 虽然格式不完全匹配，但可能可以解析
            result["issues"].append(f"camera 值 '{camera_value}' 不在标准列表中，但可能可以解析")
        else:
            result["issues"].append(f"camera 值 '{camera_value}' 可能无法被正确解析")
    
    return result

def validate_scene(scene: Dict, scene_index: int) -> Dict:
    """验证单个场景的运镜配置"""
    scene_id = scene.get("id") or scene.get("scene_number") or scene_index
    
    result = {
        "scene_id": scene_id,
        "scene_index": scene_index,
        "camera": check_camera_format(scene.get("camera")),
        "motion": None,
        "has_issues": False,
        "issues": []
    }
    
    # 检查 visual.motion
    visual = scene.get("visual", {})
    if isinstance(visual, dict):
        motion_value = visual.get("motion")
        result["motion"] = check_motion_format(motion_value)
    else:
        result["motion"] = check_motion_format(None)
    
    # 汇总问题
    if result["camera"]["issues"]:
        result["has_issues"] = True
        result["issues"].extend([f"camera: {issue}" for issue in result["camera"]["issues"]])
    
    if result["motion"] and result["motion"]["issues"]:
        result["has_issues"] = True
        result["issues"].extend([f"motion: {issue}" for issue in result["motion"]["issues"]])
    
    return result

def validate_json_file(json_path: str) -> Dict:
    """验证单个JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    results = []
    
    for i, scene in enumerate(scenes):
        result = validate_scene(scene, i)
        results.append(result)
    
    return {
        "file": os.path.basename(json_path),
        "total_scenes": len(scenes),
        "scenes": results
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="验证JSON文件中的运镜配置")
    parser.add_argument("--dir", "-d", default="lingjie/scenes", help="JSON文件目录")
    parser.add_argument("--file", "-f", help="单个JSON文件路径")
    parser.add_argument("--fix", action="store_true", help="自动修复格式问题（将字符串格式转换为对象格式）")
    
    args = parser.parse_args()
    
    if args.file:
        result = validate_json_file(args.file)
        print(f"\n文件: {result['file']}")
        print(f"总场景数: {result['total_scenes']}")
        print("\n验证结果:")
        print("-" * 80)
        
        issues_count = 0
        for scene_result in result["scenes"]:
            if scene_result["has_issues"]:
                issues_count += 1
                print(f"\n场景 {scene_result['scene_index']+1} (id={scene_result['scene_id']}):")
                for issue in scene_result["issues"]:
                    print(f"  ⚠ {issue}")
                if scene_result["camera"]["format"] != "missing":
                    print(f"  camera: {scene_result['camera']['format']} - {scene_result['camera'].get('valid', False)}")
                if scene_result["motion"] and scene_result["motion"]["format"] != "missing":
                    print(f"  motion: {scene_result['motion']['format']} - {scene_result['motion'].get('valid', False)}")
        
        if issues_count == 0:
            print("\n✓ 所有场景的运镜配置都符合规范")
        else:
            print(f"\n⚠ 发现 {issues_count} 个场景有配置问题")
    else:
        # 检查目录
        dir_path = Path(args.dir)
        json_files = sorted(dir_path.glob("*.json"))
        
        all_results = []
        total_issues = 0
        
        for json_file in json_files:
            try:
                result = validate_json_file(str(json_file))
                all_results.append(result)
                
                # 统计问题
                for scene_result in result["scenes"]:
                    if scene_result["has_issues"]:
                        total_issues += 1
            except Exception as e:
                print(f"✗ 检查 {json_file.name} 失败: {e}")
        
        # 显示汇总
        print("=" * 80)
        print("运镜配置验证结果")
        print("=" * 80)
        
        total_scenes = sum(r["total_scenes"] for r in all_results)
        scenes_with_issues = total_issues
        
        print(f"\n总文件数: {len(all_results)}")
        print(f"总场景数: {total_scenes}")
        print(f"有问题的场景: {scenes_with_issues} ({scenes_with_issues/total_scenes*100:.1f}%)")
        print(f"无问题的场景: {total_scenes - scenes_with_issues} ({(total_scenes-scenes_with_issues)/total_scenes*100:.1f}%)")
        
        # 统计格式问题
        string_motion_count = 0
        object_motion_count = 0
        missing_motion_count = 0
        invalid_camera_count = 0
        
        for result in all_results:
            for scene_result in result["scenes"]:
                if scene_result["motion"]:
                    if scene_result["motion"]["format"] == "string":
                        string_motion_count += 1
                    elif scene_result["motion"]["format"] == "object":
                        object_motion_count += 1
                    elif scene_result["motion"]["format"] == "missing":
                        missing_motion_count += 1
                
                if not scene_result["camera"].get("valid", False):
                    invalid_camera_count += 1
        
        print(f"\n格式统计:")
        print(f"  motion 字符串格式: {string_motion_count} (建议转换为对象格式)")
        print(f"  motion 对象格式: {object_motion_count} (符合规范)")
        print(f"  motion 缺失: {missing_motion_count}")
        print(f"  camera 格式问题: {invalid_camera_count}")
        
        # 显示有问题的场景详情（前10个）
        print(f"\n问题场景示例（前10个）:")
        print("-" * 80)
        shown = 0
        for result in all_results:
            for scene_result in result["scenes"]:
                if scene_result["has_issues"] and shown < 10:
                    print(f"\n{result['file']} - 场景 {scene_result['scene_index']+1} (id={scene_result['scene_id']}):")
                    for issue in scene_result["issues"][:3]:  # 只显示前3个问题
                        print(f"  ⚠ {issue}")
                    shown += 1
                    if shown >= 10:
                        break
            if shown >= 10:
                break
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

