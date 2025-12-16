#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 motion 字符串是否能被正确解析
"""

import json
import re

def analyze_motion_string(motion_str, camera_str=""):
    """分析 motion 字符串是否能被解析出运动类型"""
    if not motion_str or not isinstance(motion_str, str):
        return {"parseable": False, "reason": "空值或非字符串"}
    
    motion_lower = motion_str.lower()
    camera_lower = camera_str.lower() if camera_str else ""
    
    # 检查缩放运动关键词
    zoom_keywords = ["zoom", "dolly", "push", "pull", "推", "拉", "推进", "拉远"]
    has_zoom = any(kw in motion_lower for kw in zoom_keywords)
    
    # 检查平移运动关键词
    pan_keywords = ["pan", "move", "平移", "横移", "移动"]
    has_pan = any(kw in motion_lower for kw in pan_keywords) or any(kw in camera_lower for kw in pan_keywords)
    
    # 检查方向关键词
    direction_keywords = ["left", "right", "up", "down", "forward", "backward", 
                         "向左", "向右", "向上", "向下", "前", "后"]
    has_direction = any(kw in motion_lower for kw in direction_keywords)
    
    result = {
        "parseable": False,
        "detected_motion": None,
        "detected_direction": None,
        "reason": "",
        "suggestions": []
    }
    
    if has_zoom:
        result["parseable"] = True
        if "in" in motion_lower or "forward" in motion_lower or "close" in motion_lower or "推" in motion_str:
            result["detected_motion"] = "zoom_in"
        elif "out" in motion_lower or "backward" in motion_lower or "pull" in motion_lower or "拉" in motion_str:
            result["detected_motion"] = "zoom_out"
        else:
            result["detected_motion"] = "zoom_in"  # 默认
        result["reason"] = "检测到缩放运动关键词"
    elif has_pan:
        result["parseable"] = True
        result["detected_motion"] = "pan"
        if has_direction:
            if "left" in motion_lower or "向左" in motion_str:
                result["detected_direction"] = "right"
            elif "right" in motion_lower or "向右" in motion_str:
                result["detected_direction"] = "left"
            elif "up" in motion_lower or "向上" in motion_str:
                result["detected_direction"] = "down"
            elif "down" in motion_lower or "向下" in motion_str:
                result["detected_direction"] = "up"
            else:
                result["detected_direction"] = "right"  # 默认
        else:
            result["detected_direction"] = "right"  # 默认
        result["reason"] = "检测到平移运动关键词"
    else:
        # 没有明确的运动关键词，会使用默认运镜
        result["parseable"] = True  # 代码会使用默认运镜
        result["detected_motion"] = "default"
        result["reason"] = "未检测到明确运动关键词，将使用默认运镜（根据场景类型智能选择）"
        
        # 根据场景类型给出建议
        if "close" in motion_lower or "close-up" in motion_lower or "closeup" in motion_lower or "特写" in motion_str:
            result["suggestions"].append("特写场景建议使用轻微拉远: {\"type\": \"pull_out\", \"speed\": \"slow\"}")
        elif "wide" in motion_lower or "panoramic" in motion_lower or "远景" in motion_str:
            result["suggestions"].append("远景场景建议使用平移: {\"type\": \"pan\", \"direction\": \"left_to_right\", \"speed\": \"slow\"}")
        elif "fixed" in motion_lower or "static" in motion_lower or "静止" in motion_str:
            result["suggestions"].append("静态场景可以使用: {\"type\": \"static\"}")
        else:
            result["suggestions"].append("建议添加明确的运动类型，如: {\"type\": \"pan\", \"direction\": \"left_to_right\", \"speed\": \"slow\"}")
    
    return result

def analyze_json_file(json_path):
    """分析JSON文件中的所有motion配置"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    results = []
    
    for scene in scenes:
        scene_id = scene.get("id", "unknown")
        camera = scene.get("camera", "")
        visual = scene.get("visual", {})
        motion_value = visual.get("motion", "") if isinstance(visual, dict) else ""
        
        if isinstance(motion_value, dict):
            # 对象格式
            result = {
                "scene_id": scene_id,
                "motion_value": motion_value,
                "format": "object",
                "parseable": True,
                "reason": "对象格式（符合规范）"
            }
        elif isinstance(motion_value, str):
            # 字符串格式
            analysis = analyze_motion_string(motion_value, camera)
            result = {
                "scene_id": scene_id,
                "motion_value": motion_value,
                "camera": camera,
                "format": "string",
                **analysis
            }
        else:
            result = {
                "scene_id": scene_id,
                "motion_value": motion_value,
                "format": "missing",
                "parseable": True,
                "reason": "缺失motion字段，将使用默认运镜"
            }
        
        results.append(result)
    
    return results

if __name__ == "__main__":
    import sys
    
    json_path = "lingjie/scenes/2.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    results = analyze_json_file(json_path)
    
    print("=" * 80)
    print(f"Motion 解析分析: {json_path}")
    print("=" * 80)
    
    parseable_count = sum(1 for r in results if r.get("parseable", False))
    unparseable_count = len(results) - parseable_count
    
    print(f"\n总场景数: {len(results)}")
    print(f"可解析: {parseable_count}")
    print(f"不可解析: {unparseable_count}")
    
    print("\n详细分析:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n场景 {i} (id={result['scene_id']}):")
        print(f"  motion: {result.get('motion_value', 'N/A')}")
        print(f"  camera: {result.get('camera', 'N/A')}")
        print(f"  格式: {result.get('format', 'unknown')}")
        print(f"  可解析: {result.get('parseable', False)}")
        print(f"  原因: {result.get('reason', 'N/A')}")
        
        if result.get("detected_motion"):
            print(f"  检测到的运动: {result['detected_motion']}")
        if result.get("detected_direction"):
            print(f"  检测到的方向: {result['detected_direction']}")
        if result.get("suggestions"):
            print(f"  建议:")
            for suggestion in result["suggestions"]:
                print(f"    - {suggestion}")

