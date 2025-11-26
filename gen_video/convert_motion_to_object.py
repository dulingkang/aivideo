#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JSON文件中的motion字符串格式转换为对象格式
"""

import json
import sys
from pathlib import Path

def convert_motion_string_to_object(motion_str, camera_str=""):
    """将字符串格式的motion转换为对象格式"""
    if not motion_str or not isinstance(motion_str, str):
        return None
    
    motion_lower = motion_str.lower()
    camera_lower = camera_str.lower() if camera_str else ""
    
    # 检查是否已经是静态场景
    if "fixed" in motion_lower or "static" in motion_lower or "fade out" in motion_lower:
        return {"type": "static"}
    
    # 检查缩放运动
    if "zoom" in motion_lower or "dolly" in motion_lower or "push" in motion_lower:
        if "in" in motion_lower or "forward" in motion_lower or "close" in motion_lower:
            return {"type": "push_in", "speed": "slow"}
        elif "out" in motion_lower or "backward" in motion_lower or "pull" in motion_lower:
            return {"type": "pull_out", "speed": "slow"}
        else:
            return {"type": "push_in", "speed": "slow"}
    
    if "pull" in motion_lower:
        return {"type": "pull_out", "speed": "slow"}
    
    # 检查平移运动
    if "pan" in motion_lower or "pan" in camera_lower or "move" in motion_lower or "approach" in motion_lower:
        direction = "left_to_right"  # 默认
        if "left" in motion_lower or "left_to_right" in motion_lower:
            direction = "left_to_right"
        elif "right" in motion_lower or "right_to_left" in motion_lower:
            direction = "right_to_left"
        elif "up" in motion_lower or "upward" in motion_lower:
            direction = "up"
        elif "down" in motion_lower or "downward" in motion_lower:
            direction = "down"
        
        speed = "slow"
        if "slow" in motion_lower:
            speed = "slow"
        elif "fast" in motion_lower:
            speed = "fast"
        else:
            speed = "medium"
        
        return {"type": "pan", "direction": direction, "speed": speed}
    
    # 根据camera类型智能推断
    if "close" in camera_lower or "close-up" in camera_lower or "closeup" in camera_lower or "close" in motion_lower:
        # 特写场景：轻微拉远
        return {"type": "pull_out", "speed": "slow"}
    elif "panoramic" in camera_lower or "panoramic" in motion_lower or "wide" in camera_lower:
        # 远景场景：平移
        return {"type": "pan", "direction": "left_to_right", "speed": "slow"}
    elif "medium" in camera_lower or "medium" in motion_lower:
        # 中景场景：轻微平移或静态
        # 根据场景内容判断，对话场景用静态，动作场景用平移
        if "dialogue" in motion_lower or "negotiate" in motion_lower or "confirm" in motion_lower:
            return {"type": "static"}
        else:
            return {"type": "pan", "direction": "left_to_right", "speed": "slow"}
    elif "macro" in camera_lower or "macro" in motion_lower:
        # 微距：静态或轻微平移
        return {"type": "pan", "direction": "left_to_right", "speed": "slow"}
    else:
        # 默认：静态
        return {"type": "static"}

def convert_json_file(json_path, backup=True):
    """转换JSON文件中的motion格式"""
    json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"错误: 文件不存在: {json_path}")
        return False
    
    # 读取JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 备份
    if backup:
        backup_path = json_path.with_suffix('.json.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ 已创建备份: {backup_path}")
    
    # 转换motion
    scenes = data.get("scenes", [])
    converted_count = 0
    
    for scene in scenes:
        visual = scene.get("visual", {})
        if not isinstance(visual, dict):
            continue
        
        motion_value = visual.get("motion", "")
        camera = scene.get("camera", "")
        
        # 如果已经是对象格式，跳过
        if isinstance(motion_value, dict):
            continue
        
        # 如果是字符串格式，转换
        if isinstance(motion_value, str) and motion_value:
            new_motion = convert_motion_string_to_object(motion_value, camera)
            if new_motion:
                visual["motion"] = new_motion
                converted_count += 1
                print(f"  场景 {scene.get('id', 'unknown')}: '{motion_value}' -> {new_motion}")
    
    # 保存
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 转换完成: {converted_count} 个场景的motion已转换为对象格式")
    print(f"✓ 已保存到: {json_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_motion_to_object.py <json_file> [--no-backup]")
        print("示例: python convert_motion_to_object.py lingjie/scenes/2.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    backup = "--no-backup" not in sys.argv
    
    print("=" * 80)
    print(f"转换 motion 格式: {json_file}")
    print("=" * 80)
    print()
    
    success = convert_json_file(json_file, backup=backup)
    
    if success:
        print("\n" + "=" * 80)
        print("转换完成！")
        print("=" * 80)
    else:
        print("\n转换失败！")
        sys.exit(1)

