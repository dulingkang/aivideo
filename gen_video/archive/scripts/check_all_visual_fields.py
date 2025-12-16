#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面检查所有场景的 visual 字段，确保字段职责清晰且无错误
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

def check_visual_field(scene: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    检查单个场景的 visual 字段
    
    Returns:
        List of (issue_type, message) tuples
    """
    issues = []
    scene_id = scene.get("id", "未知")
    description = scene.get("description", "")
    visual = scene.get("visual", {}) or {}
    
    if not isinstance(visual, dict):
        return [("error", f"场景 {scene_id}: visual 不是字典类型")]
    
    composition = visual.get("composition", "").strip()
    environment = visual.get("environment", "").strip()
    character_pose = visual.get("character_pose", "").strip()
    fx = visual.get("fx", "").strip()
    
    # 1. 检查字段是否相同
    if composition and environment and composition == environment:
        issues.append(("duplicate", f"场景 {scene_id}: composition == environment"))
    if composition and fx and composition == fx:
        issues.append(("duplicate", f"场景 {scene_id}: composition == fx"))
    if environment and fx and environment == fx:
        issues.append(("duplicate", f"场景 {scene_id}: environment == fx"))
    if character_pose and composition and character_pose == composition:
        issues.append(("duplicate", f"场景 {scene_id}: character_pose == composition"))
    if character_pose and environment and character_pose == environment:
        issues.append(("duplicate", f"场景 {scene_id}: character_pose == environment"))
    if character_pose and fx and character_pose == fx:
        issues.append(("duplicate", f"场景 {scene_id}: character_pose == fx"))
    
    # 2. 检查 composition 是否包含环境信息（应该包含，这是正常的）
    # 但 composition 应该包含主体+背景关系，不应该只是环境
    
    # 3. 检查 environment 是否包含角色动作
    if environment:
        role_keywords = ["韩立", "躺在", "躺", "一动不动", "感受", "回忆", "脸色", "难看",
                         "睁大", "双目", "注视", "偏动", "头颅", "胸膛", "一鼓", "一瘪",
                         "神色", "一变", "听到", "看清", "看到", "观察", "凝视", "使用",
                         "元婴", "虚化", "精元", "灌注", "身体", "费劲", "喷出", "激射"]
        for kw in role_keywords:
            if kw in environment:
                issues.append(("logic", f"场景 {scene_id}: environment 包含角色动作 '{kw}'"))
                break
    
    # 4. 检查 character_pose 是否包含环境信息
    if character_pose:
        env_keywords = ["沙地", "沙砾", "青灰色", "天空", "太阳", "月亮", "地面", "高空",
                        "远处", "附近", "空中", "上空", "怪鸟", "小鸟", "骑士", "怪兽"]
        for kw in env_keywords:
            if kw in character_pose:
                issues.append(("logic", f"场景 {scene_id}: character_pose 包含环境信息 '{kw}'"))
                break
        
        # 检查是否包含物体描述
        object_keywords = ["沙砾", "怪鸟", "小鸟", "骑士", "怪兽", "青芒", "强风"]
        for kw in object_keywords:
            if kw in character_pose:
                issues.append(("logic", f"场景 {scene_id}: character_pose 包含物体描述 '{kw}'"))
                break
    
    # 5. 检查 fx 是否包含环境信息
    if fx:
        env_keywords = ["沙地", "沙砾", "青灰色", "天空", "太阳", "月亮", "地面", "高空",
                        "远处", "附近", "空中", "上空"]
        for kw in env_keywords:
            if kw in fx and "震动" not in fx and "声音" not in fx:
                # 震动和声音是特效，可以包含环境词
                issues.append(("logic", f"场景 {scene_id}: fx 包含环境信息 '{kw}'"))
                break
    
    # 6. 检查 composition 是否过于简单（应该包含主体+背景）
    if composition:
        # composition 应该包含主体和背景关系
        has_role = any(kw in composition for kw in ["韩立", "怪鸟", "小鸟", "骑士"])
        has_env = any(kw in composition for kw in ["沙地", "天空", "地面", "高空", "远处"])
        
        if not has_role and not has_env:
            issues.append(("warning", f"场景 {scene_id}: composition 可能过于简单"))
    
    # 7. 检查无角色场景的 character_pose
    if not description or "韩立" not in description:
        if character_pose:
            # 无角色场景不应该有 character_pose
            issues.append(("warning", f"场景 {scene_id}: 无角色场景但 character_pose 有内容"))
    
    return issues

def main():
    json_path = Path("lingjie/scenes/1.json")
    
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    
    all_issues = {
        "duplicate": [],
        "logic": [],
        "warning": [],
        "error": []
    }
    
    for scene in scenes:
        issues = check_visual_field(scene)
        for issue_type, message in issues:
            all_issues[issue_type].append(message)
    
    print("=" * 60)
    print("Visual 字段全面检查结果")
    print("=" * 60)
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    
    if total_issues == 0:
        print("\n✓ 所有场景的 visual 字段检查通过！")
        return
    
    # 显示重复问题
    if all_issues["duplicate"]:
        print("\n【重复问题】")
        for issue in all_issues["duplicate"]:
            print(f"  ⚠ {issue}")
    
    # 显示逻辑问题
    if all_issues["logic"]:
        print("\n【逻辑问题】")
        for issue in all_issues["logic"]:
            print(f"  ⚠ {issue}")
    
    # 显示警告
    if all_issues["warning"]:
        print("\n【警告】")
        for issue in all_issues["warning"]:
            print(f"  ℹ {issue}")
    
    # 显示错误
    if all_issues["error"]:
        print("\n【错误】")
        for issue in all_issues["error"]:
            print(f"  ❌ {issue}")
    
    print(f"\n总计发现 {total_issues} 个问题")

if __name__ == "__main__":
    main()

