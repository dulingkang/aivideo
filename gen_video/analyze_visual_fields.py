#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 visual 字段的使用情况，提出合理的字段结构方案
"""

import json
from pathlib import Path
from typing import Dict, Any, List

def analyze_visual_fields(json_path: str):
    """分析 visual 字段的使用情况"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    
    stats = {
        "total_scenes": len(scenes),
        "empty_environment": 0,
        "empty_fx": 0,
        "empty_character_pose": 0,
        "has_environment": 0,
        "has_fx": 0,
        "has_character_pose": 0,
    }
    
    examples = {
        "empty_env_scenes": [],
        "empty_fx_scenes": [],
        "env_examples": [],
        "fx_examples": [],
    }
    
    for scene in scenes:
        scene_id = scene.get("id", "未知")
        visual = scene.get("visual", {}) or {}
        
        # 统计 environment
        env = visual.get("environment", "")
        if not env or env == "":
            stats["empty_environment"] += 1
            examples["empty_env_scenes"].append(scene_id)
        else:
            stats["has_environment"] += 1
            if len(examples["env_examples"]) < 3:
                examples["env_examples"].append({
                    "id": scene_id,
                    "env": env,
                    "description": scene.get("description", "")[:50]
                })
        
        # 统计 fx
        fx = visual.get("fx", "")
        if not fx or fx == "":
            stats["empty_fx"] += 1
            examples["empty_fx_scenes"].append(scene_id)
        else:
            stats["has_fx"] += 1
            if len(examples["fx_examples"]) < 3:
                examples["fx_examples"].append({
                    "id": scene_id,
                    "fx": fx,
                    "description": scene.get("description", "")[:50]
                })
        
        # 统计 character_pose
        pose = visual.get("character_pose", "")
        if not pose or pose == "":
            stats["empty_character_pose"] += 1
        else:
            stats["has_character_pose"] += 1
    
    return stats, examples

def main():
    json_path = "lingjie/scenes/1.json"
    stats, examples = analyze_visual_fields(json_path)
    
    print("=" * 60)
    print("Visual 字段使用情况分析")
    print("=" * 60)
    print(f"\n总场景数: {stats['total_scenes']}")
    print(f"\nEnvironment 字段:")
    print(f"  - 有内容: {stats['has_environment']} ({stats['has_environment']/stats['total_scenes']*100:.1f}%)")
    print(f"  - 为空: {stats['empty_environment']} ({stats['empty_environment']/stats['total_scenes']*100:.1f}%)")
    print(f"\nFX 字段:")
    print(f"  - 有内容: {stats['has_fx']} ({stats['has_fx']/stats['total_scenes']*100:.1f}%)")
    print(f"  - 为空: {stats['empty_fx']} ({stats['empty_fx']/stats['total_scenes']*100:.1f}%)")
    print(f"\nCharacter_pose 字段:")
    print(f"  - 有内容: {stats['has_character_pose']} ({stats['has_character_pose']/stats['total_scenes']*100:.1f}%)")
    print(f"  - 为空: {stats['empty_character_pose']} ({stats['empty_character_pose']/stats['total_scenes']*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("字段结构优化建议")
    print("=" * 60)
    
    print("\n【方案一：简化 visual 字段】")
    print("职责划分:")
    print("  - scene_profiles.yaml: 负责所有场景级别的环境背景")
    print("    * color_palette, terrain, sky, atmosphere, background_prompt")
    print("    * 这是场景模板，适用于整个场景类型")
    print("  - JSON visual.environment: 删除或仅用于特殊环境细节")
    print("    * 如果场景有特殊环境变化（与模板不同），才填写")
    print("    * 否则留空，由 YAML 模板统一管理")
    print("  - JSON visual.composition: 保留（构图描述）")
    print("  - JSON visual.character_pose: 保留（角色动作/姿势）")
    print("  - JSON visual.fx: 保留（特效，可为空）")
    print("  - JSON visual.motion: 保留（镜头运动）")
    
    print("\n【方案二：明确字段职责】")
    print("职责划分:")
    print("  - scene_profiles.yaml: 场景模板级别的环境（整体风格）")
    print("  - JSON visual.environment: 场景实例级别的环境细节")
    print("    * 只填写与模板不同的特殊环境细节")
    print("    * 如果与模板一致，留空（由 YAML 统一管理）")
    print("  - JSON visual.composition: 构图（主体+背景关系）")
    print("  - JSON visual.character_pose: 角色动作（不包含环境）")
    print("  - JSON visual.fx: 特效（可为空，只在有特效时填写）")
    print("  - JSON visual.motion: 镜头运动")
    
    print("\n【推荐方案】")
    print("采用方案一（简化 visual 字段）:")
    print("  1. visual.environment 可以删除或仅用于特殊场景")
    print("  2. 环境背景统一由 scene_profiles.yaml 管理")
    print("  3. visual 字段只负责：composition, character_pose, fx, motion")
    print("  4. fx 可以为空（只在有特效时填写）")
    print("  5. 这样避免重复，职责更清晰")

if __name__ == "__main__":
    main()

