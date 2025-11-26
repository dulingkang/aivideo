#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 action 字段翻译和填充空的 visual 字段
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List
import shutil

# action 翻译映射
ACTION_TRANSLATIONS = {
    "lying_still": "静止躺着",
    "observe_sky": "观察天空",
    "recall_memory": "回忆",
    "suffer_pain": "承受痛苦",
    "use_technique": "使用法术",
    "turn_head": "转头",
    "observe_sunset": "观察日落",
    "focus_eyes": "聚焦眼神",
    "detect_threat": "察觉威胁",
    "identify_enemy": "识别敌人",
    "prepare_defense": "准备防御",
    "gather_sand": "聚集沙砾",
    "hold_attack": "蓄力攻击",
    "launch_attack": "发动攻击",
    "hit_target": "击中目标",
    "cannibalism": "同类相食",
    "observe_transition": "观察变化",
    "hear_sound": "听到声音",
    "observe_bird": "观察鸟类",
    "observe_riders": "观察骑士",
}

def extract_environment_from_description(description: str) -> str:
    """从 description 中提取环境信息"""
    if not description:
        return ""
    
    # 环境关键词
    env_keywords = [
        "沙地", "沙砾", "青灰色", "一望无际", "沙漠",
        "天空", "太阳", "月亮", "星辰", "弯月", "虚影",
        "高空", "低空", "上空", "地面", "远处", "附近", "空中",
        "黯淡", "明亮", "昏暗", "皎洁", "夺目", "朦胧",
        "轰隆隆", "震动", "声音", "尖鸣", "怪鸟", "黑点",
        "骑士", "怪兽", "小鸟", "盘旋"
    ]
    
    extracted_parts = []
    # 移除角色相关的词
    text_clean = description
    role_keywords = ["韩立", "躺在", "躺", "一动不动", "感受", "回忆", "脸色", "难看",
                     "睁大", "双目", "注视", "偏动", "头颅", "胸膛", "一鼓", "一瘪",
                     "神色", "一变", "听到", "看清", "看到", "观察", "凝视", "使用",
                     "元婴", "虚化", "精元", "灌注", "身体", "费劲", "偏动", "头颅",
                     "喷出", "强风", "激射", "击中", "发出", "惨叫", "撕裂", "分尸",
                     "变幻", "形态", "出现", "盘旋", "震动", "飞驰", "嘶吼"]
    
    for role_kw in role_keywords:
        text_clean = text_clean.replace(role_kw, "")
    
    # 提取环境关键词
    for kw in sorted(env_keywords, key=len, reverse=True):
        if kw in text_clean:
            # 提取包含关键词的短语
            idx = text_clean.find(kw)
            if idx >= 0:
                start = max(0, idx - 5)
                while start > 0 and text_clean[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                end = min(len(text_clean), idx + len(kw) + 10)
                while end < len(text_clean) and text_clean[end] not in "，。！？、":
                    end += 1
                
                phrase = text_clean[start:end].strip()
                phrase = re.sub(r'[，。！？、]+', '，', phrase).strip('，').strip()
                
                if phrase and phrase not in extracted_parts and 3 <= len(phrase) <= 20:
                    extracted_parts.append(phrase)
                    if len(extracted_parts) >= 2:
                        break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_character_pose_from_description(description: str) -> str:
    """从 description 中提取角色动作"""
    if not description:
        return ""
    
    # 角色动作关键词
    pose_keywords = [
        "韩立躺在", "一动不动", "韩立回忆", "脸色难看", "韩立睁大双目",
        "注视", "韩立费劲地偏动头颅", "韩立头颅一偏", "韩立胸膛一鼓",
        "韩立神色一变", "韩立看清", "韩立看到", "听到", "感受",
        "喷出", "激射", "转动", "使用", "元婴虚化", "灌注"
    ]
    
    extracted_parts = []
    for kw in sorted(pose_keywords, key=len, reverse=True):
        if kw in description:
            idx = description.find(kw)
            if idx >= 0:
                start = max(0, idx - 5)
                while start > 0 and description[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                end = min(len(description), idx + len(kw) + 10)
                while end < len(description) and description[end] not in "，。！？、":
                    end += 1
                
                phrase = description[start:end].strip()
                # 移除环境相关的词
                env_keywords = ["沙地", "沙砾", "青灰色", "天空", "太阳", "月亮", "地面", "高空"]
                for env_kw in env_keywords:
                    phrase = phrase.replace(env_kw, "")
                phrase = re.sub(r'[，。！？、]+', '，', phrase).strip('，').strip()
                
                if phrase and phrase not in extracted_parts and 3 <= len(phrase) <= 20:
                    extracted_parts.append(phrase)
                    if len(extracted_parts) >= 2:
                        break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def fix_scene(scene: Dict[str, Any]) -> List[str]:
    """修复单个场景"""
    changes = []
    scene_id = scene.get("id", "未知")
    
    # 1. 翻译 action 字段
    if "action" in scene and scene["action"]:
        old_action = scene["action"]
        new_action = ACTION_TRANSLATIONS.get(old_action, old_action)
        if new_action != old_action:
            scene["action"] = new_action
            changes.append(f"action: {old_action} -> {new_action}")
    
    # 2. 填充空的 visual 字段
    description = scene.get("description", "")
    visual = scene.get("visual", {}) or {}
    if not isinstance(visual, dict):
        visual = {}
    
    # 检查 environment（应该总是有内容，除非是无环境场景）
    if not visual.get("environment") and description:
        new_env = extract_environment_from_description(description)
        if new_env:
            visual["environment"] = new_env
            changes.append(f"environment: (空) -> {new_env}")
    
    # 检查 character_pose（如果有角色，应该有内容）
    if not visual.get("character_pose") and description and "韩立" in description:
        new_pose = extract_character_pose_from_description(description)
        if new_pose:
            visual["character_pose"] = new_pose
            changes.append(f"character_pose: (空) -> {new_pose}")
    
    # fx 可以为空，不需要强制填充
    
    if changes:
        scene["visual"] = visual
    
    return changes

def main():
    parser = argparse.ArgumentParser(description="修复 action 字段翻译和填充空的 visual 字段")
    parser.add_argument("--input", "-i", required=True, help="输入的 JSON 文件路径")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际修改文件")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return
    
    # 读取文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 备份原文件
    if not args.dry_run:
        backup_path = input_path.with_suffix('.json.bak5')
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)
            print(f"✓ 已备份原文件: {backup_path}")
    
    all_changes = []
    
    # 修复 scenes
    if "scenes" in data and isinstance(data["scenes"], list):
        for scene in data["scenes"]:
            changes = fix_scene(scene)
            if changes:
                scene_id = scene.get("id", "未知")
                all_changes.append(f"\n场景 {scene_id}:")
                all_changes.extend([f"  ✓ {change}" for change in changes])
    
    # 显示修改摘要
    if all_changes:
        print("\n" + "=" * 60)
        print("修复摘要")
        print("=" * 60)
        for change in all_changes:
            print(change)
        
        # 保存文件
        if not args.dry_run:
            with open(input_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 已保存修复后的文件: {input_path}")
        else:
            print(f"\n⚠ 预览模式，未实际修改文件")
    else:
        print("\n✓ 未发现需要修复的内容")

if __name__ == "__main__":
    main()

