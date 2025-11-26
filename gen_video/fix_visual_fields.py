#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 visual 字段，确保各字段内容准确：
- composition: 整体构图（简化版，主体+背景关系）
- environment: 只包含环境背景（场景、天空、地面等）
- character_pose: 只包含角色动作/姿势（不包含环境）
- fx: 只包含特效（光芒、声音、动作效果等）
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List
import shutil

def extract_environment_only(text: str) -> str:
    """只提取环境背景信息，不包含角色动作"""
    if not text:
        return ""
    
    # 环境关键词（场景、背景、天空、地面等）
    env_keywords = [
        "沙地", "沙砾", "青灰色", "一望无际", "沙漠",
        "天空", "太阳", "月亮", "星辰", "弯月", "虚影",
        "高空", "低空", "上空", "地面", "远处", "附近", "空中",
        "黯淡", "明亮", "昏暗", "皎洁", "夺目", "朦胧",
        "轰隆隆", "震动", "声音"
    ]
    
    extracted_parts = []
    used_indices = set()
    
    # 移除角色相关的词
    text_clean = text
    role_keywords = ["韩立", "躺在", "躺", "一动不动", "感受", "回忆", "脸色", "难看",
                     "睁大", "双目", "注视", "偏动", "头颅", "胸膛", "一鼓", "一瘪",
                     "神色", "一变", "听到", "看清", "看到", "观察", "凝视"]
    
    # 按关键词长度排序，优先匹配长关键词
    for kw in sorted(env_keywords, key=len, reverse=True):
        if kw in text_clean:
            for match in re.finditer(re.escape(kw), text_clean):
                idx = match.start()
                if idx in used_indices:
                    continue
                
                # 提取包含关键词的短语
                start = max(0, idx - 8)
                while start > 0 and text_clean[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                end = min(len(text_clean), idx + len(kw) + 12)
                while end < len(text_clean) and text_clean[end] not in "，。！？、":
                    end += 1
                
                phrase = text_clean[start:end].strip()
                # 移除角色相关的词
                for role_kw in role_keywords:
                    phrase = phrase.replace(role_kw, "")
                phrase = re.sub(r'[，。！？、]+', '，', phrase).strip('，').strip()
                
                if phrase and phrase not in extracted_parts and 3 <= len(phrase) <= 20:
                    extracted_parts.append(phrase)
                    for i in range(start, end):
                        used_indices.add(i)
                    if len(extracted_parts) >= 2:
                        break
            if len(extracted_parts) >= 2:
                break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_character_pose_only(text: str) -> str:
    """只提取角色动作/姿势，不包含环境"""
    if not text:
        return ""
    
    # 角色动作关键词
    pose_keywords = [
        "韩立躺在", "一动不动", "韩立回忆", "脸色难看", "韩立睁大双目",
        "注视", "韩立费劲地偏动头颅", "韩立头颅一偏", "韩立胸膛一鼓",
        "韩立神色一变", "韩立看清", "韩立看到", "听到", "感受",
        "喷出", "激射", "转动"
    ]
    
    extracted_parts = []
    used_indices = set()
    
    # 按关键词长度排序，优先匹配长关键词
    for kw in sorted(pose_keywords, key=len, reverse=True):
        if kw in text:
            for match in re.finditer(re.escape(kw), text):
                idx = match.start()
                if idx in used_indices:
                    continue
                
                # 提取包含关键词的短语
                start = max(0, idx - 5)
                while start > 0 and text[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                end = min(len(text), idx + len(kw) + 10)
                while end < len(text) and text[end] not in "，。！？、":
                    end += 1
                
                phrase = text[start:end].strip()
                # 移除环境相关的词
                env_keywords = ["沙地", "沙砾", "青灰色", "天空", "太阳", "月亮", "地面", "高空"]
                for env_kw in env_keywords:
                    phrase = phrase.replace(env_kw, "")
                phrase = re.sub(r'[，。！？、]+', '，', phrase).strip('，').strip()
                
                if phrase and phrase not in extracted_parts and 3 <= len(phrase) <= 20:
                    extracted_parts.append(phrase)
                    for i in range(start, end):
                        used_indices.add(i)
                    if len(extracted_parts) >= 2:
                        break
            if len(extracted_parts) >= 2:
                break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_composition_simple(text: str) -> str:
    """提取简化的构图（主体+背景关系）"""
    if not text:
        return ""
    
    # 如果描述较短，直接使用
    if len(text) <= 25:
        return text
    
    # 尝试提取前半部分（通常是主要构图信息）
    if "，" in text:
        parts = text.split("，")
        if len(parts) >= 2:
            # 取前两个部分，但简化
            composition = "，".join(parts[:2])
            if len(composition) <= 28:
                return composition
        elif len(parts) == 1:
            return parts[0][:28] + "..."
    
    # 如果还是太长，直接截断
    return text[:28] + "..."

def extract_fx_only(text: str) -> str:
    """只提取特效信息"""
    if not text:
        return ""
    
    # 特效关键词
    fx_keywords = [
        "蓝芒闪动", "青芒激射", "白濛濛", "轰隆隆", "凄厉尖鸣",
        "金属摩擦", "惨叫", "震动", "轻微震动", "盘旋", "盘旋不定",
        "清鸣", "悦耳清鸣", "撕裂", "分尸", "撕裂分尸",
        "变幻形态", "滴溜溜转动", "漫天花雨", "激射而出", "强风"
    ]
    
    extracted_parts = []
    used_indices = set()
    
    # 按关键词长度排序，优先匹配长关键词
    for kw in sorted(fx_keywords, key=len, reverse=True):
        if kw in text:
            for match in re.finditer(re.escape(kw), text):
                idx = match.start()
                if idx in used_indices:
                    continue
                
                # 提取包含关键词的短语
                start = max(0, idx - 5)
                while start > 0 and text[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                end = min(len(text), idx + len(kw) + 10)
                while end < len(text) and text[end] not in "，。！？、":
                    end += 1
                
                phrase = text[start:end].strip()
                if phrase and phrase not in extracted_parts and 3 <= len(phrase) <= 20:
                    extracted_parts.append(phrase)
                    for i in range(start, end):
                        used_indices.add(i)
                    if len(extracted_parts) >= 2:
                        break
            if len(extracted_parts) >= 2:
                break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def fix_visual_field(scene: Dict[str, Any]) -> List[str]:
    """修复 visual 字段，确保各字段内容准确"""
    changes = []
    description = scene.get("description", "")
    
    if not description:
        return changes
    
    visual = scene.get("visual", {}) or {}
    if not isinstance(visual, dict):
        visual = {}
    
    # 重新提取各字段
    new_composition = extract_composition_simple(description)
    new_environment = extract_environment_only(description)
    new_character_pose = extract_character_pose_only(description)
    new_fx = extract_fx_only(description)
    
    # 更新 composition
    if new_composition and visual.get("composition") != new_composition:
        old_comp = visual.get("composition", "")
        visual["composition"] = new_composition
        changes.append(f"composition: {old_comp[:35] if old_comp else '(空)'}... -> {new_composition[:35]}...")
    
    # 更新 environment（只包含环境背景）
    old_env = visual.get("environment", "")
    if new_environment:
        if old_env != new_environment:
            visual["environment"] = new_environment
            changes.append(f"environment: {old_env[:35] if old_env else '(空)'}... -> {new_environment[:35]}...")
    else:
        # 如果没有提取到环境信息，清空
        if old_env:
            visual["environment"] = ""
            changes.append(f"environment: {old_env[:35]}... -> (已清空)")
    
    # 更新 character_pose（只包含角色动作）
    old_pose = visual.get("character_pose", "")
    if new_character_pose:
        if old_pose != new_character_pose:
            visual["character_pose"] = new_character_pose
            changes.append(f"character_pose: {old_pose[:35] if old_pose else '(空)'}... -> {new_character_pose[:35]}...")
    else:
        # 如果没有提取到动作信息，清空
        if old_pose:
            visual["character_pose"] = ""
            changes.append(f"character_pose: {old_pose[:35]}... -> (已清空)")
    
    # 更新 fx（只包含特效）
    old_fx = visual.get("fx", "")
    if new_fx:
        if old_fx != new_fx:
            visual["fx"] = new_fx
            changes.append(f"fx: {old_fx[:35] if old_fx else '(空)'}... -> {new_fx[:35]}...")
    else:
        # 如果没有特效，清空
        if old_fx:
            visual["fx"] = ""
            changes.append(f"fx: {old_fx[:35]}... -> (已清空)")
    
    scene["visual"] = visual
    return changes

def main():
    parser = argparse.ArgumentParser(description="修复 visual 字段，确保各字段内容准确")
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
        backup_path = input_path.with_suffix('.json.bak4')
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)
            print(f"✓ 已备份原文件: {backup_path}")
    
    all_changes = []
    
    # 修复 scenes
    if "scenes" in data and isinstance(data["scenes"], list):
        for scene in data["scenes"]:
            scene_id = scene.get("id", "未知")
            changes = fix_visual_field(scene)
            if changes:
                all_changes.append(f"\n场景 {scene_id}:")
                all_changes.extend([f"  ✓ {change}" for change in changes])
    
    # 显示修改摘要
    if all_changes:
        print("\n" + "=" * 60)
        print("Visual 字段修复摘要")
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

