#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门优化 visual 字段，从 description 中智能提取并填充正确的中文内容
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List
import shutil

def has_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def extract_environment(text: str) -> str:
    """从描述中提取环境信息（场景、背景、天空、地面等）"""
    if not text:
        return ""
    
    # 环境关键词（按优先级排序，长关键词优先）
    env_keywords = [
        "青灰色沙地", "青灰色沙砾", "一望无际", "三个夺目的太阳", "四个朦胧的月亮",
        "天空", "太阳", "月亮", "高空", "低空", "上空", "地面", "远处", "附近", "空中",
        "黯淡", "明亮", "皎洁", "夺目", "朦胧", "虚影", "弯月"
    ]
    
    extracted_parts = []
    used_indices = set()
    
    # 按关键词长度排序，优先匹配长关键词
    for kw in sorted(env_keywords, key=len, reverse=True):
        if kw in text:
            # 找到所有匹配位置
            for match in re.finditer(re.escape(kw), text):
                idx = match.start()
                if idx in used_indices:
                    continue
                
                # 提取包含关键词的短语（向前向后各扩展一些字符）
                start = max(0, idx - 8)
                while start > 0 and text[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                end = min(len(text), idx + len(kw) + 12)
                while end < len(text) and text[end] not in "，。！？、":
                    end += 1
                
                phrase = text[start:end].strip()
                # 清理短语，移除重复的关键词
                phrase = re.sub(r'(.+?)\1+', r'\1', phrase)  # 移除重复
                
                if phrase and phrase not in extracted_parts and 5 <= len(phrase) <= 22:
                    extracted_parts.append(phrase)
                    # 标记这个区域已被使用
                    for i in range(start, end):
                        used_indices.add(i)
                    if len(extracted_parts) >= 2:
                        break
            if len(extracted_parts) >= 2:
                break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_character_pose(text: str) -> str:
    """从描述中提取角色姿势/动作信息"""
    if not text:
        return ""
    
    # 角色动作关键词（按优先级排序，长关键词优先）
    pose_keywords = [
        "韩立躺在", "一动不动", "韩立回忆", "脸色难看", "韩立睁大双目",
        "注视", "韩立费劲地偏动头颅", "韩立头颅一偏", "韩立胸膛一鼓",
        "韩立神色一变", "韩立看清", "韩立看到", "听到", "感受"
    ]
    
    extracted_parts = []
    used_indices = set()
    
    # 按关键词长度排序，优先匹配长关键词
    for kw in sorted(pose_keywords, key=len, reverse=True):
        if kw in text:
            # 找到所有匹配位置
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
                # 清理短语，移除重复
                phrase = re.sub(r'(.+?)\1+', r'\1', phrase)
                
                if phrase and phrase not in extracted_parts and 5 <= len(phrase) <= 22:
                    extracted_parts.append(phrase)
                    # 标记这个区域已被使用
                    for i in range(start, end):
                        used_indices.add(i)
                    if len(extracted_parts) >= 2:
                        break
            if len(extracted_parts) >= 2:
                break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_fx(text: str) -> str:
    """从描述中提取特效信息（光芒、声音、动作效果等）"""
    if not text:
        return ""
    
    # 特效关键词（按优先级排序，长关键词优先）
    fx_keywords = [
        "蓝芒闪动", "青芒激射", "白濛濛", "轰隆隆", "凄厉尖鸣",
        "金属摩擦", "惨叫", "震动", "轻微震动", "盘旋", "盘旋不定",
        "清鸣", "悦耳清鸣", "撕裂", "分尸", "撕裂分尸",
        "变幻形态", "滴溜溜转动", "漫天花雨", "激射而出"
    ]
    
    extracted_parts = []
    used_indices = set()
    
    # 按关键词长度排序，优先匹配长关键词
    for kw in sorted(fx_keywords, key=len, reverse=True):
        if kw in text:
            # 找到所有匹配位置
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
                # 清理短语，移除重复
                phrase = re.sub(r'(.+?)\1+', r'\1', phrase)
                
                if phrase and phrase not in extracted_parts and 5 <= len(phrase) <= 22:
                    extracted_parts.append(phrase)
                    # 标记这个区域已被使用
                    for i in range(start, end):
                        used_indices.add(i)
                    if len(extracted_parts) >= 2:
                        break
            if len(extracted_parts) >= 2:
                break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_composition(text: str) -> str:
    """从描述中提取构图信息（整体画面描述，简化版）"""
    if not text:
        return ""
    
    # 如果描述较短，直接使用
    if len(text) <= 28:
        return text
    
    # 尝试提取前半部分（通常是主要构图信息）
    if "，" in text:
        parts = text.split("，")
        if len(parts) >= 2:
            # 取前两个部分
            composition = "，".join(parts[:2])
            if len(composition) <= 30:
                return composition
        elif len(parts) == 1:
            # 只有一个部分，截断
            return parts[0][:28] + "..."
    
    # 如果还是太长，直接截断
    return text[:28] + "..."

def optimize_visual_field(scene: Dict[str, Any]) -> List[str]:
    """
    优化 visual 字段，从 description 中智能提取并填充
    
    Returns:
        修改列表
    """
    changes = []
    description = scene.get("description", "")
    
    if not description or not has_chinese(description):
        return changes
    
    visual = scene.get("visual", {}) or {}
    if not isinstance(visual, dict):
        visual = {}
    
    # 提取各个字段
    new_composition = extract_composition(description)
    new_environment = extract_environment(description)
    new_character_pose = extract_character_pose(description)
    new_fx = extract_fx(description)
    
    # 更新 composition（整体构图）
    if new_composition:
        old_comp = visual.get("composition", "")
        if old_comp != new_composition:
            visual["composition"] = new_composition
            changes.append(f"composition: {old_comp[:35] if old_comp else '(空)'}... -> {new_composition[:35]}...")
    
    # 更新 environment（环境背景）
    if new_environment:
        old_env = visual.get("environment", "")
        if old_env != new_environment:
            visual["environment"] = new_environment
            changes.append(f"environment: {old_env[:35] if old_env else '(空)'}... -> {new_environment[:35]}...")
    elif not new_environment and visual.get("environment"):
        # 如果没有提取到环境信息，但原来有且和 description 相同，清空
        if visual.get("environment") == description:
            visual["environment"] = ""
            changes.append(f"environment: 已清空（与 description 相同）")
    
    # 更新 character_pose（角色动作）
    if new_character_pose:
        old_pose = visual.get("character_pose", "")
        if old_pose != new_character_pose:
            visual["character_pose"] = new_character_pose
            changes.append(f"character_pose: {old_pose[:35] if old_pose else '(空)'}... -> {new_character_pose[:35]}...")
    elif not new_character_pose and visual.get("character_pose"):
        # 如果没有提取到动作信息，但原来有且和 description 相同，清空
        if visual.get("character_pose") == description:
            visual["character_pose"] = ""
            changes.append(f"character_pose: 已清空（与 description 相同）")
    
    # 更新 fx（特效，可以为空）
    old_fx = visual.get("fx", "")
    if new_fx:
        if old_fx != new_fx:
            visual["fx"] = new_fx
            changes.append(f"fx: {old_fx[:35] if old_fx else '(空)'}... -> {new_fx[:35]}...")
    else:
        # 如果没有特效，清空（如果原来和 description 相同）
        if old_fx == description or (old_fx and old_fx in description):
            visual["fx"] = ""
            changes.append(f"fx: {old_fx[:35] if old_fx else '(空)'}... -> (已清空)")
    
    # 保持 motion 字段不变（如果存在）
    
    scene["visual"] = visual
    return changes

def main():
    parser = argparse.ArgumentParser(description="优化场景 JSON 文件中的 visual 字段")
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
        backup_path = input_path.with_suffix('.json.bak2')
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)
            print(f"✓ 已备份原文件: {backup_path}")
    
    all_changes = []
    
    # 优化 scenes
    if "scenes" in data and isinstance(data["scenes"], list):
        for scene in data["scenes"]:
            scene_id = scene.get("id", "未知")
            changes = optimize_visual_field(scene)
            if changes:
                all_changes.append(f"\n场景 {scene_id}:")
                all_changes.extend([f"  ✓ {change}" for change in changes])
    
    # 显示修改摘要
    if all_changes:
        print("\n" + "=" * 60)
        print("Visual 字段优化摘要")
        print("=" * 60)
        for change in all_changes:
            print(change)
        
        # 保存文件
        if not args.dry_run:
            with open(input_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 已保存优化后的文件: {input_path}")
        else:
            print(f"\n⚠ 预览模式，未实际修改文件")
    else:
        print("\n✓ 未发现需要优化的内容")

if __name__ == "__main__":
    main()

