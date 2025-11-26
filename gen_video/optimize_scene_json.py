#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化场景 JSON 文件：
1. 缩短过长的 narration（保留核心信息）
2. 优化 visual 字段，从 description 中智能提取并填充正确的中文内容
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil

def has_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def shorten_narration(narration: str, max_length: int = 30) -> str:
    """
    缩短过长的 narration，保留核心信息
    
    Args:
        narration: 原始旁白
        max_length: 最大字符数（默认30，约3-4秒）
    
    Returns:
        缩短后的旁白
    """
    if not narration:
        return narration
    
    narration = narration.strip()
    
    # 如果已经足够短，直接返回
    if len(narration) <= max_length:
        return narration
    
    # 尝试按句号、逗号、感叹号等分割
    sentences = re.split(r'[。！？，,；;]', narration)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return narration[:max_length] + "..."
    
    # 优先保留第一句（通常是核心信息）
    result = sentences[0]
    
    # 如果第一句已经超过限制，直接截断
    if len(result) > max_length:
        # 尝试保留前半部分
        if "，" in result:
            parts = result.split("，")
            result = parts[0]
            if len(result) > max_length:
                return result[:max_length-2] + "..."
        else:
            return result[:max_length-2] + "..."
    
    # 如果第一句不够长，尝试添加第二句
    if len(sentences) > 1:
        next_sentence = sentences[1]
        if len(result + "，" + next_sentence) <= max_length:
            result = result + "，" + next_sentence
        elif len(result + "。" + next_sentence) <= max_length:
            result = result + "。" + next_sentence
    
    # 如果还是太长，截断
    if len(result) > max_length:
        result = result[:max_length-2] + "..."
    
    return result

def extract_environment(text: str) -> str:
    """从描述中提取环境信息"""
    if not text:
        return ""
    
    # 环境关键词（按优先级排序）
    env_keywords = [
        "沙漠", "沙地", "沙砾", "青灰色沙地", "青灰色沙砾", "一望无际",
        "天空", "太阳", "月亮", "星辰", "弯月", "虚影", "高空", "低空", "上空",
        "地面", "远处", "附近", "方向",
        "黯淡", "明亮", "昏暗", "皎洁", "夺目", "朦胧"
    ]
    
    extracted_parts = []
    found_keywords = set()
    
    # 优先提取长关键词
    for kw in sorted(env_keywords, key=len, reverse=True):
        if kw in text and kw not in found_keywords:
            # 提取包含关键词的短语（向前向后各扩展一些字符）
            idx = text.find(kw)
            if idx >= 0:
                # 向前找起始位置（句首、逗号、或前5个字符）
                start = max(0, idx - 5)
                while start > 0 and text[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                # 向后找结束位置（句末、逗号、或后10个字符）
                end = min(len(text), idx + len(kw) + 10)
                while end < len(text) and text[end] not in "，。！？、":
                    end += 1
                
                phrase = text[start:end].strip()
                if phrase and phrase not in extracted_parts and len(phrase) <= 25:
                    extracted_parts.append(phrase)
                    found_keywords.add(kw)
                    if len(extracted_parts) >= 2:  # 最多2个短语
                        break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_character_pose(text: str) -> str:
    """从描述中提取角色姿势/动作信息"""
    if not text:
        return ""
    
    # 角色动作关键词（按优先级排序）
    pose_keywords = [
        "韩立", "躺在", "躺", "一动不动", "感受", "回忆", "脸色难看", "脸色", "难看",
        "睁大双目", "睁大", "双目", "注视", "凝视", "注视高空",
        "偏动头颅", "偏动", "头颅", "一偏", "一瘪", "一鼓",
        "胸膛一鼓", "胸膛一瘪", "胸脯一瘪",
        "神色一变", "神色", "一变", "听到", "看清", "看到"
    ]
    
    extracted_parts = []
    found_keywords = set()
    
    # 优先提取长关键词
    for kw in sorted(pose_keywords, key=len, reverse=True):
        if kw in text and kw not in found_keywords:
            idx = text.find(kw)
            if idx >= 0:
                # 向前找起始位置
                start = max(0, idx - 5)
                while start > 0 and text[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                # 向后找结束位置
                end = min(len(text), idx + len(kw) + 10)
                while end < len(text) and text[end] not in "，。！？、":
                    end += 1
                
                phrase = text[start:end].strip()
                if phrase and phrase not in extracted_parts and len(phrase) <= 25:
                    extracted_parts.append(phrase)
                    found_keywords.add(kw)
                    if len(extracted_parts) >= 2:  # 最多2个短语
                        break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_fx(text: str) -> str:
    """从描述中提取特效信息"""
    if not text:
        return ""
    
    # 特效关键词（按优先级排序）
    fx_keywords = [
        "蓝芒闪动", "蓝芒", "闪动", "青芒激射", "青芒", "激射", 
        "白濛濛强风", "白濛濛", "强风", "轰隆隆", "尖鸣", "凄厉尖鸣",
        "金属摩擦般", "金属摩擦", "惨叫", "震动", "轻微震动",
        "盘旋不定", "盘旋", "清鸣", "悦耳清鸣",
        "撕裂分尸", "撕裂", "分尸",
        "变幻形态", "变幻", "黯淡", "夺目", "朦胧", "虚影", "皎洁", "弯月",
        "滴溜溜转动", "密密麻麻", "漫天花雨"
    ]
    
    extracted_parts = []
    found_keywords = set()
    
    # 优先提取长关键词
    for kw in sorted(fx_keywords, key=len, reverse=True):
        if kw in text and kw not in found_keywords:
            idx = text.find(kw)
            if idx >= 0:
                # 向前找起始位置
                start = max(0, idx - 5)
                while start > 0 and text[start] not in "，。！？、":
                    start -= 1
                if start > 0:
                    start += 1
                
                # 向后找结束位置
                end = min(len(text), idx + len(kw) + 10)
                while end < len(text) and text[end] not in "，。！？、":
                    end += 1
                
                phrase = text[start:end].strip()
                if phrase and phrase not in extracted_parts and len(phrase) <= 25:
                    extracted_parts.append(phrase)
                    found_keywords.add(kw)
                    if len(extracted_parts) >= 2:  # 最多2个短语
                        break
    
    if extracted_parts:
        return "，".join(extracted_parts[:2])
    return ""

def extract_composition(text: str) -> str:
    """从描述中提取构图信息（整体画面描述）"""
    if not text:
        return ""
    
    # 构图通常包含主体和背景的关系
    # 如果描述较短，直接使用；如果较长，提取核心部分
    if len(text) <= 28:
        return text
    
    # 尝试提取前半部分（通常是主要构图信息）
    if "，" in text:
        parts = text.split("，")
        if len(parts) >= 2:
            # 取前两个部分（通常是主体+背景）
            composition = "，".join(parts[:2])
            if len(composition) <= 30:
                return composition
        elif len(parts) == 1:
            # 只有一个部分，截断到合适长度
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
    
    # 更新 composition
    if new_composition and visual.get("composition") != new_composition:
        old_comp = visual.get("composition", "")
        visual["composition"] = new_composition
        if old_comp != new_composition:
            changes.append(f"visual.composition: {old_comp[:40] if old_comp else '(空)'}... -> {new_composition[:40]}...")
    
    # 更新 environment
    if new_environment and visual.get("environment") != new_environment:
        old_env = visual.get("environment", "")
        visual["environment"] = new_environment
        if old_env != new_environment:
            changes.append(f"visual.environment: {old_env[:40] if old_env else '(空)'}... -> {new_environment[:40]}...")
    
    # 更新 character_pose
    if new_character_pose and visual.get("character_pose") != new_character_pose:
        old_pose = visual.get("character_pose", "")
        visual["character_pose"] = new_character_pose
        if old_pose != new_character_pose:
            changes.append(f"visual.character_pose: {old_pose[:40] if old_pose else '(空)'}... -> {new_character_pose[:40]}...")
    
    # 更新 fx（可以为空）
    if visual.get("fx") != new_fx:
        old_fx = visual.get("fx", "")
        visual["fx"] = new_fx
        if old_fx != new_fx:
            if new_fx:
                changes.append(f"visual.fx: {old_fx[:40] if old_fx else '(空)'}... -> {new_fx[:40]}...")
            else:
                changes.append(f"visual.fx: {old_fx[:40] if old_fx else '(空)'}... -> (已清空)")
    
    # 保持 motion 字段不变（如果存在）
    if "motion" not in visual and scene.get("camera"):
        # 可以根据 camera 推断 motion，但这里先不处理
        pass
    
    scene["visual"] = visual
    return changes

def optimize_scene(scene: Dict[str, Any], max_narration_length: int = 30) -> Dict[str, List[str]]:
    """
    优化单个场景
    
    Returns:
        {"narration_changes": [...], "visual_changes": [...]}
    """
    changes = {"narration_changes": [], "visual_changes": []}
    scene_id = scene.get("id", "未知")
    
    # 优化 narration
    narration = scene.get("narration", "")
    if narration:
        old_narration = narration
        new_narration = shorten_narration(narration, max_narration_length)
        if new_narration != old_narration:
            scene["narration"] = new_narration
            changes["narration_changes"].append(
                f"场景 {scene_id}: {len(old_narration)}字 -> {len(new_narration)}字"
            )
            changes["narration_changes"].append(f"  原文: {old_narration[:50]}...")
            changes["narration_changes"].append(f"  优化: {new_narration}")
    
    # 优化 visual 字段
    visual_changes = optimize_visual_field(scene)
    if visual_changes:
        changes["visual_changes"].extend([f"场景 {scene_id}:"] + visual_changes)
    
    return changes

def main():
    parser = argparse.ArgumentParser(description="优化场景 JSON 文件：缩短 narration 并优化 visual 字段")
    parser.add_argument("--input", "-i", required=True, help="输入的 JSON 文件路径")
    parser.add_argument("--max-narration-length", "-m", type=int, default=30, 
                       help="narration 最大字符数（默认30，约3-4秒）")
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
        backup_path = input_path.with_suffix('.json.bak')
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)
            print(f"✓ 已备份原文件: {backup_path}")
    
    all_narration_changes = []
    all_visual_changes = []
    
    # 优化 opening
    if "opening" in data and data["opening"].get("narration"):
        old_narr = data["opening"]["narration"]
        new_narr = shorten_narration(old_narr, args.max_narration_length)
        if new_narr != old_narr:
            if not args.dry_run:
                data["opening"]["narration"] = new_narr
            all_narration_changes.append(f"开头: {len(old_narr)}字 -> {len(new_narr)}字")
            all_narration_changes.append(f"  原文: {old_narr}")
            all_narration_changes.append(f"  优化: {new_narr}")
    
    # 优化 scenes
    if "scenes" in data and isinstance(data["scenes"], list):
        for scene in data["scenes"]:
            changes = optimize_scene(scene, args.max_narration_length)
            if changes["narration_changes"]:
                all_narration_changes.extend(changes["narration_changes"])
            if changes["visual_changes"]:
                all_visual_changes.extend(changes["visual_changes"])
    
    # 优化 ending
    if "ending" in data and data["ending"].get("narration"):
        old_narr = data["ending"]["narration"]
        new_narr = shorten_narration(old_narr, args.max_narration_length)
        if new_narr != old_narr:
            if not args.dry_run:
                data["ending"]["narration"] = new_narr
            all_narration_changes.append(f"结尾: {len(old_narr)}字 -> {len(new_narr)}字")
            all_narration_changes.append(f"  原文: {old_narr}")
            all_narration_changes.append(f"  优化: {new_narr}")
    
    # 显示修改摘要
    if all_narration_changes or all_visual_changes:
        print("\n" + "=" * 60)
        print("优化摘要")
        print("=" * 60)
        
        if all_narration_changes:
            print("\n📝 Narration 优化:")
            for change in all_narration_changes:
                print(f"  {change}")
        
        if all_visual_changes:
            print("\n🎨 Visual 字段优化:")
            for change in all_visual_changes:
                print(f"  {change}")
        
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

