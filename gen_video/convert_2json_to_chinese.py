#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将 2.json 转换为中文：
1. 转换元数据字段（mood, lighting, camera, action, motion, face_style_auto）
2. 智能提取 visual 字段（composition, environment, character_pose, fx）
3. 优化 narration 长度
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

def has_chinese(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def shorten_narration(narration: str, max_length: int = 30) -> str:
    """缩短旁白，保留核心信息"""
    if not narration:
        return ""
    
    if len(narration) > max_length:
        # 优先按句号分割
        parts = re.split(r'[。！？]', narration)
        if parts and len(parts[0]) <= max_length:
            return parts[0].strip()
        
        # 其次按逗号分割
        parts = re.split(r'[，]', narration)
        if parts and len(parts[0]) <= max_length:
            return parts[0].strip()
        
        # 最后直接截断
        return narration[:max_length].strip()
    return narration.strip()

# 元数据翻译映射
mood_map = {
    "alert": "警觉", "calm": "平静", "surprised": "惊讶", "confused": "困惑",
    "curious": "好奇", "mysterious": "神秘", "intense": "激烈", "realization": "领悟",
    "serious": "严肃", "analytical": "分析", "majestic": "宏伟"
}

lighting_map = {
    "night": "夜晚", "day": "白天", "dusk": "黄昏", "sunset": "日落",
    "bright_normal": "明亮正常", "soft": "柔和", "dramatic": "戏剧性"
}

camera_map = {
    "close_up_disk": "圆盘特写", "medium_shot": "中景", "close_up_face": "面部特写",
    "medium_shot_two": "双人中景", "close_up_box": "木盒特写", "close_up_activation": "激活特写",
    "close_up_body": "身体特写", "close_up_head": "头部特写", "close_up_hand": "手部特写",
    "close_up_paper": "文书特写", "close_up_wrist": "手腕特写", "close_up_sword": "剑特写",
    "macro_shot": "微距", "close_up_blood": "血液特写", "close_up_contract": "契约特写",
    "wide_shot_turtle": "陆行龟全景"
}

action_map = {
    "detect_with_disk": "用圆盘探测", "endure_detection": "承受探测", "react_to_result": "对结果反应",
    "try_communicate": "尝试沟通", "recognize_language": "识别语言", "take_out_box": "取出木盒",
    "activate_pearl": "激活圆珠", "receive_pearl": "接收圆珠", "learn_language": "学习语言",
    "confirm_location": "确认位置", "agree_to_contract": "同意契约", "write_contract": "书写契约",
    "analyze_contract": "分析契约", "try_cut": "尝试切割", "confirm_cultivation": "确认修炼",
    "use_spiritual_tool": "使用灵具", "discover_secret": "发现奥秘", "write_blood_rune": "书写血咒符文",
    "contract_activate": "契约激活", "observe_giant_turtle": "观察巨龟"
}

motion_map = {
    "close-up of disk and light beam": "圆盘和光束特写",
    "medium shot of detection process": "探测过程中景",
    "close-up of surprised reaction": "惊讶反应特写",
    "medium shot of communication attempt": "沟通尝试中景",
    "close-up of recognition moment": "识别瞬间特写",
    "close-up of box and pearl": "木盒和圆珠特写",
    "close-up of activation moment": "激活瞬间特写",
    "close-up of pearl entering body": "圆珠进入身体特写",
    "close-up of learning process": "学习过程特写",
    "medium shot of conversation": "对话中景",
    "medium shot of negotiation": "谈判中景",
    "close-up of writing process": "书写过程特写",
    "close-up of paper and runes": "文书和符文特写",
    "close-up of cutting attempt": "切割尝试特写",
    "medium shot of confirmation": "确认中景",
    "close-up of sword activation": "剑激活特写",
    "macro shot of spirit stones": "灵石微距",
    "close-up of blood writing": "血液书写特写",
    "close-up of activation process": "激活过程特写",
    "wide shot of approaching turtle": "接近的陆行龟全景"
}

expression_map = {
    "focused": "专注", "calm": "平静", "surprised": "惊讶"
}

detail_map = {
    "natural": "自然", "detailed": "详细", "cinematic": "电影级", "sharp_dynamic": "锐利动态",
    "soft_concentrated": "柔和聚焦", "subtle": "微妙"
}

# Visual 字段智能提取函数
def extract_composition(text: str) -> str:
    """提取构图（整体画面布局）"""
    if not text:
        return ""
    if len(text) <= 25:
        return text
    if "，" in text:
        parts = text.split("，")
        composition = "，".join(parts[:2])
        if len(composition) <= 30:
            return composition
    return text[:25].strip() + "..."

def extract_environment(text: str) -> str:
    """提取环境（只包含环境背景，不包含角色动作和特效）"""
    if not text:
        return ""
    
    # 环境关键词
    env_patterns = [
        r"青罗沙漠[^，。！？]*", r"灵界[^，。！？]*", r"远处[^，。！？]*", r"附近[^，。！？]*",
        r"空中[^，。！？]*", r"地面[^，。！？]*", r"车厢[^，。！？]*", r"龟背[^，。！？]*"
    ]
    
    extracted_parts = []
    for pattern in env_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            phrase = match.strip()
            if phrase and phrase not in extracted_parts and len(phrase) <= 20:
                extracted_parts.append(phrase)
                if len(extracted_parts) >= 2:
                    break
        if len(extracted_parts) >= 2:
            break
    
    return "，".join(extracted_parts[:2]) if extracted_parts else ""

def extract_character_pose(text: str) -> str:
    """提取角色动作（只包含角色动作/姿势/表情，不包含环境）"""
    if not text:
        return ""
    
    # 角色动作关键词
    pose_patterns = [
        r"韩立[^，。！？]*", r"青年[^，。！？]*", r"大汉[^，。！？]*", r"骑士[^，。！？]*",
        r"神色[^，。！？]*", r"表情[^，。！？]*", r"脸色[^，。！？]*", r"惊容[^，。！？]*",
        r"对话[^，。！？]*", r"询问[^，。！？]*", r"答应[^，。！？]*", r"确认[^，。！？]*",
        r"查看[^，。！？]*", r"发现[^，。！？]*", r"刺痛[^，。！？]*", r"平静[^，。！？]*"
    ]
    
    extracted_parts = []
    for pattern in pose_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            phrase = match.strip()
            # 过滤掉包含环境信息的短语
            if any(env_kw in phrase for env_kw in ["沙漠", "灵界", "远处", "车厢", "龟背", "空中", "地面"]):
                continue
            if phrase and phrase not in extracted_parts and len(phrase) <= 20:
                extracted_parts.append(phrase)
                if len(extracted_parts) >= 2:
                    break
        if len(extracted_parts) >= 2:
            break
    
    return "，".join(extracted_parts[:2]) if extracted_parts else ""

def extract_fx(text: str) -> str:
    """提取特效（只包含视觉/听觉特效）"""
    if not text:
        return ""
    
    # 特效关键词
    fx_patterns = [
        r"光柱[^，。！？]*", r"光芒[^，。！？]*", r"灵光[^，。！？]*", r"金芒[^，。！？]*",
        r"血光[^，。！？]*", r"符文[^，。！？]*", r"激射[^，。！？]*", r"大放[^，。！？]*",
        r"飞射[^，。！？]*", r"自燃[^，。！？]*", r"凉气[^，。！？]*", r"刺痛[^，。！？]*",
        r"浮现[^，。！？]*", r"没入[^，。！？]*", r"划破[^，。！？]*", r"流出[^，。！？]*"
    ]
    
    extracted_parts = []
    for pattern in fx_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            phrase = match.strip()
            if phrase and phrase not in extracted_parts and len(phrase) <= 20:
                extracted_parts.append(phrase)
                if len(extracted_parts) >= 2:
                    break
        if len(extracted_parts) >= 2:
            break
    
    return "，".join(extracted_parts[:2]) if extracted_parts else ""

def convert_visual_to_chinese(scene: Dict[str, Any]) -> List[str]:
    """智能提取并填充 visual 字段"""
    changes = []
    description = scene.get("description", "")
    visual = scene.get("visual", {}) or {}
    
    if not isinstance(visual, dict):
        return changes
    
    if not has_chinese(description):
        return changes
    
    # 检查是否所有 visual 字段都与 description 相同
    all_visual_same_as_desc = True
    for field_name in ["composition", "environment", "character_pose", "fx"]:
        if field_name in visual and visual[field_name] != description:
            all_visual_same_as_desc = False
            break
    
    if all_visual_same_as_desc:
        changes.append("⚠ 检测到 visual 字段内容相同，将重新智能提取")
    
    # 智能提取并填充
    # composition
    new_composition = extract_composition(description)
    if visual.get("composition") != new_composition:
        visual["composition"] = new_composition
        changes.append(f"composition: {new_composition[:60]}...")
    
    # environment
    new_environment = extract_environment(description)
    if visual.get("environment") != new_environment:
        visual["environment"] = new_environment
        if new_environment:
            changes.append(f"environment: {new_environment[:60]}...")
        else:
            changes.append("environment: (已清空)")
    
    # character_pose
    new_character_pose = extract_character_pose(description)
    if visual.get("character_pose") != new_character_pose:
        visual["character_pose"] = new_character_pose
        if new_character_pose:
            changes.append(f"character_pose: {new_character_pose[:60]}...")
        else:
            changes.append("character_pose: (已清空)")
    
    # fx
    new_fx = extract_fx(description)
    if new_fx and visual.get("fx") != new_fx:
        visual["fx"] = new_fx
        changes.append(f"fx: {new_fx[:60]}...")
    elif not new_fx and "fx" in visual and visual["fx"] != "":
        visual["fx"] = ""
        changes.append("fx: (已清空)")
    
    scene["visual"] = visual
    return changes

def convert_metadata_to_chinese(scene: Dict[str, Any]) -> List[str]:
    """转换元数据字段为中文"""
    changes = []
    
    # mood
    if scene.get("mood") and scene["mood"] in mood_map:
        old_mood = scene["mood"]
        scene["mood"] = mood_map[old_mood]
        changes.append(f"mood: {old_mood} -> {scene['mood']}")
    
    # lighting
    if scene.get("lighting") and scene["lighting"] in lighting_map:
        old_lighting = scene["lighting"]
        scene["lighting"] = lighting_map[old_lighting]
        changes.append(f"lighting: {old_lighting} -> {scene['lighting']}")
    
    # camera
    if scene.get("camera") and scene["camera"] in camera_map:
        old_camera = scene["camera"]
        scene["camera"] = camera_map[old_camera]
        changes.append(f"camera: {old_camera} -> {scene['camera']}")
    
    # action
    if scene.get("action") and scene["action"] in action_map:
        old_action = scene["action"]
        scene["action"] = action_map[old_action]
        changes.append(f"action: {old_action} -> {scene['action']}")
    
    # visual.motion
    visual = scene.get("visual", {}) or {}
    if isinstance(visual, dict) and visual.get("motion") and visual["motion"] in motion_map:
        old_motion = visual["motion"]
        visual["motion"] = motion_map[old_motion]
        changes.append(f"motion: {old_motion} -> {visual['motion']}")
        scene["visual"] = visual
    
    # face_style_auto
    face_style = scene.get("face_style_auto", {}) or {}
    if isinstance(face_style, dict):
        if face_style.get("expression") and face_style["expression"] in expression_map:
            old_expr = face_style["expression"]
            face_style["expression"] = expression_map[old_expr]
            changes.append(f"face_style_auto.expression: {old_expr} -> {face_style['expression']}")
        
        if face_style.get("lighting") and face_style["lighting"] in lighting_map:
            old_light = face_style["lighting"]
            face_style["lighting"] = lighting_map[old_light]
            changes.append(f"face_style_auto.lighting: {old_light} -> {face_style['lighting']}")
        
        if face_style.get("detail") and face_style["detail"] in detail_map:
            old_detail = face_style["detail"]
            face_style["detail"] = detail_map[old_detail]
            changes.append(f"face_style_auto.detail: {old_detail} -> {face_style['detail']}")
        
        scene["face_style_auto"] = face_style
    
    return changes

def optimize_narration(scene: Dict[str, Any]) -> List[str]:
    """优化 narration 长度"""
    changes = []
    narration = scene.get("narration", "")
    
    if narration and len(narration) > 30:
        new_narration = shorten_narration(narration, max_length=30)
        if new_narration != narration:
            scene["narration"] = new_narration
            changes.append(f"narration: {len(narration)}字 -> {len(new_narration)}字")
    
    return changes

def main():
    json_path = Path("lingjie/scenes/2.json")
    
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        return
    
    # 备份
    backup_path = json_path.with_suffix(".json.bak")
    shutil.copy(json_path, backup_path)
    print(f"✓ 已备份原文件: {backup_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    
    total_changes = 0
    for scene in scenes:
        scene_id = scene.get("id", "未知")
        changes = []
        
        # 转换元数据
        metadata_changes = convert_metadata_to_chinese(scene)
        changes.extend(metadata_changes)
        
        # 转换 visual 字段
        visual_changes = convert_visual_to_chinese(scene)
        changes.extend(visual_changes)
        
        # 优化 narration
        narration_changes = optimize_narration(scene)
        changes.extend(narration_changes)
        
        if changes:
            total_changes += 1
            print(f"\n场景 {scene_id}:")
            for change in changes:
                print(f"  ✓ {change}")
    
    # 保存
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ 转换完成！共处理 {total_changes}/{len(scenes)} 个场景")
    print(f"✓ 已保存到: {json_path}")

if __name__ == "__main__":
    main()

