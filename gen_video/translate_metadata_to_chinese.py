#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 JSON 场景文件中的元数据字段翻译为中文：
- mood: 情绪/氛围
- lighting: 光照
- camera: 镜头
- visual.motion: 镜头运动
- face_style_auto: 表情、光照、细节
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import shutil

# 翻译映射表
MOOD_TRANSLATIONS = {
    "serious": "严肃",
    "mysterious": "神秘",
    "tense": "紧张",
    "painful": "痛苦",
    "determined": "坚定",
    "calm": "平静",
    "contemplative": "沉思",
    "alert": "警觉",
    "dangerous": "危险",
    "focused": "专注",
    "intense": "激烈",
    "brutal": "残酷",
    "mysterious": "神秘",
}

LIGHTING_TRANSLATIONS = {
    "day": "白天",
    "dusk": "黄昏",
    "night": "夜晚",
    "night_transition": "夜晚过渡",
    "mystical_glow": "神秘光芒",
    "soft": "柔和",
    "bright_normal": "明亮正常",
    "dramatic": "戏剧性",
}

CAMERA_TRANSLATIONS = {
    "wide_shot_overhead": "俯视全景",
    "sky_pan": "天空平移",
    "close_up_face": "面部特写",
    "medium_shot": "中景",
    "close_up_energy": "能量特写",
    "wide_pan": "全景平移",
    "sky_timelapse": "天空延时",
    "extreme_close_up_eyes": "眼部极特写",
    "sky_to_ground": "天空到地面",
    "close_up_bird": "鸟类特写",
    "low_angle_birds": "低角度鸟类",
    "medium_shot_birds": "中景鸟类",
    "wide_shot": "全景",
    "low_angle_bird": "低角度鸟类",
    "wide_shot_riders": "全景骑士",
    "dynamic_shot": "动态镜头",
}

MOTION_TRANSLATIONS = {
    "slow overhead pan showing vast desert": "缓慢俯视平移展现广阔沙漠",
    "slow pan across sky showing seven celestial bodies": "缓慢平移天空展现七个天体",
    "slow push-in to face": "缓慢推向面部",
    "subtle shake to show pain": "微妙抖动表现痛苦",
    "energy flow visualization": "能量流动可视化",
    "slow pan across desert horizon": "缓慢平移沙漠地平线",
    "time-lapse effect of sky transition": "天空过渡延时效果",
    "extreme close-up of eyes": "眼部极特写",
    "camera following dots descending": "跟随下降黑点的镜头",
    "bird flying closer, revealing details": "鸟类飞近，展现细节",
    "low angle shot of birds diving": "低角度拍摄鸟类俯冲",
    "close-up of sand gathering process": "沙砾聚集过程特写",
    "tension building, birds getting closer": "紧张加剧，鸟类靠近",
    "dynamic shot of projectiles launching": "投射物发射动态镜头",
    "birds hit and falling": "鸟类被击中坠落",
    "wide shot of brutal scene": "残酷场景全景",
    "time-lapse of sky transformation": "天空变换延时",
    "close-up of alert expression": "警觉表情特写",
    "low angle shot of bird hovering": "低角度拍摄鸟类盘旋",
    "wide shot of riders approaching": "骑士接近全景",
}

EXPRESSION_TRANSLATIONS = {
    "calm": "平静",
    "focused": "专注",
    "alert": "警觉",
    "serious": "严肃",
    "determined": "坚定",
    "neutral": "中性",
    "tired_soft": "疲惫柔和",
}

DETAIL_TRANSLATIONS = {
    "natural": "自然",
    "detailed": "详细",
    "cinematic": "电影级",
    "sharp_dynamic": "锐利动态",
    "soft_concentrated": "柔和聚焦",
    "subtle": "微妙",
}

def translate_mood(mood: str) -> str:
    """翻译 mood 字段"""
    return MOOD_TRANSLATIONS.get(mood, mood)

def translate_lighting(lighting: str) -> str:
    """翻译 lighting 字段"""
    return LIGHTING_TRANSLATIONS.get(lighting, lighting)

def translate_camera(camera: str) -> str:
    """翻译 camera 字段"""
    return CAMERA_TRANSLATIONS.get(camera, camera)

def translate_motion(motion: str) -> str:
    """翻译 visual.motion 字段"""
    return MOTION_TRANSLATIONS.get(motion, motion)

def translate_expression(expression: str) -> str:
    """翻译 face_style_auto.expression 字段"""
    return EXPRESSION_TRANSLATIONS.get(expression, expression)

def translate_detail(detail: str) -> str:
    """翻译 face_style_auto.detail 字段"""
    return DETAIL_TRANSLATIONS.get(detail, detail)

def translate_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """翻译单个场景的所有元数据字段"""
    changes = []
    
    # 翻译 mood
    if "mood" in scene and scene["mood"]:
        old_mood = scene["mood"]
        new_mood = translate_mood(old_mood)
        if new_mood != old_mood:
            scene["mood"] = new_mood
            changes.append(f"mood: {old_mood} -> {new_mood}")
    
    # 翻译 lighting
    if "lighting" in scene and scene["lighting"]:
        old_lighting = scene["lighting"]
        new_lighting = translate_lighting(old_lighting)
        if new_lighting != old_lighting:
            scene["lighting"] = new_lighting
            changes.append(f"lighting: {old_lighting} -> {new_lighting}")
    
    # 翻译 camera
    if "camera" in scene and scene["camera"]:
        old_camera = scene["camera"]
        new_camera = translate_camera(old_camera)
        if new_camera != old_camera:
            scene["camera"] = new_camera
            changes.append(f"camera: {old_camera} -> {new_camera}")
    
    # 翻译 visual.motion
    visual = scene.get("visual", {}) or {}
    if isinstance(visual, dict) and "motion" in visual and visual["motion"]:
        old_motion = visual["motion"]
        new_motion = translate_motion(old_motion)
        if new_motion != old_motion:
            visual["motion"] = new_motion
            scene["visual"] = visual
            changes.append(f"visual.motion: {old_motion[:40]}... -> {new_motion[:40]}...")
    
    # 翻译 face_style_auto
    face_style = scene.get("face_style_auto", {}) or {}
    if isinstance(face_style, dict):
        # expression
        if "expression" in face_style and face_style["expression"]:
            old_expr = face_style["expression"]
            new_expr = translate_expression(old_expr)
            if new_expr != old_expr:
                face_style["expression"] = new_expr
                changes.append(f"face_style_auto.expression: {old_expr} -> {new_expr}")
        
        # lighting (face_style_auto 中的 lighting 和场景的 lighting 不同)
        if "lighting" in face_style and face_style["lighting"]:
            old_face_lighting = face_style["lighting"]
            new_face_lighting = translate_lighting(old_face_lighting)
            if new_face_lighting != old_face_lighting:
                face_style["lighting"] = new_face_lighting
                changes.append(f"face_style_auto.lighting: {old_face_lighting} -> {new_face_lighting}")
        
        # detail
        if "detail" in face_style and face_style["detail"]:
            old_detail = face_style["detail"]
            new_detail = translate_detail(old_detail)
            if new_detail != old_detail:
                face_style["detail"] = new_detail
                changes.append(f"face_style_auto.detail: {old_detail} -> {new_detail}")
        
        if changes or face_style:
            scene["face_style_auto"] = face_style
    
    return changes

def main():
    parser = argparse.ArgumentParser(description="将 JSON 场景文件中的元数据字段翻译为中文")
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
        backup_path = input_path.with_suffix('.json.bak3')
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)
            print(f"✓ 已备份原文件: {backup_path}")
    
    all_changes = []
    
    # 翻译 scenes
    if "scenes" in data and isinstance(data["scenes"], list):
        for scene in data["scenes"]:
            scene_id = scene.get("id", "未知")
            changes = translate_scene(scene)
            if changes:
                all_changes.append(f"\n场景 {scene_id}:")
                all_changes.extend([f"  ✓ {change}" for change in changes])
    
    # 显示修改摘要
    if all_changes:
        print("\n" + "=" * 60)
        print("元数据字段翻译摘要")
        print("=" * 60)
        for change in all_changes:
            print(change)
        
        # 保存文件
        if not args.dry_run:
            with open(input_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 已保存翻译后的文件: {input_path}")
        else:
            print(f"\n⚠ 预览模式，未实际修改文件")
    else:
        print("\n✓ 未发现需要翻译的内容")

if __name__ == "__main__":
    main()

