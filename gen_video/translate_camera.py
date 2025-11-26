#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译 camera 字段
"""

import json
import re
from pathlib import Path

# camera 映射表
camera_map = {
    "telephoto": "长焦",
    "telephoto_push": "长焦推进",
    "closeup_face": "面部特写",
    "wide_destruction": "破坏全景",
    "medium_air": "空中中景",
    "low_angle_follow": "低角度跟随",
    "wide_panorama": "全景",
    "over_shoulder": "过肩",
    "medium_side": "侧面中景",
    "wide_plaza": "广场全景",
    "close_up_arms": "手臂特写",
    "close_up_talismans": "符箓特写",
    "over_shoulder_combat": "过肩战斗",
    "top_down_macro": "俯视微距",
    "wide_magic_wave": "法术波全景",
    "close_up_face_side": "侧面面部特写",
    "low_angle_burst": "低角度爆发",
    "medium_front_on": "正面中景",
    "split_focus_action": "分焦动作",
    "slow_motion_impact": "慢动作冲击",
    "wide_return_tracking": "返回跟踪全景",
    "wide_montage": "蒙太奇全景",
    "medium_three_quarter": "四分之三中景",
    "close_up_profile": "侧面特写",
    "low_angle_citywall": "低角度城墙",
    "wide_magic_sweep": "法术扫过全景",
    "wide_symmetrical": "对称全景",
    "tracking_street_level": "街道跟踪",
    "wide_loading_area": "装载区全景",
    "three_quarter_view": "四分之三视角",
    "insert_map": "插入地图",
    "top_down_spread": "俯视展开",
    "abstract_inner": "抽象内部",
    "medium_front": "正面中景",
    "dynamic_side": "动态侧面",
    "wide_room": "房间全景",
    "close_over_shoulder": "过肩特写",
    "long_shot_alley": "长镜头小巷",
    "interior_reaction": "室内反应",
    "aerial_overview": "空中俯瞰",
    "wide_wall": "墙壁全景",
    "wide_sky": "天空全景",
    "aerial_follow": "空中跟随",
    "closeup_hands": "手部特写",
    "closeup_object": "物体特写",
}

def has_chinese(text: str) -> bool:
    """检查文本是否包含中文"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def translate_english_word(word: str) -> str:
    """翻译单个英文单词"""
    word_lower = word.lower().strip()
    
    word_map = {
        "close": "特写", "up": "向上", "medium": "中景", "wide": "全景",
        "long": "长焦", "shot": "镜头", "low": "低", "angle": "角度",
        "follow": "跟随", "over": "过", "shoulder": "肩", "side": "侧面",
        "front": "正面", "back": "背面", "top": "顶部", "down": "向下",
        "aerial": "空中", "tracking": "跟踪", "panorama": "全景",
        "telephoto": "长焦", "push": "推进", "face": "面部", "hands": "手部",
        "object": "物体", "profile": "侧面", "three": "三", "quarter": "四分之一",
        "view": "视角", "magic": "法术", "wave": "波", "sweep": "扫过",
        "burst": "爆发", "impact": "冲击", "motion": "动作", "slow": "慢",
        "split": "分", "focus": "焦", "action": "动作", "return": "返回",
        "montage": "蒙太奇", "citywall": "城墙", "symmetrical": "对称",
        "street": "街道", "level": "水平", "loading": "装载", "area": "区域",
        "insert": "插入", "map": "地图", "spread": "展开", "abstract": "抽象",
        "inner": "内部", "dynamic": "动态", "room": "房间", "alley": "小巷",
        "interior": "室内", "reaction": "反应", "overview": "俯瞰",
        "wall": "墙壁", "sky": "天空", "destruction": "破坏", "air": "空中",
        "combat": "战斗", "macro": "微距", "talismans": "符箓", "arms": "手臂",
        "plaza": "广场",
    }
    
    return word_map.get(word_lower, word)

def translate_camera(camera: str) -> str:
    """翻译 camera 字段"""
    if not camera:
        return camera
    
    if has_chinese(camera):
        return camera
    
    camera_lower = camera.lower().strip()
    if camera_lower in camera_map:
        return camera_map[camera_lower]
    
    # 尝试下划线分割翻译
    if "_" in camera:
        parts = camera.split("_")
        translated_parts = []
        for part in parts:
            translated = translate_english_word(part)
            if translated and translated != part:
                translated_parts.append(translated)
            else:
                translated_parts.append(part)
        if translated_parts:
            return "".join(translated_parts)
    
    return camera

def process_json_file(json_path: Path):
    """处理单个 JSON 文件"""
    print(f"处理: {json_path.name}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    updated_count = 0
    
    for scene in scenes:
        camera = scene.get("camera", "")
        if camera:
            new_camera = translate_camera(camera)
            if new_camera != camera:
                scene["camera"] = new_camera
                updated_count += 1
    
    if updated_count > 0:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  更新了 {updated_count} 个字段")
    else:
        print(f"  无需更新")

def main():
    """主函数"""
    base_dir = Path("lingjie/scenes")
    
    # 处理 3-11.json
    for i in range(3, 12):
        json_path = base_dir / f"{i}.json"
        if json_path.exists():
            process_json_file(json_path)
        else:
            print(f"文件不存在: {json_path}")

if __name__ == "__main__":
    main()

