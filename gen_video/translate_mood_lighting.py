#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译 mood 和 lighting 字段
"""

import json
import re
from pathlib import Path

# mood 映射表
mood_map = {
    "solemn": "庄严",
    "resolute": "坚决",
    "decisive": "果断",
    "ominous": "不祥",
    "grateful": "感激",
    "urgent": "紧急",
    "shocking": "震惊",
    "heroic": "英勇",
    "foreboding": "不祥预兆",
    "scheming": "谋划",
    "uneasy": "不安",
    "respectful": "恭敬",
    "miraculous": "神奇",
    "focused": "专注",
    "menacing": "威胁",
    "fierce": "凶猛",
    "hopeful": "希望",
    "impact": "冲击",
    "concerned": "担忧",
    "somber": "阴沉",
    "informative": "信息性",
    "relieved": "宽慰",
    "awed": "敬畏",
    "formal": "正式",
    "transaction": "交易",
    "studious": "好学",
    "worried": "担心",
    "resolved": "坚定",
    "commanding": "威严",
    "grim": "严峻",
    "reunion": "团聚",
    "ready": "准备",
    "ferocious": "凶猛",
    "aggressive": "攻击性",
    "calm_under_pressure": "压力下冷静",
    "ritualistic": "仪式性",
    "wry": "苦笑",
    "peril": "危险",
    "victorious": "胜利",
    "suspicious": "怀疑",
    "gruesome": "可怕",
    "ruthless": "无情",
    "agile": "敏捷",
    "confrontational": "对抗性",
    "inquiring": "询问",
    "critical": "关键",
    "tense_standoff": "紧张对峙",
    "stalemate": "僵局",
    "catastrophic": "灾难性",
    "panic": "恐慌",
    "despair": "绝望",
    "resigned": "顺从",
    "devastated": "毁灭",
    "searching": "搜索",
    "gentle": "温和",
    "light": "轻松",
    "transit": "过渡",
    "stern": "严厉",
    "hinting": "暗示",
    "detached": "超然",
    "shocked": "震惊",
    "anxious": "焦虑",
    "triumphant": "胜利",
    "intrigued": "好奇",
    "practical": "实用",
    "friendly": "友好",
    "probing": "探索",
    "watchful": "警惕",
    "thoughtful": "深思",
    "orderly": "有序",
    "crowded": "拥挤",
    "procedural": "程序性",
    "cautious": "谨慎",
    "wistful": "渴望",
    "wary": "警惕",
    "reflective": "反思",
    "tempted": "诱惑",
    "trust": "信任",
    "tender": "温柔",
    "stealthy": "隐秘",
    "pursuit": "追逐",
    "betrayal": "背叛",
    "vigilant": "警觉",
    "unaware": "不知情",
    "impressive": "印象深刻",
    "alarmed": "警觉",
    "awe": "敬畏",
    "pensive": "沉思",
    "grand": "宏伟",
    "busy": "忙碌",
    "conversational": "对话",
    "observant": "观察",
    "impressed": "印象深刻",
    "swift": "迅速",
}

# lighting 映射表
lighting_map = {
    "soft_interior": "柔和室内",
    "cool_moon": "冷月",
    "bright": "明亮",
    "warm_soft": "温暖柔和",
    "dramatic_glow": "戏剧性光芒",
    "soft_glow": "柔和光芒",
    "dramatic_twilight": "戏剧性黄昏",
    "dramatic_dust": "戏剧性尘土",
    "warm_sunset": "温暖日落",
    "warm_lantern": "温暖灯笼",
    "warm_interior": "温暖室内",
    "warm_glow": "温暖光芒",
    "warm_torch": "温暖火把",
    "cool_night": "冷夜",
    "dramatic_flash": "戏剧性闪光",
    "warm_dusk": "温暖黄昏",
    "soft_sunset": "柔和日落",
    "warm_afternoon": "温暖下午",
    "soft_shadow": "柔和阴影",
    "candle_glow": "烛光",
    "cool_glow": "冷光",
    "moody": "情绪化",
    "internal_glow": "内部光芒",
    "soft_day": "柔和白天",
    "low": "低光",
    "soft_evening": "柔和夜晚",
    "soft_dawn": "柔和黎明",
    "cool_day": "冷日",
    "soft_dim": "柔和昏暗",
    "soft_warm": "柔和温暖",
    "cool_dawn": "冷黎明",
    "soft_bright": "柔和明亮",
    "cool_twilight": "冷黄昏",
}

def has_chinese(text: str) -> bool:
    """检查文本是否包含中文"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def translate_mood(mood: str) -> str:
    """翻译 mood 字段"""
    if not mood:
        return mood
    
    if has_chinese(mood):
        return mood
    
    mood_lower = mood.lower().strip()
    if mood_lower in mood_map:
        return mood_map[mood_lower]
    
    # 尝试下划线分割翻译
    if "_" in mood:
        parts = mood.split("_")
        translated_parts = []
        for part in parts:
            part_lower = part.lower()
            if part_lower in mood_map:
                translated_parts.append(mood_map[part_lower])
            else:
                # 简单单词翻译
                word_map = {
                    "calm": "冷静", "under": "在", "pressure": "压力",
                    "tense": "紧张", "standoff": "对峙",
                }
                if part_lower in word_map:
                    translated_parts.append(word_map[part_lower])
                else:
                    translated_parts.append(part)
        if translated_parts:
            return "".join(translated_parts)
    
    return mood

def translate_lighting(lighting: str) -> str:
    """翻译 lighting 字段"""
    if not lighting:
        return lighting
    
    if has_chinese(lighting):
        return lighting
    
    lighting_lower = lighting.lower().strip()
    if lighting_lower in lighting_map:
        return lighting_map[lighting_lower]
    
    # 尝试下划线分割翻译
    if "_" in lighting:
        parts = lighting.split("_")
        translated_parts = []
        for part in parts:
            part_lower = part.lower()
            if part_lower in lighting_map:
                translated_parts.append(lighting_map[part_lower])
            else:
                # 简单单词翻译
                word_map = {
                    "soft": "柔和", "interior": "室内", "cool": "冷",
                    "moon": "月", "warm": "温暖", "dramatic": "戏剧性",
                    "glow": "光芒", "twilight": "黄昏", "dust": "尘土",
                    "sunset": "日落", "lantern": "灯笼", "torch": "火把",
                    "night": "夜", "flash": "闪光", "dusk": "黄昏",
                    "afternoon": "下午", "shadow": "阴影", "candle": "蜡烛",
                    "moody": "情绪化", "internal": "内部", "day": "白天",
                    "low": "低", "evening": "夜晚", "dawn": "黎明",
                    "dim": "昏暗", "bright": "明亮",
                }
                if part_lower in word_map:
                    translated_parts.append(word_map[part_lower])
                else:
                    translated_parts.append(part)
        if translated_parts:
            return "".join(translated_parts)
    
    return lighting

def process_json_file(json_path: Path):
    """处理单个 JSON 文件"""
    print(f"处理: {json_path.name}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    updated_count = 0
    
    for scene in scenes:
        # 翻译 mood
        mood = scene.get("mood", "")
        if mood:
            new_mood = translate_mood(mood)
            if new_mood != mood:
                scene["mood"] = new_mood
                updated_count += 1
        
        # 翻译 lighting
        lighting = scene.get("lighting", "")
        if lighting:
            new_lighting = translate_lighting(lighting)
            if new_lighting != lighting:
                scene["lighting"] = new_lighting
                updated_count += 1
        
        # 翻译 face_style_auto.lighting
        face_style = scene.get("face_style_auto", {}) or {}
        if isinstance(face_style, dict):
            face_lighting = face_style.get("lighting", "")
            if face_lighting:
                new_face_lighting = translate_lighting(face_lighting)
                if new_face_lighting != face_lighting:
                    face_style["lighting"] = new_face_lighting
                    scene["face_style_auto"] = face_style
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

