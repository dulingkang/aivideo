#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将场景模板和角色模板中的英文 prompt、details、keywords 转换成中文
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any

def translate_prompt_to_chinese(prompt: str) -> str:
    """将英文 prompt 转换成中文 prompt"""
    if not prompt:
        return prompt
    
    # 英文到中文的映射（针对 prompt 中的关键词）
    translations = {
        # 颜色相关
        "golden sand": "金黄色沙子",
        "warm orange tones": "暖橙色调",
        "beige and tan hues": "米色和褐色调",
        "no white sand, no red sand": "非白沙，非红沙",
        
        # 地形相关
        "rolling sand waves": "翻滚的沙浪",
        "vast golden desert": "广阔的金色沙漠",
        "distant stone peaks": "远方的石峰",
        "drifting sand": "飘散的沙子",
        
        # 天空相关
        "intense sunlight": "强烈的阳光",
        "heat haze": "热浪扭曲",
        "clear bright sky": "晴朗明亮的天空",
        "golden sunset rays": "金色的夕阳光线",
        
        # 氛围相关
        "arid atmosphere": "干旱的氛围",
        "vast empty landscape": "广阔空旷的景观",
        "subtle spiritual energy": "微妙的灵气",
        "mysterious qi fluctuations": "神秘的灵气波动",
        
        # 建筑相关
        "green palace": "青色宫殿",
        "cyan and jade tones": "青绿色调",
        "spiritual blue glow": "灵光蓝光",
        "ancient green stone": "古老的青石",
        "no red, no yellow": "非红色，非黄色",
        "massive green palace": "巨大的青色宫殿",
        "ancient stone pillars with glowing runes": "带有发光符文的古老石柱",
        "vast empty plaza": "广阔空旷的广场",
        "towering architecture": "高耸的建筑",
        
        # 灵界相关
        "spiritual light waves": "灵光波纹",
        "ethereal void space": "虚无的空间",
        "mysterious light mist": "神秘的光雾",
        "flowing spiritual energy": "流动的灵气",
        "ancient mysterious atmosphere": "古老神秘的氛围",
        "solemn and majestic": "庄严肃穆",
        "rich spiritual energy": "浓郁的灵气",
        "otherworldly palace": "异界宫殿",
        
        # 风格相关
        "xianxia cultivation world": "仙侠修炼世界",
        "ancient Chinese fantasy": "中国古代玄幻",
        "cinematic xianxia style": "电影级仙侠风格",
        "xianxia spirit realm palace": "仙侠灵界宫殿",
        "ancient Chinese fantasy architecture": "中国古代玄幻建筑",
        "mystical spiritual palace": "神秘灵宫",
        "cultivation world temple": "修炼世界寺庙",
        "cinematic sense of scale": "电影级规模感",
        "xianxia desert frontier": "仙侠沙漠边境",
        
        # 战场相关
        "dusty grey-green desert": "灰青色尘沙沙漠",
        "sand yellow haze": "沙黄色雾霭",
        "dark brown turtle shells": "深棕色龟壳",
        "no lush green": "非茂盛绿色",
        "battlefield blood accents": "战场的血色点缀",
        "ring of massive turtle wagons": "巨大的陆行龟环形车队",
        "linked shell battlements": "连接的龟壳防御工事",
        "rolling dunes around battlefield": "战场周围翻滚的沙丘",
        "scarred desert ground": "伤痕累累的沙漠地面",
        "sunlight filtered through sandstorm": "穿过沙暴的阳光",
        "dust curtain diffused light": "尘幕散射的光线",
        "heat shimmer sky": "热浪闪烁的天空",
        "no clear blue sky": "非晴朗蓝天",
        "perilous battlefield atmosphere": "危险的战场氛围",
        "dust-choked air": "尘烟弥漫的空气",
        "metallic blood scent mood": "金属血腥味的氛围",
        "high tension warzone": "高度紧张的战争区域",
        "xianxia desert battlefield": "仙侠沙漠战场",
        "turtle caravan defense circle": "龟车防御圈",
        "sand worm siege": "沙虫围攻",
        "dust-choked warzone": "尘烟弥漫的战争区域",
        
        # 草原相关
        "muted grey-green prairie": "灰绿色草原",
        "pale blue sky": "淡蓝色天空",
        "dusty brown tracks": "尘土飞扬的棕色小径",
        "no lush saturated colors": "非茂盛的饱和色彩",
        "endless flat prairie": "无边平坦的草原",
        "rolling grass waves": "翻滚的草浪",
        "scattered low knolls": "散布的低丘",
        "occasional deadwood silhouettes": "偶尔出现的枯木剪影",
        "multiple suns in pale sky": "淡色天空中的多个太阳",
        "dust-filtered sunlight": "尘沙过滤的阳光",
        "soft white haze": "柔和的白色雾霭",
        "oppressive silent prairie": "压抑寂静的草原",
        "hidden danger mood": "隐藏危险的氛围",
        "low wind hum": "低沉的风声",
        "xianxia prairie march": "仙侠草原行军",
        "cautious caravan field": "谨慎的商队草原",
        "muted spirit grassland": "灰暗的灵草原",
    }
    
    # 直接翻译整个 prompt
    result = prompt
    for en, cn in translations.items():
        result = result.replace(en, cn)
    
    # 处理括号和权重格式，保持格式不变
    # 例如: (golden sand:1.4) -> (金黄色沙子:1.4)
    
    return result

def convert_scene_profile(scene_profile: Dict[str, Any]) -> Dict[str, Any]:
    """转换单个场景模板"""
    # 转换 color_palette.prompt
    if "color_palette" in scene_profile and "prompt" in scene_profile["color_palette"]:
        scene_profile["color_palette"]["prompt"] = translate_prompt_to_chinese(
            scene_profile["color_palette"]["prompt"]
        )
    
    # 转换 terrain.prompt 和 terrain.details
    if "terrain" in scene_profile:
        if "prompt" in scene_profile["terrain"]:
            scene_profile["terrain"]["prompt"] = translate_prompt_to_chinese(
                scene_profile["terrain"]["prompt"]
            )
        if "details" in scene_profile["terrain"]:
            # details 通常较短，直接用中文描述替换
            pass  # 保持原有的中文 description
    
    # 转换 sky.prompt 和 sky.details
    if "sky" in scene_profile:
        if "prompt" in scene_profile["sky"]:
            scene_profile["sky"]["prompt"] = translate_prompt_to_chinese(
                scene_profile["sky"]["prompt"]
            )
        if "details" in scene_profile["sky"]:
            pass  # 保持原有的中文 description
    
    # 转换 atmosphere.prompt 和 atmosphere.details
    if "atmosphere" in scene_profile:
        if "prompt" in scene_profile["atmosphere"]:
            scene_profile["atmosphere"]["prompt"] = translate_prompt_to_chinese(
                scene_profile["atmosphere"]["prompt"]
            )
        if "details" in scene_profile["atmosphere"]:
            pass  # 保持原有的中文 description
    
    # 转换 style.keywords
    if "style" in scene_profile and "keywords" in scene_profile["style"]:
        keywords = scene_profile["style"]["keywords"]
        if isinstance(keywords, list):
            scene_profile["style"]["keywords"] = [
                translate_prompt_to_chinese(kw) if isinstance(kw, str) else kw
                for kw in keywords
            ]
    
    # 转换 background_prompt
    if "background_prompt" in scene_profile:
        scene_profile["background_prompt"] = translate_prompt_to_chinese(
            scene_profile["background_prompt"]
        )
    
    return scene_profile

def main():
    # 处理场景模板
    scene_profile_path = Path(__file__).parent / "scene_profiles.yaml"
    
    if not scene_profile_path.exists():
        print(f"⚠ 场景模板文件不存在: {scene_profile_path}")
        return
    
    # 读取文件
    with open(scene_profile_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if "scenes" not in data:
        print("⚠ 未找到 scenes 字段")
        return
    
    # 备份原文件
    backup_path = scene_profile_path.with_suffix('.yaml.bak')
    if not backup_path.exists():
        import shutil
        shutil.copy2(scene_profile_path, backup_path)
        print(f"✓ 已备份原文件: {backup_path}")
    
    # 转换所有场景
    converted_count = 0
    for scene_key, scene_profile in data["scenes"].items():
        scene_name = scene_profile.get("scene_name", scene_key)
        print(f"\n转换场景: {scene_name} ({scene_key})")
        
        converted = convert_scene_profile(scene_profile)
        data["scenes"][scene_key] = converted
        converted_count += 1
    
    # 保存文件
    with open(scene_profile_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ 已转换 {converted_count} 个场景模板")
    print(f"✓ 已保存到: {scene_profile_path}")

if __name__ == "__main__":
    main()

