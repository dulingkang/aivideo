#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复场景模板中剩余的英文内容
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any, List

# 英文到中文的详细映射
TRANSLATIONS = {
    # details 字段翻译
    "massive green palace, pillars with runes, vast empty plaza": "巨大的青色宫殿，带符文的石柱，广阔空旷的广场",
    "spiritual light waves, ethereal void, mysterious light mist": "灵光波纹，虚无空间，神秘光雾",
    "ancient mysterious, solemn and majestic, rich spiritual energy": "古老神秘，庄严肃穆，浓郁的灵气",
    "ringed formation of land turtles, shells linked as battlements, dunes and shifting sand": "陆行龟环形队形，龟壳连接的防御工事，翻滚的沙丘",
    "sun dimmed by dust, light diffused through sand curtain, shimmering heat": "被尘埃遮蔽的太阳，通过沙幕散射的光线，闪烁的热浪",
    "perilous, dust-choked, metallic blood scent, strained tension": "危险，尘烟弥漫，金属血腥味，紧张的氛围",
    "flat prairie, endless grass waves, occasional low knolls and dead trees": "平坦的草原，无边的草浪，偶尔出现的低丘和枯木",
    "multiple suns and moons, light filtered by dust veil, pale sky": "多轮日月，被尘幕过滤的光线，苍白的天空",
    "oppressive silence, hidden danger, low moaning wind": "压抑的寂静，隐藏的危险，低沉的风声",
    "thirty-zhang stone walls with runes, massive gates, straight stone streets": "三十丈高的带符文石墙，巨大的城门，笔直的石街",
    "clear yet harsh light, drifting dust veils": "清朗但冷硬的光线，飘动的尘雾",
    "military alertness, crowded refugees, looming beast tide pressure": "军事戒备，拥挤的难民，迫近的兽潮压力",
    "towering walls, broad parapets, rows of ballistae and catapults, battlefield extending beyond": "高耸的城墙，宽阔的女墙平台，成排的投石机和巨弩，延伸至战场的区域",
    "dust-laden sky, mingled battle smoke, streaks of spell light": "充满尘埃的天空，混合的战烟，灵光条纹",
    "tense vigilance, clashing armor, pervasive killing intent": "紧张的戒备，兵甲碰撞，弥漫的杀气",
    "straight cobbled streets, dense storefronts, weapon stalls and awnings": "笔直的石板街道，密集的店铺，武器摊和遮阳棚",
    "hazy sunlight filtered by city dust, multiple pale suns overhead": "被城市尘埃过滤的朦胧阳光，头顶多轮淡日",
    "crowded press, tense vigilance, vendors and armor clatter": "拥挤的人群，紧张的戒备，商贩和兵甲碰撞声",
    "layered courtyards, covered walkways, bamboo groves, guarded gates": "多重院落，有顶的回廊，竹林，守卫的大门",
    "soft daylight filtered by eaves, occasional lantern glow at night": "被屋檐过滤的柔和日光，夜晚偶尔的灯笼光芒",
    "prestigious clientele, patrolling guards, incense calm": "贵客云集，巡逻的守卫，香火宁静",
    "endless spiritual mist abyss, space behind sealed door, mist thick as sea": "无尽的灵雾深渊，封印门后的空间，如海般浓厚的灵雾",
    "spiritual light waves, ethereal void, mysterious light mist, spiritual pressure dense": "灵光波纹，虚无空间，神秘光雾，浓郁的灵压",
    "oppressive mysterious, unknown deep, rich spiritual energy, ancient inheritance": "压抑神秘，未知深邃，浓郁的灵气，古老的传承",
    "floating mountains in air, spiritual beasts as light, valleys with blooming spiritual flowers": "悬浮在空中的山峰，化光而行的灵兽，盛开灵花的山谷",
    "cloud sea churning, spiritual light flowing, dreamlike void": "翻涌的云海，流动的灵光，梦幻的虚空",
    "fantasy mysterious atmosphere, trial ground, rich spiritual energy, dreamlike yet real": "奇幻神秘的氛围，试炼之地，浓郁的灵气，梦幻而真实",
    "city floating in cloud sea, palaces suspended, spiritual bridges, bustling spirit market": "悬浮在云海的城池，悬空的宫阙，灵桥，繁华的灵市",
    "cloud sea churning, spiritual energy steaming, starry sky brilliant, ethereal sky": "翻涌的云海，蒸腾的灵气，璀璨的星空，虚无的天空",
    "majestic peaceful atmosphere, spirit tribe sacred land, rich spiritual energy, solemn mysterious": "宏伟安宁的氛围，灵族圣地，浓郁的灵气，庄严神秘",
    "broken True Immortal mansion, floating in sealed space, covered with True Immortal Realm runes": "残破的真仙遗府，悬浮在封印空间中，刻满真仙界符文",
    "sealed ancient space, True Immortal Realm aura, spiritual light flowing, ethereal immortal space": "封印的古老空间，真仙界气息，流动的灵光，虚无的仙域",
    "ancient solemn atmosphere, True Immortal legacy, mysterious deep, broken and weathered": "古老庄严的氛围，真仙遗韵，神秘深邃，残破沧桑",
    "huge underground chamber, spirit realm source floating in center, light chains connecting to void": "巨大的地下空间，灵界界源悬浮在中央，光链连接虚空",
    "underground chamber ceiling, spiritual light flowing, source light illuminating space, ethereal underground": "地下空间穹顶，流动的灵光，界源光芒照亮空间，虚无的地下",
    "solemn sacred atmosphere, realm source power, danger lurking, True Immortal threat": "庄严神圣的氛围，界源之力，潜伏的危机，真仙威胁",
    "country-sized capital, Confucian architecture, sacred mountain, palace complexes": "如小国般的巨城，儒门建筑群，圣山，连绵的宫殿",
    "clear sky, holy light, dense spiritual energy like mist": "清朗的天空，圣光，如雾般浓郁的灵气",
    "solemn sacred capital atmosphere, Confucian holy land, rich spiritual energy, prosperous peaceful": "庄严神圣的圣城氛围，儒门圣地，浓郁的灵气，繁华安宁",
    
    # prompt 字段中的英文关键词翻译
    "towering stone walls": "高耸的石墙",
    "dark iron gates": "暗色铁门",
    "subtle rune glow": "微妙的符文光芒",
    "no bright festive colors": "非明亮节日色彩",
    "colossal rune-covered walls": "巨大的符文覆盖的城墙",
    "massive black gates": "巨大的黑色城门",
    "straight cobbled avenues": "笔直的鹅卵石大道",
    "clear sky with harsh light": "清朗但冷硬的天空",
    "dust veils crossing sun": "穿过太阳的尘幕",
    "crowded refugee tension": "拥挤难民的紧张",
    "military readiness mood": "军事戒备的氛围",
    "looming threat vibe": "迫近威胁的氛围",
    "stone battlements with iron-gray tones": "铁灰色的石制防御工事",
    "dusty battlefield haze": "尘沙战场的雾霭",
    "no lush colors": "非茂盛色彩",
    "towering fortress wall with parapets": "带女墙的高耸要塞城墙",
    "rows of ballistae and catapults": "成排的投石机和巨弩",
    "battlefield beyond the wall": "城墙外的战场",
    "dusty sky with battle smoke": "充满战烟的尘沙天空",
    "spell light streaks": "灵光条纹",
    "tense battlefield atmosphere": "紧张的战场氛围",
    "clashing armor sounds": "兵甲碰撞声",
    "pervasive killing intent": "弥漫的杀气",
    "stone streets with muted colors": "灰暗色彩的石街",
    "dusty market tones": "尘沙市场的色调",
    "no neon colors": "非霓虹色彩",
    "straight cobbled street lined with shops": "两旁店铺的笔直石板街",
    "crowded weapon stalls": "拥挤的武器摊",
    "canvas awnings": "帆布遮阳棚",
    "hazy sun above city dust": "城市尘埃上方的朦胧太阳",
    "multiple pale suns": "多轮淡日",
    "crowded tense marketplace atmosphere": "拥挤紧张的市集氛围",
    "clanging weapons, shouted sales": "碰撞的武器，吆喝的叫卖",
    "dark wood courtyards with jade tiles": "带玉瓦的深木色院落",
    "muted teal walls": "灰暗的青绿色墙壁",
    "golden plaque glow": "金色匾额的光芒",
    "layered courtyards with covered walkways": "带顶回廊的多重院落",
    "bamboo shadows across stone": "石上的竹影",
    "guarded gate presence": "守卫大门的氛围",
    "soft courtyard light filtered by eaves": "被屋檐过滤的柔和院落光线",
    "tranquil yet guarded inn atmosphere": "宁静但戒备的客栈氛围",
    "incense haze": "香火雾霭",
    "deep blue abyss": "深蓝深渊",
    "cyan blue tones": "青蓝色调",
    "ethereal blue mist": "虚无的蓝色雾霭",
    "endless spiritual mist abyss": "无尽的灵雾深渊",
    "sealed space": "封印空间",
    "dense spiritual mist": "浓厚的灵雾",
    "ethereal void": "虚无空间",
    "dense spiritual pressure": "浓厚的灵压",
    "oppressive mysterious atmosphere": "压抑神秘的氛围",
    "unknown deep space": "未知深邃的空间",
    "ancient inheritance": "古老的传承",
    "green spirit mountain": "绿色灵山",
    "emerald green tones": "翠绿色调",
    "spiritual green glow": "灵光绿光",
    "floating mountains in air": "悬浮在空中的山峰",
    "spiritual beasts as light": "化光而行的灵兽",
    "valleys with spiritual flowers": "盛开灵花的山谷",
    "dreamlike landscape": "梦幻的景观",
    "cloud sea churning": "翻涌的云海",
    "spiritual light flowing": "流动的灵光",
    "dreamlike void": "梦幻的虚空",
    "ethereal sky": "虚无的天空",
    "cyan city": "青色城池",
    "light blue tones": "淡蓝色调",
    "silver white": "银白色",
    "floating city in cloud sea": "悬浮在云海的城池",
    "suspended palaces": "悬空的宫阙",
    "spiritual bridges": "灵桥",
    "bustling spirit market": "繁华的灵市",
    "spiritual energy steaming": "蒸腾的灵气",
    "starry sky brilliant": "璀璨的星空",
    "majestic peaceful atmosphere": "宏伟安宁的氛围",
    "spirit tribe sacred land": "灵族圣地",
    "solemn mysterious": "庄严神秘",
    "golden immortal mansion": "金色仙府",
    "pale gold tones": "淡金色调",
    "ancient bronze": "古铜色",
    "immortal golden glow": "仙光金芒",
    "no blue, no green": "非蓝色，非绿色",
    "broken True Immortal mansion": "残破的真仙遗府",
    "floating in sealed space": "悬浮在封印空间中",
    "True Immortal Realm runes": "真仙界符文",
    "ancient immortal architecture": "古老的仙府建筑",
    "sealed ancient space": "封印的古老空间",
    "True Immortal Realm aura": "真仙界气息",
    "ethereal immortal space": "虚无的仙域",
    "ancient solemn atmosphere": "古老庄严的氛围",
    "True Immortal legacy": "真仙遗韵",
    "mysterious deep": "神秘深邃",
    "broken and weathered": "残破沧桑",
    "deep blue chamber": "深蓝色空间",
    "dark blue tones": "深蓝色调",
    "realm source light": "界源光芒",
    "huge underground chamber": "巨大的地下空间",
    "spirit realm source floating": "悬浮的灵界界源",
    "light chains connecting": "连接的光链",
    "vast underground space": "广阔的地下空间",
    "underground chamber ceiling": "地下空间穹顶",
    "source light illuminating": "界源光芒照亮",
    "solemn sacred atmosphere": "庄严神圣的氛围",
    "realm source power": "界源之力",
    "danger lurking": "潜伏的危机",
    "True Immortal threat": "真仙威胁",
    "jade-white capital city": "玉白色圣城",
    "pale gold accents": "淡金色点缀",
    "no dark colors": "非暗色",
    "country-sized capital city": "如小国般的巨城",
    "Confucian-style palaces": "儒门风格的宫殿",
    "sacred mountain backdrop": "圣山背景",
    "endless palace complexes": "无尽的宫殿群",
    "clear sky with holy light": "带圣光的清朗天空",
    "dense spiritual energy mist": "如雾般浓郁的灵气",
    "radiant atmosphere": "光辉的氛围",
    
    # keywords 翻译
    "xianxia fortress city": "仙侠要塞城市",
    "beast tide defense": "兽潮防御",
    "crowded martial streets": "拥挤的武街",
    "xianxia fortress wall": "仙侠要塞城墙",
    "battle-ready parapets": "战备女墙",
    "siege defense scene": "围城防御场景",
    "xianxia frontier market": "仙侠边城市集",
    "weapon bazaar": "武器市场",
    "dusty fortress streets": "尘沙要塞街道",
    "xianxia noble inn": "仙侠贵族客栈",
    "layered courtyards": "多重院落",
    "guarded residence": "守卫的居所",
    "xianxia sealed space": "仙侠封印空间",
    "spiritual mist abyss": "灵雾深渊",
    "ancient inheritance realm": "古老传承领域",
    "mystical void space": "神秘虚无空间",
    "xianxia trial mountain": "仙侠试炼山",
    "floating spirit mountain": "悬浮灵山",
    "illusory spirit realm": "幻灵领域",
    "spiritual trial ground": "灵界试炼地",
    "xianxia spirit city": "仙侠灵城",
    "floating spirit realm city": "悬浮灵界城池",
    "ancient spirit capital": "古老灵都",
    "xianxia True Immortal mansion": "仙侠真仙遗府",
    "immortal realm ruins": "仙域废墟",
    "ancient immortal legacy": "古老仙府传承",
    "True Immortal Realm architecture": "真仙界建筑",
    "xianxia realm source": "仙侠界源",
    "spirit realm core": "灵界核心",
    "underground sacred chamber": "地下神圣空间",
    "realm power source": "界源力量",
    "xianxia capital city": "仙侠圣城",
    "Confucian holy capital": "儒门圣都",
    "sacred mountain city": "圣山城市",
    "spirit realm metropolis": "灵界大都会",
}

def translate_text(text: str) -> str:
    """翻译文本"""
    if not text or not isinstance(text, str):
        return text
    
    # 先尝试完整匹配
    if text in TRANSLATIONS:
        return TRANSLATIONS[text]
    
    # 尝试部分匹配和替换
    result = text
    for en, cn in TRANSLATIONS.items():
        if en in result:
            result = result.replace(en, cn)
    
    return result

def fix_scene_profile(scene_profile: Dict[str, Any]) -> List[str]:
    """修复单个场景模板，返回修改列表"""
    changes = []
    
    # 修复 details 字段
    for section in ["terrain", "sky", "atmosphere"]:
        if section in scene_profile and "details" in scene_profile[section]:
            old_details = scene_profile[section]["details"]
            if isinstance(old_details, str) and any(c.isalpha() and ord(c) < 128 for c in old_details):
                # 包含英文字母
                new_details = translate_text(old_details)
                if new_details != old_details:
                    scene_profile[section]["details"] = new_details
                    changes.append(f"{section}.details: {old_details[:50]}... -> {new_details[:50]}...")
    
    # 修复 prompt 字段中的英文
    for section in ["color_palette", "terrain", "sky", "atmosphere"]:
        if section in scene_profile and "prompt" in scene_profile[section]:
            old_prompt = scene_profile[section]["prompt"]
            if isinstance(old_prompt, str):
                # 检查是否包含英文单词（至少3个字母）
                if re.search(r'[a-zA-Z]{3,}', old_prompt):
                    new_prompt = translate_text(old_prompt)
                    if new_prompt != old_prompt:
                        scene_profile[section]["prompt"] = new_prompt
                        changes.append(f"{section}.prompt: 已修复英文内容")
    
    # 修复 keywords
    if "style" in scene_profile and "keywords" in scene_profile["style"]:
        keywords = scene_profile["style"]["keywords"]
        if isinstance(keywords, list):
            new_keywords = []
            for kw in keywords:
                if isinstance(kw, str):
                    # 检查是否包含英文
                    if re.search(r'[a-zA-Z]{3,}', kw):
                        new_kw = translate_text(kw)
                        new_keywords.append(new_kw)
                        if new_kw != kw:
                            changes.append(f"keywords: {kw} -> {new_kw}")
                    else:
                        new_keywords.append(kw)
                else:
                    new_keywords.append(kw)
            scene_profile["style"]["keywords"] = new_keywords
    
    # 修复 background_prompt
    if "background_prompt" in scene_profile:
        old_bg = scene_profile["background_prompt"]
        if isinstance(old_bg, str) and re.search(r'[a-zA-Z]{3,}', old_bg):
            new_bg = translate_text(old_bg)
            if new_bg != old_bg:
                scene_profile["background_prompt"] = new_bg
                changes.append("background_prompt: 已修复英文内容")
    
    return changes

def main():
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
    
    # 修复所有场景
    total_changes = 0
    for scene_key, scene_profile in data["scenes"].items():
        scene_name = scene_profile.get("scene_name", scene_key)
        changes = fix_scene_profile(scene_profile)
        
        if changes:
            print(f"\n场景: {scene_name} ({scene_key})")
            for change in changes:
                print(f"  ✓ {change}")
            total_changes += len(changes)
            data["scenes"][scene_key] = scene_profile
    
    # 保存文件
    if total_changes > 0:
        with open(scene_profile_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"\n✓ 共修复 {total_changes} 处，已保存到: {scene_profile_path}")
    else:
        print("\n✓ 未发现需要修复的内容")

if __name__ == "__main__":
    main()

