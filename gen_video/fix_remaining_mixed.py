#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复剩余的中英混合字段和 expression 字段
"""

import json
import re
from pathlib import Path

# expression 映射表
expression_map = {
    "curious": "好奇",
    "confused": "困惑",
    "horrified": "惊恐",
    "shocked": "震惊",
    "aghast": "惊骇",
    "cold": "冷漠",
    "ruthless": "无情",
    "menacing": "威胁",
    "straining": "紧张",
    "fierce": "凶猛",
    "neutral": "中性",
    "serene": "宁静",
    "concerned": "担忧",
    "pensive": "沉思",
    "joyful": "快乐",
    "thoughtful": "深思",
    "resolved": "坚定",
    "grateful": "感激",
    "suspicious": "怀疑",
    "ominous": "不祥",
    "grim": "严峻",
    "resolute": "坚决",
    "terrified": "恐惧",
    "sorrowful": "悲伤",
    "grief": "悲痛",
    "soft": "柔和",
    "shy": "害羞",
    "relieved": "宽慰",
    "solemn": "庄严",
    "commanding": "威严",
    "respectful": "恭敬",
    "knowing": "了解",
    "exhausted": "疲惫",
    "pained": "痛苦",
    "alarmed": "警觉",
    "worried": "担心",
    "concentrated": "专注",
    "stern": "严厉",
    "absorbed": "专注",
    "polite": "礼貌",
    "unimpressed": "不以为然",
    "composed": "镇定",
    "attentive": "专注",
    "puzzled": "困惑",
    "anxious": "焦虑",
    "thinking": "思考",
    "awe": "敬畏",
    "appraising": "评估",
    "satisfied": "满意",
    "feral": "野性",
    "aroused": "兴奋",
    "anguished": "痛苦",
    "snarl": "咆哮",
    "strained": "紧张",
    "wry": "苦笑",
    "tired": "疲惫",
    "impressed": "印象深刻",
    "observant": "观察",
}

def has_chinese(text: str) -> bool:
    """检查文本是否包含中文"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def translate_expression(expr: str) -> str:
    """翻译 expression 字段"""
    if not expr:
        return expr
    
    # 检查是否已包含中文
    if has_chinese(expr):
        return expr
    
    # 检查映射表
    expr_lower = expr.lower().strip()
    if expr_lower in expression_map:
        return expression_map[expr_lower]
    
    return expr

def split_compound_word(word: str) -> list:
    """分割复合英文单词"""
    word_lower = word.lower()
    
    # 常见单词列表（按长度降序排列，优先匹配长单词）
    common_words = [
        "synchronized", "transformation", "acupuncture", "talisman", "movement",
        "battleline", "following", "isolating", "emphasizing", "highlighting",
        "panoramic", "alongside", "overhead", "tracking", "shooting",
        "arrangement", "process", "reactions", "walking", "leaving",
        "cagedbeast", "needleshooting", "sittingup", "armmovement",
        "intodustcloud", "witharrows", "followingblades", "fromchaos",
        "intocaptureresolve", "withglowinglinesspreading", "emphasizinginsight",
        "followingemergence", "withsparks", "highlightingrelief",
        "followingridersintocamp", "followingleadingcart", "acrossfaces",
        "aroundtable", "astheydepart",
        # 基础单词
        "needle", "sitting", "arm", "caged", "beast", "side", "along",
        "into", "dust", "cloud", "with", "arrows", "blades", "him",
        "from", "chaos", "capture", "resolve", "glowing", "lines",
        "spreading", "insight", "emergence", "sparks", "stone", "arc",
        "relief", "riders", "camp", "leading", "cart", "faces", "wall",
        "token", "activity", "table", "aerial", "depart", "spear",
        "shot", "lens", "frame", "close", "medium", "wide", "long",
        "push", "pull", "pan", "tilt", "track", "follow", "chase",
        "swing", "drift", "cross", "reveal", "move", "motion", "impact",
        "collapse", "rise", "shutter", "freeze", "up", "down", "forward",
        "backward", "upward", "lateral", "across", "around", "aside",
        "slow", "fast", "quick", "rapid", "gentle", "slight", "dramatic",
    ]
    
    # 尝试匹配
    words = []
    i = 0
    while i < len(word_lower):
        matched = False
        for w in sorted(common_words, key=len, reverse=True):
            if word_lower[i:].startswith(w):
                words.append(w)
                i += len(w)
                matched = True
                break
        if not matched:
            words.append(word_lower[i])
            i += 1
    
    return words

def translate_english_word_in_motion(word: str) -> str:
    """翻译 motion 中的英文单词"""
    word_lower = word.lower().strip()
    
    # 完整单词映射（优先）
    full_translations = {
        "needleshooting": "针射",
        "sittingup": "坐起",
        "acupunctureprocess": "针灸过程",
        "armmovement": "手臂移动",
        "talismanprocess": "符箓过程",
        "cagedbeast": "笼中妖兽",
        "alongbattleline": "沿着战线",
        "synchronizedwithtransformation": "与变形同步",
        "intodustcloud": "进入尘土云",
        "witharrows": "与箭",
        "followingblades": "跟随刀刃",
        "isolatinghimfromchaos": "从混乱中隔离他",
        "intocaptureresolve": "进入捕捉解决",
        "withglowinglinesspreading": "与发光线条扩散",
        "emphasizinginsight": "强调洞察",
        "followingemergence": "跟随出现",
        "withsparks": "与火花",
        "highlightingrelief": "突出宽慰",
        "followingridersintocamp": "跟随骑手进入营地",
        "followingleadingcart": "跟随领先马车",
        "acrossfaces": "横跨面孔",
        "upwall": "向上墙",
        "tokenarc": "令牌弧",
        "acrossactivity": "横跨活动",
        "aroundtable": "围绕桌子",
        "astheydepart": "当他们离开",
        "followingblades": "跟随刀刃",
        "fromchaos": "从混乱",
        "withsparks": "与火花",
        "lowangle": "低角度",
        "pullback": "拉回",
    }
    
    if word_lower in full_translations:
        return full_translations[word_lower]
    
    # 单个单词映射
    word_translations = {
        "shot": "镜头", "lens": "镜头", "frame": "帧",
        "close": "特写", "medium": "中景", "wide": "全景", "long": "长焦",
        "push": "推进", "pull": "拉回", "pan": "平移", "tilt": "倾斜",
        "track": "跟踪", "follow": "跟随", "chase": "追逐", "swing": "摆动", "sway": "摇摆",
        "drift": "漂移", "arc": "弧形", "cross": "交叉", "reveal": "揭示",
        "move": "移动", "motion": "动作", "impact": "冲击", "collapse": "倒塌",
        "rise": "上升", "shutter": "快门", "freeze": "冻结", "panoramic": "全景",
        "alongside": "旁边", "small": "小", "low": "低", "angle": "角度",
        "pull": "拉回", "back": "向后", "top": "顶部", "down": "向下",
        "dynamic": "动态", "handheld": "手持",
        "up": "向上", "down": "向下", "forward": "向前", "backward": "向后",
        "upward": "向上", "lateral": "横向", "along": "沿着", "across": "横跨",
        "into": "进入", "around": "围绕", "aside": "旁边",
        "slow": "缓慢", "fast": "快速", "quick": "快速", "rapid": "快速",
        "gentle": "轻柔", "slight": "轻微", "dramatic": "戏剧性",
        "needle": "针", "shooting": "射击", "sitting": "坐",
        "arrangement": "安排", "acupuncture": "针灸", "process": "过程",
        "arm": "手臂", "movement": "移动", "reactions": "反应", "walking": "行走",
        "talisman": "符箓", "stop": "停止", "leaving": "离开", "caravan": "车队",
        "caged": "笼中", "beast": "妖兽", "side": "侧面", "battleline": "战线",
        "synchronized": "同步", "with": "与", "transformation": "变形",
        "dust": "尘土", "cloud": "云", "arrows": "箭", "blades": "刀刃",
        "isolating": "隔离", "him": "他", "from": "从", "chaos": "混乱",
        "capture": "捕捉", "resolve": "解决", "glowing": "发光", "lines": "线条",
        "spreading": "扩散", "emphasizing": "强调", "insight": "洞察",
        "emergence": "出现", "sparks": "火花", "stone": "石头",
        "highlighting": "突出", "relief": "宽慰", "riders": "骑手", "camp": "营地",
        "leading": "领先", "cart": "马车", "faces": "面孔", "wall": "墙",
        "token": "令牌", "activity": "活动", "table": "桌子", "overhead": "上方",
        "aerial": "空中", "depart": "离开", "spear": "长矛",
    }
    
    if word_lower in word_translations:
        return word_translations[word_lower]
    
    # 尝试分割复合单词
    split_words = split_compound_word(word)
    if len(split_words) > 1:
        translated_parts = []
        for w in split_words:
            if w in word_translations:
                translated_parts.append(word_translations[w])
            else:
                translated_parts.append(w)
        return "".join(translated_parts)
    
    return word

def fix_mixed_motion(motion: str) -> str:
    """修复中英混合的 motion 字段"""
    if not motion:
        return motion
    
    # 如果已经是纯中文，直接返回
    if not re.search(r'[a-zA-Z]', motion):
        return motion
    
    # 如果已经是纯英文，返回原值（应该已经被翻译了）
    if not has_chinese(motion):
        return motion
    
    # 处理中英混合的情况
    # 先处理连字符的情况，如 "推进-in" -> "推进"
    motion = re.sub(r'([\u4e00-\u9fff]+)-([a-z]+)', r'\1', motion)
    
    # 使用正则表达式分割中英文（包括连字符）
    parts = re.split(r'([a-zA-Z]+|-)', motion)
    
    result = ""
    i = 0
    while i < len(parts):
        part = parts[i]
        if not part:
            i += 1
            continue
        
        if re.match(r'^[a-zA-Z]+$', part):
            # 这是英文部分，尝试翻译
            translated = translate_english_word_in_motion(part)
            result += translated
        elif part == '-' and i + 1 < len(parts):
            # 处理连字符，如果后面是英文，尝试翻译
            next_part = parts[i + 1]
            if re.match(r'^[a-zA-Z]+$', next_part):
                # 跳过连字符，直接翻译下一个单词
                translated = translate_english_word_in_motion(next_part)
                result += translated
                i += 1  # 跳过下一个部分
            else:
                # 保留连字符
                result += part
        else:
            # 这是中文或其他字符，直接保留
            result += part
        
        i += 1
    
    # 清理多余的空格和连字符
    result = re.sub(r'\s+', '', result)
    result = re.sub(r'-+', '', result)
    
    return result

def process_json_file(json_path: Path):
    """处理单个 JSON 文件"""
    print(f"处理: {json_path.name}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    updated_count = 0
    
    for scene in scenes:
        # 修复 face_style_auto.expression
        face_style = scene.get("face_style_auto", {}) or {}
        if isinstance(face_style, dict):
            expression = face_style.get("expression", "")
            if expression:
                new_expression = translate_expression(expression)
                if new_expression != expression:
                    face_style["expression"] = new_expression
                    scene["face_style_auto"] = face_style
                    updated_count += 1
        
        # 修复 visual.motion 中的中英混合
        visual = scene.get("visual", {}) or {}
        if isinstance(visual, dict):
            motion = visual.get("motion", "")
            if motion:
                new_motion = fix_mixed_motion(motion)
                if new_motion != motion:
                    visual["motion"] = new_motion
                    scene["visual"] = visual
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

