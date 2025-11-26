#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译 JSON 场景文件中的字段（除了 description 和 narration）
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# 翻译映射表
TRANSLATIONS = {
    # 标题
    "陌生之地·初入灵界": "Strange Land · Entering the Spirit Realm",
    "血咒文书·天东商号": "Blood Curse Document · Tiandong Trading Company",
    
    # 情绪 (mood)
    "平静": "calm",
    "严肃": "serious",
    "神秘": "mysterious",
    "紧张": "tense",
    "痛苦": "painful",
    "坚定": "determined",
    "沉思": "contemplative",
    "警觉": "alert",
    "危险": "dangerous",
    "专注": "focused",
    "激烈": "intense",
    "残酷": "brutal",
    "宏伟": "grand",
    "惊讶": "surprised",
    "困惑": "confused",
    "好奇": "curious",
    "领悟": "understanding",
    "分析": "analyzing",
    
    # 光照 (lighting)
    "柔和": "soft",
    "白天": "daylight",
    "神秘光芒": "mysterious glow",
    "黄昏": "dusk",
    "夜晚过渡": "night transition",
    "夜晚": "night",
    "明亮正常": "bright normal",
    "戏剧性": "dramatic",
    
    # 动作 (action)
    "标题展示": "title display",
    "静止躺着": "lying still",
    "观察天空": "observing sky",
    "回忆": "recalling",
    "承受痛苦": "enduring pain",
    "使用法术": "casting spell",
    "转头": "turning head",
    "观察日落": "observing sunset",
    "聚焦眼神": "focusing gaze",
    "察觉威胁": "detecting threat",
    "识别敌人": "identifying enemy",
    "准备防御": "preparing defense",
    "聚集沙砾": "gathering sand",
    "蓄力攻击": "charging attack",
    "发动攻击": "launching attack",
    "击中目标": "hitting target",
    "同类相食": "cannibalism",
    "观察变化": "observing change",
    "听到声音": "hearing sound",
    "观察鸟类": "observing bird",
    "观察骑士": "observing riders",
    "观察巨龟": "observing giant turtle",
    "结束画面展示": "ending display",
    "用圆盘探测": "detecting with disk",
    "承受探测": "enduring detection",
    "对结果反应": "reacting to result",
    "尝试沟通": "attempting communication",
    "识别语言": "identifying language",
    "取出木盒": "taking out wooden box",
    "激活圆珠": "activating bead",
    "接收圆珠": "receiving bead",
    "学习语言": "learning language",
    "确认位置": "confirming location",
    "同意契约": "agreeing to contract",
    "书写契约": "writing contract",
    "分析契约": "analyzing contract",
    "尝试切割": "attempting to cut",
    "确认修炼": "confirming cultivation",
    "use灵具": "using spirit tool",
    "发现奥秘": "discovering secret",
    "书写血咒符文": "writing blood curse runes",
    "契约激活": "contract activation",
    
    # 镜头 (camera)
    "固定镜头": "fixed shot",
    "俯视全景": "aerial panoramic view",
    "天空平移": "sky pan",
    "面部特写": "close-up face",
    "中景": "medium shot",
    "能量特写": "energy close-up",
    "全景平移": "panoramic pan",
    "天空延时": "sky time-lapse",
    "眼部极特写": "extreme eye close-up",
    "天空到地面": "sky to ground",
    "鸟类特写": "bird close-up",
    "低角度鸟类": "low angle bird",
    "close_up_mouth": "close-up mouth",
    "动态镜头": "dynamic shot",
    "中景鸟类": "medium shot bird",
    "全景": "panoramic",
    "面部特写": "face close-up",
    "全景骑士": "panoramic riders",
    "圆盘特写": "disk close-up",
    "双人medium shot": "two-person medium shot",
    "face close-up": "face close-up",
    "木盒特写": "wooden box close-up",
    "激活特写": "activation close-up",
    "body特写": "body close-up",
    "头部特写": "head close-up",
    "手部特写": "hand close-up",
    "手腕特写": "wrist close-up",
    "文书特写": "document close-up",
    "血液特写": "blood close-up",
    "契约特写": "contract close-up",
    "剑特写": "sword close-up",
    "微距": "macro",
    "陆行龟panoramic": "land turtle panoramic",
    
    # 表情 (expression)
    "警觉": "alert",
    "专注": "focused",
    "平静": "calm",
    "惊讶": "surprised",
    
    # 细节 (detail)
    "自然": "natural",
    
    # 通用词汇
    "韩立": "Han Li",
    "元婴虚化之法": "Nascent Soul Void method",
    "元婴虚化": "Nascent Soul Void",
    "虚化": " void",
    "仙域": "immortal realm",
    "天空": "sky",
    "太阳": "sun",
    "月亮": "moon",
    "虚影": "phantom",
    "出现": "appearing",
    "云雾": "mist",
    "缭绕": "swirling",
    "卷轴": "scroll",
    "灵气": "spiritual energy",
    "粒子": "particles",
    "飘浮": "floating",
    "开场": "opening",
    "结尾": "ending",
    "沙地": "sand ground",
    "沙砾": "sand",
    "青灰色": "grayish",
    "一望无际": "endless",
    "广阔": "vast",
    "沙漠": "desert",
    "高空": "high altitude",
    "远处": "distance",
    "上空": "above",
    "地面": "ground",
    "震动": "shaking",
    "骑士": "riders",
    "怪兽": "monster",
    "怪鸟": "strange bird",
    "鹰首蝠身": "eagle head bat body",
    "黑色": "black",
    "赤红色": "crimson red",
    "火红": "fiery red",
    "小鸟": "small bird",
    "盘旋": "circling",
    "扑到": "diving to",
    "目标": "target",
    "头颅": "head",
    "偏动": "tilt",
    "胸脯": "chest",
    "瘪": "collapse",
    "吸起": "suck up",
    "一团": "a ball of",
    "嘴边": "mouth",
    "滴溜溜": "spinning",
    "转动": "rotating",
    "十余丈": "over ten zhang",
    "三十余丈": "over thirty zhang",
    "胸膛": "chest",
    "一鼓": "expand",
    "喷出": "spit out",
    "强风": "strong wind",
    "化为": "transform into",
    "青芒": "green light",
    "激射": "shoot out",
    "击中": "hit",
    "发出": "emit",
    "金属摩擦": "metal friction",
    "惨叫": "scream",
    "剩下": "remaining",
    "扑向": "pounce on",
    "受伤": "injured",
    "同类": "same species",
    "撕裂": "tear apart",
    "分尸": "dismember",
    "最后": "last",
    "一个": "one",
    "黯淡": "dim",
    "开始": "begin",
    "变幻": "transform",
    "形态": "form",
    "即将": "about to",
    "出现": "appear",
    "七个": "seven",
    "弯月": "crescent moon",
    "神色": "expression",
    "一变": "change",
    "听到": "hear",
    "传来": "come from",
    "轰隆隆": "rumbling",
    "声音": "sound",
    "画面": "scene",
    "淡入": "fade in",
    "展现": "show",
    "金色": "golden",
    "光芒": "light",
    "闪动": "flickering",
    "背景音乐": "background music",
    "渐起": "gradually rise",
    "躺在": "lying on",
    "一动不动": "motionless",
    "感受": "feeling",
    "燥热": "heat",
    "三个": "three",
    "夺目": "dazzling",
    "四个": "four",
    "朦胧": "hazy",
    "回忆": "recall",
    "空间节点": "spatial node",
    "遭遇": "encounter",
    "脸色": "face",
    "难看": "ugly",
    "冰凤": "Ice Phoenix",
    "分开": "separate",
    "禁制": "restriction",
    "发作": "activate",
    "使用": "use",
    "元婴虚化": "Nascent Soul Void",
    "之法": "method",
    "将": "infuse",
    "精元": "essence",
    "灌注": "infuse",
    "身体": "body",
    "费劲": "with effort",
    "地": "ly",
    "看到": "see",
    "变成": "become",
    "两个": "two",
    "最后": "finally",
    "开始": "begin",
    "睁大": "open wide",
    "双目": "eyes",
    "瞳孔": "pupil",
    "中": "in",
    "隐隐": "faintly",
    "有": "have",
    "蓝芒": "blue light",
    "闪动": "flickering",
    "注视": "gaze at",
    "空中": "air",
    "传来": "come from",
    "凄厉": "piercing",
    "尖鸣": "screech",
    "二十多个": "over twenty",
    "黑点": "black dots",
    "向下": "downward",
    "急坠": "plummet",
    "看清": "see clearly",
    "是": "is",
    "狰狞": "ferocious",
    "异常": "extremely",
    "淡出": "fade out",
    "夜色": "night",
    "下": "under",
    "仙城": "immortal city",
    "剪影": "silhouette",
    "洒落": "scatter",
    "灵光": "spiritual light",
    "逐渐": "gradually",
    "变暗": "darken",
    "全景镜头": "panoramic shot",
    "电影级": "cinematic",
    "光影": "lighting",
    "4k": "4k",
    "氛围": "atmosphere",
    "平静": "calm",
    "神秘": "mysterious",
    "缓慢": "slow",
    "镜头": "camera",
    "移动": "move",
    "展现": "show",
    "平移": "pan",
    "展现": "show",
    "七个": "seven",
    "天体": "celestial bodies",
    "推向": "push to",
    "微妙": "subtle",
    "抖动": "shake",
    "表现": "express",
    "能量": "energy",
    "流动": "flow",
    "可视化": "visualization",
    "地平线": "horizon",
    "过渡": "transition",
    "延时": "time-lapse",
    "效果": "effect",
    "变换": "transform",
    "跟随": "follow",
    "下降": "descending",
    "飞近": "fly close",
    "展现": "show",
    "细节": "details",
    "拍摄": "shoot",
    "俯冲": "dive",
    "过程": "process",
    "加剧": "intensify",
    "靠近": "approach",
    "投射物": "projectile",
    "发射": "launch",
    "被": "be",
    "坠落": "fall",
    "场景": "scene",
    "表情": "expression",
    "接近": "approach",
    "逐渐": "gradually",
    
    # 2.json 特有词汇
    "青年": "young man",
    "圆盘": "disk",
    "对准": "aim at",
    "青濛濛光柱": "bluish light beam",
    "激射而出": "shoot out",
    "光柱照在": "light beam shines on",
    "任凭": "allow",
    "探测": "detection",
    "满脸惊容": "full of surprise",
    "冲身后大叫": "shout to companions behind",
    "疤面大汉": "scarred big man",
    "走过来": "walk over",
    "开口问": "ask",
    "听不懂": "cannot understand",
    "神色一动": "expression changes",
    "觉得耳熟": "feels familiar",
    "淡黄色": "pale yellow",
    "木盒": "wooden box",
    "乳黄色圆珠": "milky yellow bead",
    "指环晶石": "ring crystal",
    "光芒大放": "light radiates",
    "没入": "enter into",
    "凉气窜入脑中": "cool air rushes into brain",
    "头颅刺痛": "head stings",
    "神识": "divine sense",
    "浮现": "appear",
    "瞬间掌握": "instantly master",
    "青罗沙漠": "Qingluo Desert",
    "灵界": "Spirit Realm",
    "签订": "sign",
    "血咒文书": "blood curse document",
    "答应": "agree",
    "拿出血咒文书": "take out blood curse document",
    "咬破手指": "bite finger",
    "书写": "write",
    "查看": "examine",
    "认出": "recognize",
    "符箓": "talisman",
    "约束力": "binding force",
    "用匕首划": "cut with dagger",
    "手腕": "wrist",
    "无法划破": "cannot cut",
    "询问": "ask",
    "是否": "whether",
    "修炼": "cultivate",
    "金刚诀": "Vajra Art",
    "第三层": "third layer",
    "拿出": "take out",
    "金莹剑": "Golden Ying Sword",
    "戴上手套": "put on gloves",
    "剑身": "sword blade",
    "金芒四射": "golden light radiates",
    "发现": "discover",
    "奥秘": "secret",
    "灵具": "spirit tool",
    "剑柄": "sword hilt",
    "灵石": "spirit stone",
    "手套": "gloves",
    "划破": "cut",
    "鲜血流出": "blood flows out",
    "血咒符文": "blood curse runes",
    "血光大放": "blood light radiates",
    "符文飞射而出": "runes shoot out",
    "文书自燃": "document self-ignites",
    "灰尘大起": "dust rising",
    "巨大的": "giant",
    "陆行龟": "land turtle",
    "appear": "appear",
    "车厢门": "carriage door",
    "打开": "open",
    "distance": "distance",
    "approach": "approach",
    
    # 常用连接词和助词
    "的": " ",
    "一只": "a ",
    "一只巨大的": "a giant ",
    "身上": " on body",
    "大放": " radiates",
    "和": " and ",
    "光束": "light beam",
    "反应": "reaction",
    "特写": "close-up",
    "沟通": "communication",
    "尝试": "attempt",
    "换了一种语言": "changed language",
    "一动": " changes",
    "瞬间": "instant",
    "掏出": "take out",
    "里面": "inside",
    "一颗": "a ",
    "圆珠": "bead",
    "按在": "press on",
    "上": " on",
    "窜入": "rush into",
    "脑": "brain",
    "进入": "enter",
    "刺痛": "stings",
    "浮现": "appear",
    "众多": "many",
    "东西": "things",
    "语言": "language",
    "与": "with",
    "对话": "dialogue",
    "确认": "confirm",
    "这里": "here",
    "提出": "propose",
    "签订": "sign",
    "过程": "process",
    "文书": "document",
    "和": "and",
    "符文": "runes",
    "其": "its",
    "否": "whether",
    "修炼": "cultivate",
    "第三层": "third layer",
    "激活": "activate",
    "发现": "discover",
    "奥秘": "secret",
    "灵具": "spirit tool",
    "剑柄": "sword hilt",
    "有": "have",
    "灵石": "spirit stone",
    "也": "also",
    "血液": "blood",
    "夜空": "night sky",
    "飘落": "fall",
    "效果": "effect",
    "大汉": "big man",
    "换": "change",
    "了": "",
    "一种": "a kind of",
    "觉得": "feel",
    "耳熟": "familiar",
    "掏出": "take out",
    "淡黄色": "pale yellow",
    "木盒": "wooden box",
    "里面": "inside",
    "是": "is",
    "乳黄色": "milky yellow",
    "将": "infuse",
    "按在": "press on",
    "指环": "ring",
    "晶石": "crystal",
    "光芒": "light",
    "没入": "enter into",
    "凉气": "cool air",
    "头颅": "head",
    "神识": "divine sense",
    "中": "in",
    "瞬间": "instantly",
    "掌握": "master",
    "青罗": "Qingluo",
    "沙漠": "desert",
    "提出": "propose",
    "签订": "sign",
    "血咒": "blood curse",
    "答应": "agree",
    "拿出": "take out",
    "咬破": "bite",
    "手指": "finger",
    "查看": "examine",
    "认出": "recognize",
    "符箓": "talisman",
    "分析": "analyze",
    "约束力": "binding force",
    "用": "use",
    "匕首": "dagger",
    "划": "cut",
    "但": "but",
    "无法": "cannot",
    "划破": "cut",
    "询问": "ask",
    "是否": "whether",
    "确认": "confirm",
    "拿出": "take out",
    "戴上": "put on",
    "剑身": "sword blade",
    "金芒": "golden light",
    "四射": "radiates",
    "发现": "discover",
    "奥秘": "secret",
    "也": "also",
    "划破": "cut",
    "鲜血": "blood",
    "流出": "flow out",
    "书写": "write",
    "血光": "blood light",
    "飞射而出": "shoot out",
    "自燃": "self-ignites",
    "灰尘": "dust",
    "大起": "rising",
    "巨大的": "giant",
    "车厢": "carriage",
    "门": "door",
    "打开": "open",
    "仙侠": "xianxia",
    "ending": "ending",
    "under": "under",
    "scatter": "scatter",
    "atmosphere": "atmosphere",
    "展开": "unfurl",
    "一道": "a beam of ",
    "而出": " out",
    "识别": "identify",
    "学习": "learn",
    "谈判": "negotiate",
    "切割": "cut",
    "mistswirling": "mist swirling",
    "immortal realmabove": "immortal realm above",
    "goldenscroll": "golden scroll",
    "goldenlight": "golden light",
    "scroll": "scroll",
    "xianxia": "xianxia",
    "opening": "opening",
    "immortal realmsky": "immortal realm sky",
    "spiritual energyparticles": "spiritual energy particles",
    "floating": "floating",
    "panoramic shot": "panoramic shot",
    "cinematiclighting": "cinematic lighting",
    "young maninfuse": "young man infuse",
    "diskaim at": "disk aim at",
    "bluish light beamshoot out": "bluish light beam shoot out",
    "distancedust": "distance dust",
    "rising": "rising",
    "giantland turtleappear": "giant land turtle appear",
    "carriage dooropen": "carriage door open",
    "剑": "sword",
}


def translate_text(text: str) -> str:
    """翻译文本，保留原有结构"""
    if not text or not isinstance(text, str):
        return text
    
    # 如果已经是英文（不包含中文字符），直接返回
    if not any('\u4e00' <= char <= '\u9fff' for char in text):
        return text
    
    result = text
    # 按长度排序，先替换长的短语
    sorted_translations = sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    for chinese, english in sorted_translations:
        if chinese in result:
            # 如果英文翻译以空格开头，确保替换后格式正确
            if english.startswith(' '):
                result = result.replace(chinese, english)
            else:
                result = result.replace(chinese, english)
    
    # 清理多余的空格和标点，并在英文单词之间添加空格
    import re
    # 在英文单词和中文之间添加空格（如果还没有）
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)  # 小写后跟大写
    result = re.sub(r'([a-zA-Z])([a-z])', r'\1 \2', result)  # 英文后跟小写（可能是新词）
    # 清理连续的空格
    result = re.sub(r'\s+', ' ', result)
    # 清理标点符号前后的空格
    result = re.sub(r'\s+([，。、])', r'\1', result)
    result = re.sub(r'([，。、])\s+', r'\1 ', result)
    # 修复常见的单词连接问题
    result = result.replace('distancedust', 'distance dust')
    result = result.replace('giantland', 'giant land')
    result = result.replace('turtleappear', 'turtle appear')
    result = result.replace('dooropen', 'door open')
    result = result.replace('mistswirling', 'mist swirling')
    result = result.replace('realmabove', 'realm above')
    result = result.replace('goldenscroll', 'golden scroll')
    result = result.replace('goldenlight', 'golden light')
    result = result.replace('realm sky', 'realm sky')
    result = result.replace('energyparticles', 'energy particles')
    result = result.replace('maninfuse', 'man infuse')
    result = result.replace('diskaim', 'disk aim')
    result = result.replace('beamshoot', 'beam shoot')
    result = result.replace('light beamshoot', 'light beam shoot')
    
    return result.strip()


def translate_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """翻译场景对象中除 description 和 narration 外的所有字段"""
    translated = scene.copy()
    
    # 翻译顶层字段
    if "mood" in translated:
        translated["mood"] = translate_text(translated["mood"])
    if "lighting" in translated:
        translated["lighting"] = translate_text(translated["lighting"])
    if "action" in translated:
        translated["action"] = translate_text(translated["action"])
    if "camera" in translated:
        translated["camera"] = translate_text(translated["camera"])
    if "prompt" in translated:
        translated["prompt"] = translate_text(translated["prompt"])
    
    # 翻译 visual 对象
    if "visual" in translated and isinstance(translated["visual"], dict):
        visual = translated["visual"].copy()
        for key in ["composition", "environment", "character_pose", "fx", "motion"]:
            if key in visual and visual[key]:
                visual[key] = translate_text(visual[key])
        translated["visual"] = visual
    
    # 翻译 face_style_auto 对象
    if "face_style_auto" in translated and isinstance(translated["face_style_auto"], dict):
        face_style = translated["face_style_auto"].copy()
        for key in ["expression", "lighting", "detail"]:
            if key in face_style and face_style[key]:
                face_style[key] = translate_text(face_style[key])
        translated["face_style_auto"] = face_style
    
    return translated


def translate_json_file(input_path: Path, output_path: Path = None):
    """翻译 JSON 文件"""
    if output_path is None:
        output_path = input_path
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 翻译标题
    if "title" in data:
        data["title"] = translate_text(data["title"])
    
    # 翻译所有场景
    if "scenes" in data and isinstance(data["scenes"], list):
        data["scenes"] = [translate_scene(scene) for scene in data["scenes"]]
    
    # 保存翻译后的文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 翻译完成: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python translate_json_fields.py <json_file> [output_file]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file
    
    if not input_file.exists():
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    translate_json_file(input_file, output_file)

