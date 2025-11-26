#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将 JSON 场景文件转换为中文：
1. 转换元数据字段（mood, lighting, camera, action, motion, face_style_auto）
2. 智能提取 visual 字段（composition, environment, character_pose, fx）
3. 优化 narration 长度
"""

import json
import re
import shutil
import sys
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
    "serious": "严肃", "analytical": "分析", "majestic": "宏伟", "mystical": "神秘",
    "tense": "紧张", "painful": "痛苦", "determined": "坚定", "contemplative": "沉思",
    "brutal": "残酷", "dramatic": "戏剧性", "peaceful": "平静", "excited": "兴奋"
}

lighting_map = {
    "night": "夜晚", "day": "白天", "dusk": "黄昏", "sunset": "日落",
    "bright_normal": "明亮正常", "soft": "柔和", "dramatic": "戏剧性",
    "mystical_glow": "神秘光芒", "night_transition": "夜晚过渡",
    "afternoon": "下午", "alley_day": "小巷白天", "alley_tinted": "小巷色调",
    "book_glow": "书本光芒", "booklight": "书本光", "bright_day": "明亮白天",
    "contrast_glow": "对比光芒", "cool_tone": "冷色调", "courtyard_day": "庭院白天",
    "courtyard_evening": "庭院傍晚", "dawn": "黎明", "daylight": "日光",
    "daylight_dusty": "尘土日光", "daylight_glow": "日光光芒", "dust_burst": "尘土爆发",
    "dust_filtered": "过滤尘土", "dust_flash": "尘土闪光", "dusty_crimson": "尘土深红",
    "eerie_black_flame": "诡异黑焰", "forge_glow": "锻造光芒", "glare_backlit": "逆光眩光",
    "glow": "光芒", "harsh_daylight": "刺眼日光", "inner_glow": "内部光芒",
    "interior_bright": "室内明亮", "interior_day": "室内白天", "interior_dim": "室内昏暗",
    "interior_shadow": "室内阴影", "interior_soft": "室内柔和", "interior_warm": "室内温暖",
    "lampglow": "灯光", "lamplight": "灯光", "late_afternoon": "傍晚",
    "moonlight": "月光", "moonlit_evening": "月夜", "predawn": "黎明前",
    "radiant": "辐射", "street_day": "街道白天", "tavern_dim": "酒馆昏暗",
    "tavern_warm": "酒馆温暖", "torchlight": "火把光", "twilight": "暮色",
    "walkway_day": "走道白天"
}

camera_map = {
    "close_up_disk": "圆盘特写", "medium_shot": "中景", "close_up_face": "面部特写",
    "medium_shot_two": "双人中景", "close_up_box": "木盒特写", "close_up_activation": "激活特写",
    "close_up_body": "身体特写", "close_up_head": "头部特写", "close_up_hand": "手部特写",
    "close_up_paper": "文书特写", "close_up_wrist": "手腕特写", "close_up_sword": "剑特写",
    "macro_shot": "微距", "close_up_blood": "血液特写", "close_up_contract": "契约特写",
    "wide_shot_turtle": "陆行龟全景", "wide_shot": "全景", "wide_shot_overhead": "俯视全景",
    "sky_pan": "天空平移", "close_up_energy": "能量特写", "wide_pan": "全景平移",
    "sky_timelapse": "天空延时", "extreme_close_up_eyes": "眼部极特写",
    "sky_to_ground": "天空到地面", "close_up_bird": "鸟类特写", "low_angle_birds": "低角度鸟类",
    "close_up_mouth": "嘴部特写", "medium_shot_birds": "中景鸟类", "low_angle_bird": "低角度鸟类",
    "wide_shot_riders": "全景骑士", "dynamic_shot": "动态镜头", "close_up_head": "头部特写"
}

action_map = {
    "detect_with_disk": "用圆盘探测", "endure_detection": "承受探测", "react_to_result": "对结果反应",
    "try_communicate": "尝试沟通", "recognize_language": "识别语言", "take_out_box": "取出木盒",
    "activate_pearl": "激活圆珠", "receive_pearl": "接收圆珠", "learn_language": "学习语言",
    "confirm_location": "确认位置", "agree_to_contract": "同意契约", "write_contract": "书写契约",
    "analyze_contract": "分析契约", "try_cut": "尝试切割", "confirm_cultivation": "确认修炼",
    "use_spiritual_tool": "使用灵具", "discover_secret": "发现奥秘", "write_blood_rune": "书写血咒符文",
    "contract_activate": "契约激活", "observe_giant_turtle": "观察巨龟", "lying_still": "静止躺着",
    "observe_sky": "观察天空", "recall_memory": "回忆", "suffer_pain": "承受痛苦",
    "use_technique": "使用法术", "turn_head": "转头", "observe_sunset": "观察日落",
    "focus_eyes": "聚焦眼神", "detect_threat": "察觉威胁", "identify_enemy": "识别敌人",
    "prepare_defense": "准备防御", "gather_sand": "聚集沙砾", "hold_attack": "蓄力攻击",
    "launch_attack": "发动攻击", "hit_target": "击中目标", "cannibalism": "同类相食",
    "observe_transition": "观察变化", "hear_sound": "听到声音", "observe_bird": "观察鸟类",
    "observe_riders": "观察骑士"
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
    "wide shot of approaching turtle": "接近的陆行龟全景",
    "slow overhead pan showing vast desert": "缓慢俯视平移展现广阔沙漠",
    "slow pan across sky showing seven celestial bodies": "缓慢平移天空展现七个天体",
    "slow push-in to face": "缓慢推向面部",
    "subtle shake to show pain": "微妙抖动表现痛苦",
    "energy flow visualization": "能量流动可视化",
    "slow pan across desert horizon": "缓慢平移沙漠地平线",
    "time-lapse effect of sky transition": "天空过渡延时效果",
    "extreme close-up of eyes": "眼部极特写",
    "camera following dots descending": "镜头跟随黑点下降",
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
    "wide shot of riders approaching": "骑士接近全景"
}

expression_map = {
    "focused": "专注", "calm": "平静", "surprised": "惊讶", "alert": "警觉",
    "serious": "严肃", "determined": "坚定"
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
        r"空中[^，。！？]*", r"地面[^，。！？]*", r"车厢[^，。！？]*", r"龟背[^，。！？]*",
        r"沙地[^，。！？]*", r"沙砾[^，。！？]*", r"天空[^，。！？]*", r"太阳[^，。！？]*",
        r"月亮[^，。！？]*", r"高空[^，。！？]*", r"上空[^，。！？]*", r"宫殿[^，。！？]*",
        r"城市[^，。！？]*", r"街道[^，。！？]*", r"房间[^，。！？]*", r"室内[^，。！？]*"
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
        r"查看[^，。！？]*", r"发现[^，。！？]*", r"刺痛[^，。！？]*", r"平静[^，。！？]*",
        r"躺在[^，。！？]*", r"一动不动[^，。！？]*", r"回忆[^，。！？]*", r"睁大[^，。！？]*",
        r"注视[^，。！？]*", r"偏动[^，。！？]*", r"胸膛[^，。！？]*", r"神色[^，。！？]*",
        r"看清[^，。！？]*", r"看到[^，。！？]*", r"听到[^，。！？]*", r"使用[^，。！？]*"
    ]
    
    extracted_parts = []
    for pattern in pose_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            phrase = match.strip()
            # 过滤掉包含环境信息的短语
            if any(env_kw in phrase for env_kw in ["沙漠", "灵界", "远处", "车厢", "龟背", "空中", "地面", "沙地", "沙砾", "天空", "太阳", "月亮"]):
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
        r"浮现[^，。！？]*", r"没入[^，。！？]*", r"划破[^，。！？]*", r"流出[^，。！？]*",
        r"震动[^，。！？]*", r"声音[^，。！？]*", r"尖鸣[^，。！？]*", r"惨叫[^，。！？]*",
        r"闪动[^，。！？]*", r"流转[^，。！？]*", r"波动[^，。！？]*", r"能量[^，。！？]*",
        r"灵气[^，。！？]*", r"烟雾[^，。！？]*", r"火焰[^，。！？]*", r"冰霜[^，。！？]*",
        r"雷电[^，。！？]*", r"沙尘[^，。！？]*", r"粒子[^，。！？]*", r"扭曲[^，。！？]*",
        r"虚影[^，。！？]*", r"幻影[^，。！？]*", r"炸裂[^，。！？]*", r"轰鸣[^，。！？]*"
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

def translate_camera_simple(camera: str) -> str:
    """简单翻译camera名称（对于未映射的值）"""
    if has_chinese(camera):
        return camera
    
    # 如果包含下划线，说明是复合词，需要整体翻译
    if '_' in camera:
        # 先检查是否已经有部分翻译（包含中文字符）
        if any(ord(c) > 127 for c in camera):
            # 已经是混合翻译，返回原值让用户手动处理
            return camera
        
        # 下划线分隔的复合词翻译
        parts = camera.split('_')
        translations = []
        
        word_map = {
            'close': '特写', 'closeup': '特写', 'medium': '中景', 'wide': '全景',
            'long': '远景', 'low': '低角度', 'high': '高角度', 'aerial': '航拍',
            'overhead': '俯视', 'tracking': '跟踪', 'dynamic': '动态', 'slow': '慢',
            'motion': '运动', 'push': '推进', 'pull': '拉', 'pan': '平移',
            'follow': '跟随', 'reveal': '揭示', 'establishing': '建立', 'action': '动作',
            'interior': '室内', 'exterior': '室外', 'street': '街道', 'courtyard': '庭院',
            'moonlit': '月光', 'door': '门', 'face': '面部', 'hands': '手', 'arms': '手臂',
            'profile': '侧面', 'ring': '戒指', 'talismans': '符箓', 'text': '文字',
            'shot': '镜头', 'angle': '角度', 'two': '双人', 'three': '三人',
            'group': '群体', 'shop': '商店', 'table': '桌子', 'walk': '行走',
            'up': '', 'of': '', 'the': '', 'a': '', 'an': '', 'and': '', 'or': '',
            'circular': '环形', 'pullback': '拉回', 'overview': '概览', 'burst': '爆发',
            'citywall': '城墙', 'inside': '内部', 'carriage': '车厢', 'reaction': '反应',
            'macro': '微距', 'spread': '展开', 'level': '水平', 'forward': '向前',
            'shoulder': '肩膀', 'combat': '战斗', 'impact': '冲击', 'side': '侧面',
            'focus': '焦点', 'split': '分割', 'telephoto': '长焦', 'quarter': '四分之一',
            'montage': '蒙太奇', 'panorama': '全景', 'plaza': '广场', 'room': '房间',
            'sky': '天空', 'stair': '楼梯', 'symmetrical': '对称', 'wall': '墙',
            'alley': '小巷', 'loading': '装载', 'magic': '魔法', 'sweep': '扫过',
            'wave': '波浪', 'return': '返回', 'crowd': '人群', 'destruction': '破坏',
            'table': '桌子', 'establishing': '建立', 'insert': '插入', 'map': '地图',
            'abstract': '抽象', 'inner': '内部'
        }
        
        for part in parts:
            if part.lower() in word_map:
                trans = word_map[part.lower()]
                if trans:  # 跳过空翻译
                    translations.append(trans)
            else:
                # 如果不在映射表中，保留原词（可能是专有名词）
                translations.append(part)
        
        # 组合翻译结果
        if translations:
            return ''.join(translations) if len(translations) == 1 else ' '.join(translations)
    
    # 如果没有下划线，尝试直接替换
    word_map = {
        'close': '特写', 'medium': '中景', 'wide': '全景', 'low': '低角度',
        'aerial': '航拍', 'tracking': '跟踪', 'dynamic': '动态'
    }
    
    for eng, chi in word_map.items():
        if camera.lower() == eng:
            return chi
    
    # 如果无法翻译，返回原值
    return camera

def translate_action_simple(action: str) -> str:
    """简单翻译action名称（对于未映射的值）"""
    if has_chinese(action):
        return action
    
    # 如果包含下划线，说明是复合词，需要整体翻译
    if '_' in action:
        # 先检查是否已经有部分翻译（包含中文字符）
        if any(ord(c) > 127 for c in action):
            # 已经是混合翻译，返回原值让用户手动处理
            return action
        
        # 下划线分隔的复合词翻译
        parts = action.split('_')
        translations = []
        
        word_map = {
            'enter': '进入', 'exit': '离开', 'observe': '观察', 'hear': '听到',
            'see': '看到', 'open': '打开', 'close': '关闭', 'walk': '行走',
            'run': '跑', 'sit': '坐', 'stand': '站', 'lie': '躺',
            'talk': '说话', 'ask': '问', 'answer': '回答', 'think': '思考',
            'use': '使用', 'take': '拿', 'give': '给', 'put': '放',
            'kill': '杀死', 'attack': '攻击', 'defend': '防御', 'escape': '逃跑',
            'fight': '战斗', 'battle': '战斗', 'meet': '见面', 'leave': '离开',
            'arrive': '到达', 'depart': '出发', 'return': '返回', 'go': '去',
            'come': '来', 'look': '看', 'watch': '观看', 'listen': '听',
            'visitors': '访客', 'visitor': '访客', 'door': '门', 'eyes': '眼睛',
            'eye': '眼睛', 'body': '身体', 'purpose': '目的', 'explain': '解释',
            'examine': '检查', 'diagnose': '诊断', 'arrange': '安排', 'cushion': '垫子',
            'acupuncture': '针灸', 'move': '移动', 'arms': '手臂', 'impressed': '印象深刻',
            'respectful': '尊重', 'miraculous': '神奇'
        }
        
        for part in parts:
            if part.lower() in word_map:
                translations.append(word_map[part.lower()])
            else:
                # 如果不在映射表中，尝试保留原词
                translations.append(part)
        
        # 组合翻译结果
        if translations:
            return ''.join(translations) if len(translations) == 1 else ''.join(translations)
    
    # 如果没有下划线，尝试直接替换
    word_map = {
        'observe': '观察', 'hear': '听到', 'see': '看到', 'open': '打开'
    }
    
    for eng, chi in word_map.items():
        if action.lower() == eng:
            return chi
    
    # 如果无法翻译，返回原值
    return action

def convert_metadata_to_chinese(scene: Dict[str, Any]) -> List[str]:
    """转换元数据字段为中文"""
    changes = []
    
    # mood
    if scene.get("mood") and not has_chinese(scene["mood"]):
        old_mood = scene["mood"]
        if old_mood in mood_map:
            scene["mood"] = mood_map[old_mood]
            changes.append(f"mood: {old_mood} -> {scene['mood']}")
        else:
            # 如果不在映射表中，尝试保留原值（可能是中文）
            pass
    
    # lighting
    if scene.get("lighting") and not has_chinese(scene["lighting"]):
        old_lighting = scene["lighting"]
        if old_lighting in lighting_map:
            scene["lighting"] = lighting_map[old_lighting]
            changes.append(f"lighting: {old_lighting} -> {scene['lighting']}")
        else:
            # 如果不在映射表中，尝试简单翻译
            new_lighting = translate_action_simple(old_lighting)  # 复用简单翻译
            if new_lighting != old_lighting:
                scene["lighting"] = new_lighting
                changes.append(f"lighting: {old_lighting} -> {scene['lighting']}")
    
    # camera
    if scene.get("camera") and not has_chinese(scene["camera"]):
        old_camera = scene["camera"]
        if old_camera in camera_map:
            scene["camera"] = camera_map[old_camera]
            changes.append(f"camera: {old_camera} -> {scene['camera']}")
        else:
            # 如果不在映射表中，使用简单翻译
            new_camera = translate_camera_simple(old_camera)
            if new_camera != old_camera:
                scene["camera"] = new_camera
                changes.append(f"camera: {old_camera} -> {scene['camera']}")
    
    # action
    if scene.get("action") and not has_chinese(scene["action"]):
        old_action = scene["action"]
        if old_action in action_map:
            scene["action"] = action_map[old_action]
            changes.append(f"action: {old_action} -> {scene['action']}")
        else:
            # 如果不在映射表中，使用简单翻译
            new_action = translate_action_simple(old_action)
            if new_action != old_action:
                scene["action"] = new_action
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

def process_json_file(json_path: Path, verbose: bool = True) -> tuple:
    """处理单个JSON文件"""
    if not json_path.exists():
        if verbose:
            print(f"❌ 文件不存在: {json_path}")
        return False, 0
    
    # 备份
    backup_path = json_path.with_suffix(".json.bak")
    shutil.copy(json_path, backup_path)
    if verbose:
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
            if verbose:
                print(f"\n场景 {scene_id}:")
                for change in changes:
                    print(f"  ✓ {change}")
    
    # 保存
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ 转换完成！共处理 {total_changes}/{len(scenes)} 个场景")
        print(f"✓ 已保存到: {json_path}")
    
    return True, total_changes

def main():
    if len(sys.argv) > 1:
        # 从命令行参数获取文件列表
        file_numbers = sys.argv[1:]
    else:
        # 默认处理 3-11.json
        file_numbers = [str(i) for i in range(3, 12)]
    
    base_path = Path("lingjie/scenes")
    
    total_files = 0
    total_scenes = 0
    
    for file_num in file_numbers:
        json_path = base_path / f"{file_num}.json"
        print(f"\n{'='*60}")
        print(f"处理文件: {json_path}")
        print(f"{'='*60}")
        
        success, scene_count = process_json_file(json_path, verbose=True)
        if success:
            total_files += 1
            total_scenes += scene_count
    
    print(f"\n{'='*60}")
    print(f"批量处理完成！")
    print(f"  - 处理文件数: {total_files}")
    print(f"  - 处理场景数: {total_scenes}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

