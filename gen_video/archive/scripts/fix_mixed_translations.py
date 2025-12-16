#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复中英混合的翻译问题
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List

def has_chinese(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def has_mixed_translation(text: str) -> bool:
    """检查是否包含中英混合翻译（包含下划线和中文）"""
    if not text:
        return False
    return has_chinese(text) and ('_' in text or any(c.isalpha() and ord(c) < 128 for c in text if c.isalpha()))

def fix_mixed_translation(text: str, field_type: str = "action") -> str:
    """修复混合翻译 - 如果无法完整翻译，保留英文原值"""
    if not has_mixed_translation(text):
        return text
    
    # 如果已经是纯中文，直接返回
    if has_chinese(text) and not any(c.isalpha() and ord(c) < 128 for c in text if c.isalpha() and c != '_'):
        # 只包含中文和下划线，尝试清理下划线
        return text.replace('_', '')
    
    # 如果是action字段，特殊处理
    if field_type == "action":
        # 常见混合翻译修复
        fixes = {
            "听到_visitors": "听到访客",
            "explain_purpose": "解释目的",
            "examine_body": "检查身体",
            "arrange_cushion": "安排垫子",
            "second_acupuncture": "第二次针灸",
            "move_arms": "移动手臂",
            "open_eyes": "睁开眼睛",
            "打开_eyes": "睁开眼睛",
            "enter_carriage": "进入车厢",
            "进入_carriage": "进入车厢",
            "hear_visitors": "听到访客",
            "拿_out_mirror": "拿出镜子",
            "shoot_去lden_needles": "发射金针",
            "进入_grassland": "进入草原",
            "观察_anxiety": "观察焦虑",
            "进入_bookshop": "进入书店",
            "离开_courtyard": "离开庭院",
            "听到_battle": "听到战斗",
            "杀死_python": "杀死蟒蛇",
            "triple_杀死": "三连杀",
            "返回_camp": "返回营地",
            "bone_spears_杀死": "骨矛杀死",
            "snake_杀死s_rider": "蛇杀死骑士",
            "杀死_wolf": "杀死狼",
            "close_up_door": "门特写",
            "close_up": "特写",
            "closeup": "特写",
            "medium_shot": "中景",
            "wide_shot": "全景",
            "low_angle": "低角度",
            "dynamic_close": "动态特写",
            "wide_action": "全景动作",
            "medium_two_shot": "双人中景",
            "close_up_hands": "手部特写",
            "wide_two_shot": "双人全景",
            "dodge_attacks": "躲避攻击",
            "kill_snakes": "杀死蛇",
            "kill_all_beasts": "杀死所有野兽",
            "standoff": "对峙"
        }
        
        if text in fixes:
            return fixes[text]
        
        # 处理包含下划线的action（尝试智能修复）
        if '_' in text:
            parts = text.split('_')
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
                'respectful': '尊重', 'miraculous': '神奇', 'carriage': '车厢',
                'mirror': '镜子', 'out': '出', 'grassland': '草原', 'anxiety': '焦虑',
                'bookshop': '书店', 'courtyard': '庭院', 'python': '蟒蛇', 'triple': '三连',
                'camp': '营地', 'bone': '骨', 'spears': '矛', 'snake': '蛇', 's': '的',
                'rider': '骑士', 'wolf': '狼', 'golden': '金', 'needles': '针', 'shoot': '发射'
            }
            
            translations = []
            for part in parts:
                if part.lower() in word_map:
                    translations.append(word_map[part.lower()])
                elif has_chinese(part):
                    translations.append(part)
                else:
                    # 如果无法翻译，保留原词
                    translations.append(part)
            
            if translations:
                return ''.join(translations)
    
    # 如果是camera字段
    if field_type == "camera":
        fixes = {
            "特写_up_门": "门特写",
            "特写_up": "特写",
            "特写_up_戒指": "戒指特写",
            "中景_双人_镜头": "双人中景",
            "低角度_角度": "低角度",
            "低角度_角度_揭示": "低角度揭示",
            "低角度_角度_动作": "低角度动作",
            "全景_双人_镜头": "双人全景",
            "全景_动作": "全景动作",
            "动态_特写": "动态特写",
            "特写_up_手": "手部特写",
            "航拍_circular_拉back": "航拍环形拉回",
            "中景_跟踪_lineup": "中景跟踪队列",
            "特写_up_transformation": "变换特写",
            "全景_低角度_角度_fol低角度": "全景低角度跟随",
            "s低角度_运动_side": "低角度运动侧面",
            "航拍_推进_forward": "航拍推进",
            "中景_inside_carriage": "车厢内中景",
            "双人_镜头_特写": "双人特写",
            "s低角度_推进": "低角度推进",
            "特写_up_文字": "文字特写",
            "tight_特写": "紧特写",
            "中景_动作": "中景动作",
            "全景_桌子": "桌子全景",
            "中景_三人_镜头": "三人中景",
            "中景_fol低角度": "中景跟随",
            "全景_crowd": "人群全景",
            "全景_月光": "月光全景",
            "中景_群体": "群体中景",
            "全景_stair": "楼梯全景",
            "中景_室内": "室内中景",
            "跟踪_镜头": "跟踪镜头",
            "全景_揭示": "全景揭示",
            "特写_双人_镜头": "双人特写"
        }
        
        if text in fixes:
            return fixes[text]
        
        # 处理包含下划线的camera名称
        if '_' in text and not has_chinese(text):
            parts = text.split('_')
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
                'up': '', 'of': '', 'the': '', 'a': '', 'an': '', 'and': '', 'or': ''
            }
            
            translations = []
            for part in parts:
                if part.lower() in word_map:
                    trans = word_map[part.lower()]
                    if trans:
                        translations.append(trans)
                else:
                    translations.append(part)
            
            if translations and all(has_chinese(t) or not t for t in translations):
                return ''.join(translations) if len(translations) == 1 else ''.join(translations)
        
        # 如果无法完整翻译，返回英文原值（去掉下划线，用空格代替）
        if '_' in text:
            return text.replace('_', ' ')
    
    # 如果无法修复，且包含下划线，用空格代替
    if '_' in text:
        return text.replace('_', ' ')
    
    return text  # 如果无法修复，返回原值

def fix_json_file(json_path: Path):
    """修复JSON文件中的混合翻译"""
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    fixed_count = 0
    
    for scene in scenes:
        scene_id = scene.get("id", "未知")
        changes = []
        
        # 修复action
        action = scene.get("action", "")
        if action and has_mixed_translation(action):
            new_action = fix_mixed_translation(action, "action")
            if new_action != action:
                scene["action"] = new_action
                changes.append(f"action: {action} -> {new_action}")
        
        # 修复camera
        camera = scene.get("camera", "")
        if camera and has_mixed_translation(camera):
            new_camera = fix_mixed_translation(camera, "camera")
            if new_camera != camera:
                scene["camera"] = new_camera
                changes.append(f"camera: {camera} -> {new_camera}")
        
        if changes:
            fixed_count += 1
            print(f"场景 {scene_id}:")
            for change in changes:
                print(f"  ✓ {change}")
    
    if fixed_count > 0:
        # 保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 修复完成！共修复 {fixed_count} 个场景")
    else:
        print("\n✓ 未发现需要修复的混合翻译")

def main():
    import sys
    
    if len(sys.argv) > 1:
        file_numbers = sys.argv[1:]
    else:
        file_numbers = [str(i) for i in range(3, 12)]
    
    base_path = Path("lingjie/scenes")
    
    for file_num in file_numbers:
        json_path = base_path / f"{file_num}.json"
        print(f"\n{'='*60}")
        print(f"处理文件: {json_path}")
        print(f"{'='*60}")
        fix_json_file(json_path)

if __name__ == "__main__":
    main()

