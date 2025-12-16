#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复中英混合的 action 和 camera 字段
"""

import json
import re
from pathlib import Path

# 英文单词翻译映射
word_map = {
    "mutant": "变异", "wolves": "狼群", "wolf": "狼",
    "up": "向上", "around": "周围", "caravan": "车队",
    "caged": "笼中", "beast": "妖兽", "cagedbeast": "笼中妖兽",
    "quarters": "住所", "against": "对抗", "bolts": "闪电",
    "tend": "照料", "after": "之后", "战斗": "战斗",
    "meditate": "打坐", "思考": "思考", "gate": "门",
    "打开": "打开", "s": "", "city": "城市", "deliver": "交付",
    "去": "去", "ods": "货物", "request": "请求", "离开": "离开",
    "行走": "行走", "见面": "见面", "luo": "罗", "面部": "面对",
    "guards": "守卫", "进入": "进入", "meeting": "会议",
    "回答": "回答", "questions": "问题", "出发": "出发",
    "secretly": "秘密地", "听到": "听到", "alarm": "警报",
    "观察": "观察", "pythons": "蟒蛇", "战斗": "战斗",
    "clash": "冲突", "casualties": "伤亡", "看到": "看到",
    "devastation": "毁灭", "call": "呼叫", "aerial": "空中",
    "ruins": "废墟", "citylord": "城主", "到达": "到达",
    "s": "", "suspect": "怀疑", "出发": "出发", "ure": "出发",
    "secret": "秘密", "beasts": "妖兽", "攻击": "攻击",
    "dodge": "闪避", "s": "", "杀死": "杀死", "snakes": "蛇",
    "all": "所有", "beasts": "妖兽", "站": "站", "off": "离开",
    "特写": "特写", "up": "向上",
}

def has_chinese(text: str) -> bool:
    """检查文本是否包含中文"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def has_english(text: str) -> bool:
    """检查文本是否包含英文"""
    if not text:
        return False
    return bool(re.search(r'[a-zA-Z]', str(text)))

def translate_english_word(word: str) -> str:
    """翻译单个英文单词"""
    word_lower = word.lower().strip()
    return word_map.get(word_lower, word)

def fix_mixed_text(text: str) -> str:
    """修复中英混合的文本"""
    if not text:
        return text
    
    # 如果已经是纯中文或纯英文，直接返回
    if not (has_chinese(text) and has_english(text)):
        return text
    
    # 提取英文部分并翻译
    parts = re.split(r'([a-zA-Z]+)', text)
    
    result = ""
    for part in parts:
        if not part:
            continue
        if re.match(r'^[a-zA-Z]+$', part):
            # 这是英文部分，尝试翻译
            translated = translate_english_word(part)
            if translated != part:
                result += translated
            else:
                # 如果无法翻译，尝试分割复合单词
                if len(part) > 1:
                    # 尝试按常见单词边界分割
                    split_words = []
                    i = 0
                    while i < len(part):
                        matched = False
                        for w in sorted(word_map.keys(), key=len, reverse=True):
                            if part[i:].lower().startswith(w.lower()):
                                split_words.append(part[i:i+len(w)])
                                i += len(w)
                                matched = True
                                break
                        if not matched:
                            split_words.append(part[i])
                            i += 1
                    
                    translated_parts = []
                    for w in split_words:
                        tw = translate_english_word(w)
                        if tw != w:
                            translated_parts.append(tw)
                        else:
                            translated_parts.append(w)
                    result += "".join(translated_parts)
                else:
                    result += part
        else:
            # 这是中文或其他字符，直接保留
            result += part
    
    return result

def process_json_file(json_path: Path):
    """处理单个 JSON 文件"""
    print(f"处理: {json_path.name}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    updated_count = 0
    
    for scene in scenes:
        # 修复 action
        action = scene.get("action", "")
        if action:
            new_action = fix_mixed_text(action)
            if new_action != action:
                scene["action"] = new_action
                updated_count += 1
        
        # 修复 camera
        camera = scene.get("camera", "")
        if camera:
            new_camera = fix_mixed_text(camera)
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

