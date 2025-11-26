#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复所有混合翻译 - 对于无法完整翻译的，保留英文原值
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any

def has_chinese(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def has_mixed(text: str) -> bool:
    """检查是否包含中英混合（包含下划线和中文）"""
    if not text:
        return False
    # 检查是否同时包含中文和英文单词（通过下划线或字母判断）
    has_cn = has_chinese(text)
    has_en = bool(re.search(r'[a-zA-Z]', text)) and '_' in text
    return has_cn and has_en

def fix_mixed_field(value: str) -> str:
    """修复混合字段 - 如果无法完整翻译，保留英文"""
    if not has_mixed(value):
        return value
    
    # 如果包含下划线，尝试恢复为纯英文或纯中文
    if '_' in value:
        # 检查是否可以从备份恢复
        # 如果无法恢复，将下划线替换为空格，保留英文
        # 或者如果大部分是中文，只保留中文部分
        
        # 统计中文字符和英文字符数量
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', value))
        en_chars = len(re.findall(r'[a-zA-Z]', value))
        
        if cn_chars > en_chars:
            # 中文为主，尝试提取中文部分
            cn_parts = re.findall(r'[\u4e00-\u9fff]+', value)
            if cn_parts:
                return ''.join(cn_parts)
        else:
            # 英文为主，保留英文，去掉下划线用空格代替
            return value.replace('_', ' ')
    
    return value

def fix_json_file(json_path: Path):
    """修复JSON文件中的混合翻译"""
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        return
    
    # 检查备份文件
    backup_path = json_path.with_suffix(".json.bak")
    backup_data = None
    if backup_path.exists():
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
        except:
            pass
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    backup_scenes = backup_data.get("scenes", []) if backup_data else []
    
    fixed_count = 0
    
    for scene in scenes:
        scene_id = scene.get("id", "未知")
        changes = []
        
        # 尝试从备份恢复原值
        backup_scene = None
        if backup_scenes:
            backup_scene = next((s for s in backup_scenes if s.get("id") == scene_id), None)
        
        # 修复action
        action = scene.get("action", "")
        if action and has_mixed(action):
            # 优先从备份恢复
            if backup_scene and backup_scene.get("action"):
                new_action = backup_scene["action"]
            else:
                new_action = fix_mixed_field(action)
            
            if new_action != action:
                scene["action"] = new_action
                changes.append(f"action: {action} -> {new_action}")
        
        # 修复camera
        camera = scene.get("camera", "")
        if camera and has_mixed(camera):
            # 优先从备份恢复
            if backup_scene and backup_scene.get("camera"):
                new_camera = backup_scene["camera"]
            else:
                new_camera = fix_mixed_field(camera)
            
            if new_camera != camera:
                scene["camera"] = new_camera
                changes.append(f"camera: {camera} -> {new_camera}")
        
        # 修复mood
        mood = scene.get("mood", "")
        if mood and has_mixed(mood):
            if backup_scene and backup_scene.get("mood"):
                new_mood = backup_scene["mood"]
            else:
                new_mood = fix_mixed_field(mood)
            
            if new_mood != mood:
                scene["mood"] = new_mood
                changes.append(f"mood: {mood} -> {new_mood}")
        
        # 修复lighting
        lighting = scene.get("lighting", "")
        if lighting and has_mixed(lighting):
            if backup_scene and backup_scene.get("lighting"):
                new_lighting = backup_scene["lighting"]
            else:
                new_lighting = fix_mixed_field(lighting)
            
            if new_lighting != lighting:
                scene["lighting"] = new_lighting
                changes.append(f"lighting: {lighting} -> {new_lighting}")
        
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

