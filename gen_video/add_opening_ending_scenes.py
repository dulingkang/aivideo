#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每一集添加通用的开始和结束场景
"""

import json
from pathlib import Path

# 通用开始场景
OPENING_SCENE = {
    "id": 0,  # 开始场景ID为0
    "duration": 3,
    "description": "画面淡入，展现本集标题画面，背景音乐渐起。",
    "mood": "平静",
    "lighting": "柔和",
    "action": "标题展示",
    "camera": "固定镜头",
    "visual": {
        "composition": "标题画面居中显示",
        "environment": "背景画面",
        "character_pose": "",
        "fx": "淡入效果",
        "motion": "固定镜头"
    },
    "prompt": "画面淡入，展现本集标题画面，背景音乐渐起。",
    "narration": "我是云卷仙音，欢迎回来。",
    "face_style_auto": {
        "expression": "平静",
        "lighting": "柔和",
        "detail": "自然",
        "strength": 0.8
    }
}

# 通用结束场景
ENDING_SCENE = {
    "id": 999,  # 结束场景ID为999
    "duration": 3,
    "description": "画面淡出，显示下集预告或结束画面。",
    "mood": "平静",
    "lighting": "柔和",
    "action": "结束画面展示",
    "camera": "固定镜头",
    "visual": {
        "composition": "结束画面居中显示",
        "environment": "背景画面",
        "character_pose": "",
        "fx": "淡出效果",
        "motion": "固定镜头"
    },
    "prompt": "画面淡出，显示下集预告或结束画面。",
    "narration": "下一集再见。我是云卷仙音，原著作者忘语。",
    "face_style_auto": {
        "expression": "平静",
        "lighting": "柔和",
        "detail": "自然",
        "strength": 0.8
    }
}

def add_opening_ending_scenes(json_path: Path):
    """为JSON文件添加开始和结束场景"""
    print(f"处理: {json_path.name}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    
    # 检查是否已经有开始和结束场景
    has_opening_scene = False
    has_ending_scene = False
    
    if scenes:
        # 检查第一个场景是否为开始场景
        if scenes[0].get("id") == 0:
            has_opening_scene = True
        # 检查最后一个场景是否为结束场景
        if scenes[-1].get("id") == 999:
            has_ending_scene = True
    
    updated = False
    
    # 添加开始场景
    if not has_opening_scene:
        # 创建开始场景的副本
        opening_scene = OPENING_SCENE.copy()
        # 使用episode信息更新narration（如果有）
        episode = data.get("episode", "")
        title = data.get("title", "")
        if episode and title:
            opening_scene["narration"] = f"我是云卷仙音，欢迎回来。第{episode}集：{title}。"
        
        scenes.insert(0, opening_scene)
        updated = True
        print(f"  添加了开始场景")
    
    # 添加结束场景
    if not has_ending_scene:
        # 创建结束场景的副本
        ending_scene = ENDING_SCENE.copy()
        # 使用episode信息更新narration（如果有）
        episode = data.get("episode", "")
        next_episode = episode + 1 if episode else None
        if next_episode:
            ending_scene["narration"] = f"下一集再见。我是云卷仙音，原著作者忘语。第{next_episode}集即将开始。"
        
        scenes.append(ending_scene)
        updated = True
        print(f"  添加了结束场景")
    
    if updated:
        data["scenes"] = scenes
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 已更新")
    else:
        print(f"  - 无需更新（已有开始和结束场景）")

def main():
    """主函数"""
    base_dir = Path("lingjie/scenes")
    
    # 处理所有集数（1-11）
    for i in range(1, 12):
        json_path = base_dir / f"{i}.json"
        if json_path.exists():
            add_opening_ending_scenes(json_path)
        else:
            print(f"文件不存在: {json_path}")

if __name__ == "__main__":
    main()

