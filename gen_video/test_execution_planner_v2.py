#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution Planner v2 测试脚本
展示如何从 Scene JSON v2 自动选择图像生成引擎
"""

import json
import sys
from pathlib import Path

# 添加 gen_video 到路径
sys.path.insert(0, str(Path(__file__).parent))

from model_selector import ModelSelector


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return None


def test_execution_planner():
    """测试 Execution Planner v2"""
    print("=" * 60)
    print("Execution Planner v2 测试")
    print("=" * 60)
    print()
    
    # 加载配置
    config = load_config()
    if not config:
        return
    
    # 创建 ModelSelector
    selector = ModelSelector(config)
    
    # 加载 v2 JSON
    v2_json_path = Path(__file__).parent.parent / "lingjie" / "episode" / "1.v2.json"
    if not v2_json_path.exists():
        print(f"❌ v2 JSON 文件不存在: {v2_json_path}")
        return
    
    with open(v2_json_path, "r", encoding="utf-8") as f:
        episode_data = json.load(f)
    
    scenes = episode_data.get("scenes", [])
    print(f"📖 加载了 {len(scenes)} 个场景")
    print()
    
    # 测试前 5 个场景
    test_scenes = scenes[:5]
    
    for idx, scene in enumerate(test_scenes):
        scene_id = scene.get("scene_id", idx)
        scene_role = scene.get("scene_role", "")
        character = scene.get("character", {}) or {}
        camera = scene.get("camera", {}) or {}
        intent = scene.get("intent", {}) or {}
        
        print(f"场景 {scene_id} ({scene_role}):")
        print(f"  - 角色: {'有' if character.get('present') else '无'}")
        if character.get("present"):
            print(f"  - 可见度: {character.get('visibility', 'unknown')}")
            print(f"  - 脸部可见: {character.get('face_visible', False)}")
        print(f"  - 镜头: {camera.get('shot', 'unknown')}")
        print(f"  - 意图: {intent.get('type', 'unknown')}")
        
        # 调用 Execution Planner
        decision = selector.select_engine_for_scene_v2(scene)
        
        print(f"  → 决策结果:")
        print(f"     引擎: {decision['engine']}")
        print(f"     模式: {decision['mode']}")
        print(f"     锁脸: {decision['lock_face']}")
        print(f"     任务类型: {decision['task_type']}")
        print()
    
    # 统计结果
    print("=" * 60)
    print("统计结果")
    print("=" * 60)
    
    engine_counts = {}
    for scene in scenes:
        decision = selector.select_engine_for_scene_v2(scene)
        engine = decision['engine']
        engine_counts[engine] = engine_counts.get(engine, 0) + 1
    
    for engine, count in sorted(engine_counts.items()):
        percentage = (count / len(scenes)) * 100
        print(f"  {engine}: {count} 个场景 ({percentage:.1f}%)")
    
    print()
    print("✅ 测试完成")


if __name__ == "__main__":
    test_execution_planner()

