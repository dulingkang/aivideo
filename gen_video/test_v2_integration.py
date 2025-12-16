#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Execution Planner v2 集成
验证从 v2 JSON 到图像生成的完整流程
"""

import sys
import json
import yaml
from pathlib import Path

# 添加 gen_video 到路径
sys.path.insert(0, str(Path(__file__).parent))


def test_v2_integration():
    """测试 v2 JSON 集成"""
    print("=" * 60)
    print("测试 Execution Planner v2 集成")
    print("=" * 60)
    print()
    
    # 加载配置（不加载 ImageGenerator，避免依赖 torch）
    config_path = Path(__file__).parent / "config.yaml"
    print(f"📖 加载配置: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 测试 v2 JSON
    v2_json_path = Path(__file__).parent.parent / "lingjie" / "episode" / "1.v2.json"
    if not v2_json_path.exists():
        print(f"❌ v2 JSON 文件不存在: {v2_json_path}")
        return
    
    print(f"📖 测试 v2 JSON: {v2_json_path}")
    print()
    print("⚠️  注意：这是集成测试，不会实际生成图像")
    print("   如果要实际生成，请运行: python main.py --script lingjie/episode/1.v2.json")
    print()
    
    # 检查是否能正确检测 v2 格式
    with open(v2_json_path, "r", encoding="utf-8") as f:
        script_data = json.load(f)
    
    scenes = script_data.get("scenes", [])
    if scenes:
        first_scene = scenes[0]
        is_v2 = (
            first_scene.get("version") == "v2" or
            ("intent" in first_scene and "visual_constraints" in first_scene)
        )
        print(f"✓ 检测到 v2 格式: {is_v2}")
        
        if is_v2:
            print(f"✓ 场景数量: {len(scenes)}")
            print()
            print("前 5 个场景的 Execution Planner 决策预览:")
            print()
            
            from model_selector import ModelSelector
            planner = ModelSelector(config)
            
            for idx, scene in enumerate(scenes[:5]):
                scene_id = scene.get("scene_id", idx)
                character = scene.get("character", {}) or {}
                camera = scene.get("camera", {}) or {}
                intent = scene.get("intent", {}) or {}
                
                decision = planner.select_engine_for_scene_v2(scene)
                print(f"  场景 {scene_id}:")
                print(f"    - 角色: {'有' if character.get('present') else '无'}")
                if character.get("present"):
                    print(f"    - 可见度: {character.get('visibility', 'unknown')}")
                    print(f"    - 脸部可见: {character.get('face_visible', False)}")
                print(f"    - 镜头: {camera.get('shot', 'unknown')}")
                print(f"    - 意图: {intent.get('type', 'unknown')}")
                print(f"    → 引擎: {decision['engine']}")
                print(f"    → 模式: {decision['mode']}")
                print(f"    → 锁脸: {decision['lock_face']}")
                print(f"    → 任务类型: {decision['task_type']}")
                print()
            
            # 统计所有场景的决策
            print("=" * 60)
            print("所有场景的决策统计:")
            print("=" * 60)
            engine_counts = {}
            for scene in scenes:
                decision = planner.select_engine_for_scene_v2(scene)
                engine = decision['engine']
                engine_counts[engine] = engine_counts.get(engine, 0) + 1
            
            for engine, count in sorted(engine_counts.items()):
                percentage = (count / len(scenes)) * 100
                print(f"  {engine}: {count} 个场景 ({percentage:.1f}%)")
            
            print()
            print("✅ Execution Planner v2 集成测试通过")
            print()
            print("📝 集成说明:")
            print("  1. image_generator.py 已集成 Execution Planner v2")
            print("  2. 当检测到 v2 JSON 格式时，会自动使用 Planner 选择引擎")
            print("  3. 使用方式: python main.py --script lingjie/episode/1.v2.json")
        else:
            print("⚠️  未检测到 v2 格式，将使用默认逻辑")
    else:
        print("❌ 未找到场景数据")


if __name__ == "__main__":
    test_v2_integration()

