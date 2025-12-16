#!/usr/bin/env python3
"""诊断场景生成问题：检查scene_002和scene_004是否正确识别韩立并加载LoRA"""

import json
import sys
from pathlib import Path

def analyze_scene(scene, scene_id):
    """分析场景配置"""
    print(f"\n{'='*60}")
    print(f"分析场景 {scene_id} (id={scene.get('id')})")
    print(f"{'='*60}")
    
    # 场景基本信息
    print(f"\n📝 场景描述: {scene.get('description', '')}")
    print(f"📝 Prompt: {scene.get('prompt', '')}")
    
    # 检查是否包含韩立关键词
    keywords = ["han li", "hanli", "韩立", "主角"]
    combined_text = " ".join([
        scene.get("title", ""),
        scene.get("description", ""),
        scene.get("prompt", ""),
        scene.get("narration", ""),
    ]).lower()
    
    print(f"\n🔍 角色关键词检测:")
    found_keywords = []
    for kw in keywords:
        if kw in combined_text:
            found_keywords.append(kw)
            print(f"  ✓ 找到关键词: '{kw}'")
    
    if not found_keywords:
        print(f"  ❌ 未找到任何韩立关键词！这可能是问题所在。")
    
    # 检查character_pose
    visual = scene.get("visual", {}) or {}
    character_pose = visual.get("character_pose", "")
    if character_pose:
        print(f"\n🎭 Character Pose: {character_pose}")
        combined_text += " " + character_pose.lower()
    
    # 检查characters字段
    characters = scene.get("characters", [])
    if characters:
        print(f"\n👥 Characters字段: {characters}")
    else:
        print(f"\n👥 Characters字段: 未设置")
    
    # 检查camera类型
    camera = scene.get("camera", "")
    print(f"\n📷 Camera: {camera}")
    
    # 预测会被识别为什么角色
    if any(kw in combined_text for kw in ["han li", "hanli", "韩立"]):
        print(f"\n✅ 应该被识别为: hanli")
        return "hanli"
    else:
        print(f"\n❌ 可能不会被识别为hanli")
        return None

def main():
    script_json = Path(__file__).parent.parent / "lingjie" / "episode" / "1.json"
    
    if not script_json.exists():
        print(f"❌ 脚本文件不存在: {script_json}")
        return
    
    with open(script_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    
    # 分析scene_002 (id=1)
    scene_002 = None
    scene_004 = None
    
    for scene in scenes:
        scene_id = scene.get("id")
        if scene_id == 1:  # scene_002
            scene_002 = scene
        elif scene_id == 3:  # scene_004 (因为从0开始计数)
            scene_004 = scene
    
    if scene_002:
        analyze_scene(scene_002, "scene_002")
    else:
        print("❌ 未找到scene_002 (id=1)")
    
    if scene_004:
        analyze_scene(scene_004, "scene_004")
    else:
        print("❌ 未找到scene_004 (id=3)")
    
    print(f"\n{'='*60}")
    print("💡 建议检查:")
    print("1. 生成日志中是否有 '检测到角色: hanli（韩立）' 的输出")
    print("2. 生成日志中是否有 '自动加载LoRA: hanli' 的输出")
    print("3. 生成日志中是否有 '使用韩立的参考图' 的输出")
    print("4. InstantID的ip_adapter_scale和controlnet_conditioning_scale是否正确设置")
    print("5. LoRA权重是否正确应用（alpha=0.70）")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

