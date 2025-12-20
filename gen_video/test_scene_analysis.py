#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试场景分析流程，定位卡住位置
"""

import sys
import time
import traceback
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("测试场景分析流程")
print("="*60)

# 测试场景
test_prompt = "韩立, Gray-green desert floor, 韩立保持不动，静静体会脚下炽热的沙地。"
test_shot_type = "wide"

print(f"\n测试 Prompt: {test_prompt}")
print(f"测试 Shot Type: {test_shot_type}")

try:
    print("\n[1] 导入模块...")
    from utils.scene_analyzer import analyze_scene, LocalSceneAnalyzer
    print("  ✓ 导入成功")
    
    print("\n[2] 测试本地分析器...")
    local_analyzer = LocalSceneAnalyzer()
    print("  ✓ LocalSceneAnalyzer 初始化成功")
    
    print("\n[3] 调用 analyze_scene (use_llm=False)...")
    start_time = time.time()
    result = analyze_scene(
        prompt=test_prompt,
        current_shot_type=test_shot_type,
        use_llm=False
    )
    elapsed = time.time() - start_time
    print(f"  ✓ analyze_scene 完成 (耗时: {elapsed:.2f}秒)")
    print(f"    结果类型: {type(result)}")
    print(f"    推荐镜头: {result.recommended_shot_type.value}")
    print(f"    姿态类型: {result.posture_type}")
    
    print("\n[4] 测试 PostureController...")
    from utils.posture_controller import PostureController
    posture_controller = PostureController()
    print("  ✓ PostureController 初始化成功")
    
    if result.posture_type:
        print(f"\n[5] 获取姿态提示词 (posture={result.posture_type})...")
        start_time = time.time()
        posture_prompt = posture_controller.get_posture_prompt(result.posture_type, use_chinese=False)
        elapsed = time.time() - start_time
        print(f"  ✓ get_posture_prompt 完成 (耗时: {elapsed:.2f}秒)")
        print(f"    结果: {posture_prompt}")
    
    print("\n[6] 测试 Execution Planner V3...")
    from execution_planner_v3 import ExecutionPlannerV3
    print("  ✓ ExecutionPlannerV3 导入成功")
    
    # 创建测试场景
    test_scene = {
        "prompt": test_prompt,
        "camera": {"shot": test_shot_type, "angle": "eye_level"},
        "character": {"present": True, "id": "hanli", "face_visible": False, "visibility": "low"},
        "environment": {},
        "quality_target": {
            "style": "xianxia_anime",
            "detail_level": "high",
            "lighting_style": "soft",
            "motion_intensity": "static",
            "camera_stability": "stable"
        }
    }
    
    print("\n[7] 初始化 Execution Planner V3...")
    import yaml
    with open("gen_video/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    planner = ExecutionPlannerV3(config=config)
    print("  ✓ ExecutionPlannerV3 初始化成功")
    
    print("\n[8] 分析场景...")
    start_time = time.time()
    strategy = planner.analyze_scene(test_scene)
    elapsed = time.time() - start_time
    print(f"  ✓ analyze_scene 完成 (耗时: {elapsed:.2f}秒)")
    print(f"    参考强度: {strategy.reference_strength}%")
    print(f"    身份引擎: {strategy.identity_engine.value}")
    
    print("\n[9] 构建 Prompt...")
    start_time = time.time()
    final_prompt = planner.build_prompt(
        scene=test_scene,
        original_prompt=test_prompt
    )
    elapsed = time.time() - start_time
    print(f"  ✓ build_prompt 完成 (耗时: {elapsed:.2f}秒)")
    print(f"    最终 Prompt: {final_prompt[:200]}...")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    traceback.print_exc()
    sys.exit(1)

