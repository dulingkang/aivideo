#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Execution Executor V2.1

功能：
1. 测试JSON转换器
2. 测试Execution Validator
3. 测试Execution Executor
4. 验证v2.1-exec格式的完整流程
"""

import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入utils模块（避免gen_video包的依赖）
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from json_v2_to_v21_converter import JSONV2ToV21Converter
from execution_validator import ExecutionValidator
from execution_executor_v21 import (
    ExecutionExecutorV21,
    ExecutionConfig,
    ExecutionMode
)


def test_converter():
    """测试JSON转换器"""
    print("=" * 60)
    print("测试1: JSON v2 → v2.1-exec 转换器")
    print("=" * 60)
    
    # 创建测试场景（v2格式）
    scene_v2 = {
        "scene_id": 1,
        "episode_id": "test_ep1",
        "version": "v2",
        "intent": {
            "type": "character_intro",
            "emotion": "tense"
        },
        "camera": {
            "shot": "wide",
            "angle": "top_down"
        },
        "character": {
            "present": True,
            "id": "hanli",
            "gender": "male",
            "pose": "lying_motionless",
            "face_visible": False
        },
        "visual_constraints": {
            "environment": "Gray-green desert floor"
        },
        "prompt": "Han Li lying on the desert floor",
        "narration": {
            "text": "韩立保持不动，静静体会脚下炽热的沙地。",
            "voice_id": "yunjuan_xianyin"
        }
    }
    
    # 转换
    converter = JSONV2ToV21Converter()
    scene_v21 = converter.convert_scene(scene_v2)
    
    print("\n✓ 转换成功")
    print(f"  Shot: {scene_v21['shot']['type']} (锁定: {scene_v21['shot']['locked']})")
    print(f"  Pose: {scene_v21['pose']['type']} (修正: {scene_v21['pose'].get('auto_corrected', False)})")
    print(f"  Model: {scene_v21['model_route']['base_model']} + {scene_v21['model_route']['identity_engine']}")
    print(f"  决策原因: {scene_v21['model_route'].get('decision_reason', 'N/A')}")
    
    # 保存测试结果
    output_path = Path(__file__).parent / "test_outputs" / "scene_v21_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scene_v21, f, ensure_ascii=False, indent=2)
    print(f"\n  测试结果已保存: {output_path}")
    
    return scene_v21


def test_validator(scene_v21):
    """测试Execution Validator"""
    print("\n" + "=" * 60)
    print("测试2: Execution Validator")
    print("=" * 60)
    
    validator = ExecutionValidator()
    result = validator.validate_scene(scene_v21)
    
    print(f"\n校验结果: {'✓ 通过' if result.is_valid else '✗ 失败'}")
    print(f"  错误数: {result.errors_count}")
    print(f"  警告数: {result.warnings_count}")
    
    if result.issues:
        print("\n问题列表:")
        for issue in result.issues:
            level_icon = "❌" if issue.level.value == "error" else "⚠️" if issue.level.value == "warning" else "ℹ️"
            print(f"  {level_icon} [{issue.field}] {issue.message}")
            if issue.suggestion:
                print(f"     建议: {issue.suggestion}")
    
    # 生成报告
    report = validator.generate_report(result, scene_v21.get("scene_id"))
    print("\n" + report)
    
    return result.is_valid


def test_executor(scene_v21):
    """测试Execution Executor"""
    print("\n" + "=" * 60)
    print("测试3: Execution Executor V2.1")
    print("=" * 60)
    
    # 创建执行器（严格模式）
    config = ExecutionConfig(mode=ExecutionMode.STRICT)
    executor = ExecutionExecutorV21(config=config)
    
    # 执行场景（模拟模式，不实际生成）
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n执行模式: {config.mode.value}")
    print(f"输出目录: {output_dir}")
    
    # 构建决策trace
    decision_trace = executor._build_decision_trace(scene_v21)
    print("\n决策Trace:")
    print(json.dumps(decision_trace, ensure_ascii=False, indent=2))
    
    # 构建Prompt
    prompt = executor._build_prompt(scene_v21)
    print(f"\n生成的Prompt:")
    print(f"  {prompt}")
    
    # 构建Negative Prompt
    negative = executor._build_negative_prompt(scene_v21)
    print(f"\nNegative Prompt:")
    print(f"  {', '.join(negative[:5])}...")
    
    print("\n✓ 执行器测试完成（模拟模式）")
    print("  注意: 实际生成需要集成ImageGenerator/VideoGenerator")


def test_pose_correction():
    """测试Pose修正策略（两级）"""
    print("\n" + "=" * 60)
    print("测试4: Pose修正策略（两级）")
    print("=" * 60)
    
    from execution_rules_v2_1 import get_execution_rules, ShotType
    
    rules = get_execution_rules()
    
    # 测试Level 1修正（硬规则冲突）
    print("\nLevel 1修正（硬规则冲突）:")
    pose_decision = rules.validate_pose(ShotType.WIDE, "lying")
    print(f"  wide + lying → {pose_decision.pose_type.value}")
    print(f"  修正级别: {pose_decision.correction_level}")
    print(f"  修正原因: {pose_decision.correction_reason}")
    
    # 测试Level 2修正（语义冲突）
    print("\nLevel 2修正（语义冲突）:")
    story_context = {
        "requires_walking": True,
        "is_injured": False
    }
    pose_decision2 = rules.validate_pose(ShotType.MEDIUM, "stand", story_context)
    print(f"  medium + stand (需要行走) → {pose_decision2.pose_type.value}")
    if pose_decision2.correction_level == "level2":
        print(f"  修正级别: {pose_decision2.correction_level}")
        print(f"  修正原因: {pose_decision2.correction_reason}")


def test_character_anchor():
    """测试角色锚系统"""
    print("\n" + "=" * 60)
    print("测试5: 角色锚系统")
    print("=" * 60)
    
    from character_anchor_v2_1 import get_character_anchor_manager
    
    anchor_manager = get_character_anchor_manager()
    
    # 注册角色
    anchor_manager.register_character(
        character_id="hanli",
        gender="male",
        lora_path="hanli_character_v1.safetensors",
        lora_weight=0.6,
        instantid_enabled=True,
        instantid_strength=0.75
    )
    
    # 获取角色锚
    anchor = anchor_manager.get_anchor("hanli")
    print(f"\n角色锚配置:")
    print(f"  LoRA: {anchor.lora_path} (权重: {anchor.lora_weight})")
    print(f"  InstantID: {'启用' if anchor.instantid_enabled else '禁用'} (强度: {anchor.instantid_strength})")
    print(f"  性别负锁: {len(anchor.gender_negative_lock)} 项")
    
    # 判断是否使用InstantID
    should_use = anchor_manager.should_use_instantid("hanli", face_visible=True)
    print(f"  应该使用InstantID: {should_use}")
    
    # 获取性别负锁
    negative = anchor_manager.get_negative_prompt_with_gender_lock("hanli")
    print(f"\n性别负锁示例:")
    print(f"  {', '.join(negative[:5])}...")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Execution Executor V2.1 测试套件")
    print("=" * 60)
    
    try:
        # 测试1: JSON转换器
        scene_v21 = test_converter()
        
        # 测试2: Execution Validator
        is_valid = test_validator(scene_v21)
        if not is_valid:
            print("\n⚠️  JSON校验失败，但继续测试...")
        
        # 测试3: Execution Executor
        test_executor(scene_v21)
        
        # 测试4: Pose修正策略
        test_pose_correction()
        
        # 测试5: 角色锚系统
        test_character_anchor()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

