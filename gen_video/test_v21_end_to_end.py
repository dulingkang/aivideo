#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.1系统端到端测试

测试完整流程：
1. JSON转换（v2 → v2.1-exec）
2. JSON校验
3. 场景生成（使用Execution Executor）
"""

import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

def test_conversion_and_validation():
    """测试转换和校验"""
    print("=" * 60)
    print("测试1: JSON转换和校验")
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
        },
        "duration_sec": 4.0,
        "target_fps": 24
    }
    
    try:
        # 直接导入，避免通过__init__.py（可能有依赖问题）
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "json_v2_to_v21_converter",
            Path(__file__).parent / "utils" / "json_v2_to_v21_converter.py"
        )
        converter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter_module)
        JSONV2ToV21Converter = converter_module.JSONV2ToV21Converter
        
        spec2 = importlib.util.spec_from_file_location(
            "execution_validator",
            Path(__file__).parent / "utils" / "execution_validator.py"
        )
        validator_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(validator_module)
        ExecutionValidator = validator_module.ExecutionValidator
        
        # 转换
        converter = JSONV2ToV21Converter()
        scene_v21 = converter.convert_scene(scene_v2)
        
        print(f"\n✓ 转换成功")
        print(f"  Shot: {scene_v21['shot']['type']} (锁定: {scene_v21['shot']['locked']})")
        print(f"  Pose: {scene_v21['pose']['type']} (修正: {scene_v21['pose'].get('auto_corrected', False)})")
        print(f"  Model: {scene_v21['model_route']['base_model']} + {scene_v21['model_route']['identity_engine']}")
        
        # 校验
        validator = ExecutionValidator()
        result = validator.validate_scene(scene_v21)
        
        print(f"\n✓ 校验结果: {'通过' if result.is_valid else '失败'}")
        print(f"  错误: {result.errors_count}, 警告: {result.warnings_count}")
        
        return scene_v21, result.is_valid
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_executor_prompt_building():
    """测试Executor的Prompt构建"""
    print("\n" + "=" * 60)
    print("测试2: Execution Executor Prompt构建")
    print("=" * 60)
    
    try:
        # 直接导入
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "execution_executor_v21",
            Path(__file__).parent / "utils" / "execution_executor_v21.py"
        )
        executor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(executor_module)
        ExecutionExecutorV21 = executor_module.ExecutionExecutorV21
        ExecutionConfig = executor_module.ExecutionConfig
        ExecutionMode = executor_module.ExecutionMode
        
        # 创建测试场景（v2.1-exec格式）
        scene_v21 = {
            "version": "v2.1-exec",
            "scene_id": 1,
            "shot": {"type": "medium", "locked": True},
            "pose": {"type": "stand", "locked": True},
            "character": {
                "present": True,
                "id": "hanli",
                "gender": "male"
            },
            "prompt": {
                "positive_core": "Han Li, calm expression",
                "scene_description": "quiet forest clearing",
                "style": "xianxia"
            },
            "negative_lock": {
                "gender": True,
                "extra": ["female", "woman"]
            }
        }
        
        # 创建执行器
        config = ExecutionConfig(mode=ExecutionMode.STRICT)
        executor = ExecutionExecutorV21(config=config)
        
        # 构建Prompt
        prompt = executor._build_prompt(scene_v21)
        negative = executor._build_negative_prompt(scene_v21)
        
        print(f"\n生成的Prompt:")
        print(f"  {prompt}")
        print(f"\nNegative Prompt:")
        print(f"  {', '.join(negative[:5])}...")
        
        # 构建决策trace
        decision_trace = executor._build_decision_trace(scene_v21)
        print(f"\n决策Trace:")
        print(json.dumps(decision_trace, ensure_ascii=False, indent=2))
        
        print("\n✓ Prompt构建测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_correction_levels():
    """测试Pose修正的两级策略"""
    print("\n" + "=" * 60)
    print("测试3: Pose修正策略（两级）")
    print("=" * 60)
    
    try:
        # 直接导入
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "execution_rules_v2_1",
            Path(__file__).parent / "utils" / "execution_rules_v2_1.py"
        )
        rules_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules_module)
        get_execution_rules = rules_module.get_execution_rules
        ShotType = rules_module.ShotType
        
        rules = get_execution_rules()
        
        # Level 1: 硬规则冲突
        print("\nLevel 1修正（硬规则冲突）:")
        pose_decision1 = rules.validate_pose(ShotType.WIDE, "lying")
        print(f"  wide + lying → {pose_decision1.pose_type.value}")
        print(f"  修正级别: {pose_decision1.correction_level}")
        print(f"  修正原因: {pose_decision1.correction_reason}")
        
        # Level 2: 语义冲突
        print("\nLevel 2修正（语义冲突）:")
        story_context = {
            "requires_walking": True,
            "is_injured": False
        }
        pose_decision2 = rules.validate_pose(ShotType.MEDIUM, "stand", story_context)
        if pose_decision2.correction_level == "level2":
            print(f"  medium + stand (需要行走) → {pose_decision2.pose_type.value}")
            print(f"  修正级别: {pose_decision2.correction_level}")
            print(f"  修正原因: {pose_decision2.correction_reason}")
        else:
            print(f"  无需Level 2修正")
        
        print("\n✓ Pose修正策略测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("v2.1系统端到端测试")
    print("=" * 60)
    
    results = []
    
    # 测试1: 转换和校验
    scene_v21, is_valid = test_conversion_and_validation()
    results.append(("转换和校验", is_valid))
    
    # 测试2: Prompt构建
    results.append(("Prompt构建", test_executor_prompt_building()))
    
    # 测试3: Pose修正策略
    results.append(("Pose修正策略", test_pose_correction_levels()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ 所有端到端测试通过")
        print("\n下一步: 集成到主流程并测试实际生成")
    else:
        print("\n✗ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

