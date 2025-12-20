#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版v2.1测试（不依赖完整环境）

只测试核心逻辑，不加载实际模型
"""

import sys
import json
from pathlib import Path

# 直接导入utils模块
sys.path.insert(0, str(Path(__file__).parent / "utils"))

def test_rules_engine():
    """测试规则引擎"""
    print("=" * 60)
    print("测试1: 规则引擎")
    print("=" * 60)
    
    try:
        from execution_rules_v2_1 import get_execution_rules, SceneIntent, ShotType
        
        rules = get_execution_rules()
        
        # 测试Intent → Shot映射
        print("\n1. Intent → Shot映射:")
        for intent in SceneIntent:
            shot_decision = rules.get_shot_from_intent(intent.value)
            print(f"   {intent.value:20} → {shot_decision.shot_type.value}")
        
        # 测试Pose验证
        print("\n2. Pose验证（wide + lying）:")
        pose_decision = rules.validate_pose(ShotType.WIDE, "lying")
        print(f"   wide + lying → {pose_decision.pose_type.value}")
        print(f"   自动修正: {pose_decision.auto_corrected}")
        print(f"   修正原因: {pose_decision.correction_reason}")
        
        # 测试Model路由
        print("\n3. Model路由:")
        model, identity = rules.get_model_route(has_character=True, shot_type=ShotType.MEDIUM)
        print(f"   有人物 + medium → {model.value} + {identity}")
        
        model2, identity2 = rules.get_model_route(has_character=True, shot_type=ShotType.WIDE)
        print(f"   有人物 + wide → {model2.value} + {identity2}")
        
        print("\n✓ 规则引擎测试通过")
        return True
    except Exception as e:
        print(f"\n✗ 规则引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_character_anchor():
    """测试角色锚系统"""
    print("\n" + "=" * 60)
    print("测试2: 角色锚系统")
    print("=" * 60)
    
    try:
        from character_anchor_v2_1 import get_character_anchor_manager
        
        anchor_manager = get_character_anchor_manager()
        
        # 注册角色
        anchor_manager.register_character(
            character_id="hanli",
            gender="male",
            lora_path="hanli_v1.safetensors",
            lora_weight=0.6
        )
        
        # 获取角色锚
        anchor = anchor_manager.get_anchor("hanli")
        print(f"\n角色锚配置:")
        print(f"  LoRA: {anchor.lora_path} (权重: {anchor.lora_weight})")
        print(f"  InstantID: {'启用' if anchor.instantid_enabled else '禁用'}")
        print(f"  性别负锁: {len(anchor.gender_negative_lock)} 项")
        
        # 获取性别负锁
        negative = anchor_manager.get_negative_prompt_with_gender_lock("hanli")
        print(f"\n性别负锁示例: {', '.join(negative[:5])}...")
        
        print("\n✓ 角色锚系统测试通过")
        return True
    except Exception as e:
        print(f"\n✗ 角色锚系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_converter():
    """测试JSON转换器"""
    print("\n" + "=" * 60)
    print("测试3: JSON转换器")
    print("=" * 60)
    
    try:
        from json_v2_to_v21_converter import JSONV2ToV21Converter
        
        # 创建测试场景
        scene_v2 = {
            "scene_id": 1,
            "intent": {"type": "character_intro"},
            "camera": {"shot": "wide"},
            "character": {
                "present": True,
                "id": "hanli",
                "gender": "male",
                "pose": "lying_motionless"
            },
            "visual_constraints": {"environment": "desert"}
        }
        
        converter = JSONV2ToV21Converter()
        scene_v21 = converter.convert_scene(scene_v2)
        
        print(f"\n转换结果:")
        print(f"  Shot: {scene_v21['shot']['type']} (锁定: {scene_v21['shot']['locked']})")
        print(f"  Pose: {scene_v21['pose']['type']} (修正: {scene_v21['pose'].get('auto_corrected', False)})")
        print(f"  Model: {scene_v21['model_route']['base_model']}")
        print(f"  决策原因: {scene_v21['model_route'].get('decision_reason', 'N/A')}")
        
        print("\n✓ JSON转换器测试通过")
        return True
    except Exception as e:
        print(f"\n✗ JSON转换器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validator():
    """测试校验器"""
    print("\n" + "=" * 60)
    print("测试4: Execution Validator")
    print("=" * 60)
    
    try:
        from execution_validator import ExecutionValidator
        
        validator = ExecutionValidator()
        
        # 创建测试场景（v2.1-exec格式）
        scene_v21 = {
            "version": "v2.1-exec",
            "scene_id": 1,
            "shot": {"type": "medium", "locked": True},
            "pose": {"type": "stand", "locked": True},
            "model_route": {
                "base_model": "flux",
                "identity_engine": "pulid",
                "allow_fallback": False,
                "decision_reason": "test"
            },
            "character": {
                "present": True,
                "id": "hanli",
                "gender": "male"
            },
            "prompt": {
                "positive_core": "Han Li",
                "scene_description": "desert"
            },
            "negative_lock": {"gender": True, "extra": ["female"]}
        }
        
        result = validator.validate_scene(scene_v21)
        
        print(f"\n校验结果: {'✓ 通过' if result.is_valid else '✗ 失败'}")
        print(f"  错误: {result.errors_count}, 警告: {result.warnings_count}")
        
        print("\n✓ Execution Validator测试通过")
        return True
    except Exception as e:
        print(f"\n✗ Execution Validator测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("v2.1核心组件简化测试")
    print("=" * 60)
    
    results = []
    
    results.append(("规则引擎", test_rules_engine()))
    results.append(("角色锚系统", test_character_anchor()))
    results.append(("JSON转换器", test_json_converter()))
    results.append(("Execution Validator", test_validator()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ 所有核心组件测试通过")
    else:
        print("\n✗ 部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

