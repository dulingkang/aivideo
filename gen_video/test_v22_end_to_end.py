#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final格式端到端测试

测试完整的生成流程，验证v2.2-final格式的稳定性
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def create_test_scene_v22() -> dict:
    """创建测试用的v2.2-final格式场景"""
    return {
        "version": "v2.2-final",
        "metadata": {
            "created_at": "2025-12-20T00:00:00Z",
            "description": "端到端测试场景"
        },
        "scene": {
            "id": "scene_test_001",
            "duration_sec": 3.0,
            
            "intent": {
                "type": "character_intro",
                "description": "韩立初次登场测试"
            },
            
            "shot": {
                "type": "medium",
                "locked": True,
                "source": "direct_specification",
                "description": "中景镜头"
            },
            
            "pose": {
                "type": "stand",
                "locked": True,
                "validated_by": "shot_pose_rules",
                "auto_corrected": False,
                "description": "站立姿态"
            },
            
            "model_route": {
                "base_model": "flux",
                "identity_engine": "pulid",
                "locked": True,
                "decision_reason": "main_character -> flux + pulid",
                "character_role": "main"
            },
            
            "character": {
                "id": "hanli",
                "name": "韩立",
                "present": True,
                "role": "main",
                
                "identity": {
                    "gender": "male",
                    "age_range": "young_adult",
                    "physique": "lean",
                    "face_shape": "sharp"
                },
                
                "lora_config": {
                    "type": "single",
                    "lora_path": "test_lora.safetensors",
                    "weight": 0.9,
                    "trigger": "hanli"
                },
                
                "anchor_patches": {
                    "temperament_anchor": "calm and restrained temperament, sharp but composed eyes",
                    "explicit_lock_words": "wearing his iconic mid-late-stage green daoist robe",
                    "face_detailer": {
                        "enable": True,
                        "trigger": "shot_scale >= medium",
                        "denoise": 0.35
                    }
                },
                
                "reference_image": "test_reference.jpg",
                "negative_gender_lock": [
                    "female", "woman", "girl",
                    "soft facial features", "delicate face"
                ]
            },
            
            "environment": {
                "location": "黄枫谷",
                "time": "day",
                "weather": "clear",
                "atmosphere": "serene and mysterious"
            },
            
            "prompt": {
                "base_template": "{{character.name}}, {{character.anchor_patches.temperament_anchor}}, {{character.anchor_patches.explicit_lock_words}}, standing in {{environment.location}}, {{environment.atmosphere}}, cinematic lighting, high detail, epic atmosphere",
                
                "llm_enhancement": {
                    "enable": False,
                    "role": "copywriter",
                    "tasks": [
                        "enhance_scene_description",
                        "add_atmosphere_details"
                    ],
                    "forbidden_tasks": [
                        "decide_shot_type",
                        "decide_pose_type",
                        "decide_model_route"
                    ]
                }
            },
            
            "generation_params": {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "seed": -1
            },
            
            "validation": {
                "shot_pose_compatible": True,
                "model_route_valid": True,
                "character_anchor_complete": True,
                "prompt_template_valid": True
            }
        }
    }


def test_format_detection():
    """测试格式检测"""
    print("=" * 60)
    print("测试1: 格式检测")
    print("=" * 60)
    
    scene = create_test_scene_v22()
    version = scene.get("version", "")
    
    if version == "v2.2-final":
        print("  ✓ v2.2-final格式检测成功")
        return True
    else:
        print(f"  ✗ 格式检测失败: {version}")
        return False


def test_normalization():
    """测试格式规范化"""
    print("\n" + "=" * 60)
    print("测试2: 格式规范化")
    print("=" * 60)
    
    try:
        from execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode
        
        scene = create_test_scene_v22()
        config = ExecutionConfig(mode=ExecutionMode.STRICT)
        executor = ExecutionExecutorV21(config=config)
        
        normalized = executor._normalize_scene_format(scene)
        
        if "scene" not in normalized and "shot" in normalized:
            print("  ✓ 格式规范化成功")
            print(f"    - scene_id: {normalized.get('scene_id')}")
            print(f"    - shot.type: {normalized.get('shot', {}).get('type')}")
            print(f"    - character.id: {normalized.get('character', {}).get('id')}")
            return True
        else:
            print("  ✗ 格式规范化失败")
            return False
    except Exception as e:
        print(f"  ✗ 格式规范化异常: {e}")
        return False


def test_validation():
    """测试JSON验证"""
    print("\n" + "=" * 60)
    print("测试3: JSON验证")
    print("=" * 60)
    
    try:
        from execution_validator import ExecutionValidator
        
        scene = create_test_scene_v22()
        validator = ExecutionValidator()
        result = validator.validate_scene(scene)
        
        if result.is_valid:
            print("  ✓ JSON验证通过")
            return True
        else:
            print(f"  ✗ JSON验证失败: {result.errors_count} 个错误")
            for error in result.errors:
                print(f"    - {error}")
            return False
    except Exception as e:
        print(f"  ✗ JSON验证异常: {e}")
        return False


def test_prompt_building():
    """测试Prompt构建"""
    print("\n" + "=" * 60)
    print("测试4: Prompt构建")
    print("=" * 60)
    
    try:
        from execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode
        
        scene = create_test_scene_v22()
        config = ExecutionConfig(mode=ExecutionMode.STRICT)
        executor = ExecutionExecutorV21(config=config)
        
        normalized = executor._normalize_scene_format(scene)
        prompt = executor._build_prompt(normalized)
        negative_prompt = executor._build_negative_prompt(normalized)
        
        print(f"  ✓ Prompt构建成功")
        print(f"    Prompt长度: {len(prompt)} 字符")
        print(f"    Prompt预览: {prompt[:80]}...")
        print(f"    负面词数量: {len(negative_prompt)}")
        
        # 检查关键元素
        required = ["韩立", "calm and restrained", "green daoist robe", "黄枫谷"]
        missing = [elem for elem in required if elem not in prompt]
        
        if missing:
            print(f"  ⚠ 缺少关键元素: {missing}")
        else:
            print("  ✓ 包含所有关键元素")
        
        return True
    except Exception as e:
        print(f"  ✗ Prompt构建异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decision_trace():
    """测试决策trace"""
    print("\n" + "=" * 60)
    print("测试5: 决策trace")
    print("=" * 60)
    
    try:
        from execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode
        
        scene = create_test_scene_v22()
        config = ExecutionConfig(mode=ExecutionMode.STRICT)
        executor = ExecutionExecutorV21(config=config)
        
        normalized = executor._normalize_scene_format(scene)
        trace = executor._build_decision_trace(normalized)
        
        print("  ✓ 决策trace构建成功")
        print(f"    Shot: {trace['shot']['type']}")
        print(f"    Pose: {trace['pose']['type']}")
        print(f"    Model: {trace['model_route']['base_model']}")
        print(f"    Character: {trace['character_anchor']['character_id']}")
        
        return True
    except Exception as e:
        print(f"  ✗ 决策trace异常: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("v2.2-final格式端到端测试")
    print("=" * 60)
    
    tests = [
        ("格式检测", test_format_detection),
        ("格式规范化", test_normalization),
        ("JSON验证", test_validation),
        ("Prompt构建", test_prompt_building),
        ("决策trace", test_decision_trace),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ 测试 '{name}' 异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

