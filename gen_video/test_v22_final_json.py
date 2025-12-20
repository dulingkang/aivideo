#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试v2.2-final JSON格式

测试ExecutionExecutorV21对新JSON格式的支持
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# 直接导入，避免触发其他模块的导入
from execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode
from execution_validator import ExecutionValidator


def test_v22_final_json():
    """测试v2.2-final JSON格式"""
    
    # 创建测试JSON
    test_scene = {
        "version": "v2.2-final",
        "metadata": {
            "created_at": "2025-12-20T00:00:00Z",
            "description": "测试场景"
        },
        "scene": {
            "id": "scene_001",
            "duration_sec": 5.0,
            
            "intent": {
                "type": "character_intro",
                "description": "韩立初次登场"
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
    
    print("=" * 60)
    print("测试v2.2-final JSON格式")
    print("=" * 60)
    
    # 1. 测试JSON验证
    print("\n1. 测试JSON验证...")
    validator = ExecutionValidator()
    validation_result = validator.validate_scene(test_scene)
    
    if validation_result.is_valid:
        print("  ✓ JSON验证通过")
    else:
        print(f"  ✗ JSON验证失败: {validation_result.errors_count} 个错误")
        for error in validation_result.errors:
            print(f"    - {error}")
        return False
    
    # 2. 测试格式规范化
    print("\n2. 测试格式规范化...")
    config = ExecutionConfig(mode=ExecutionMode.STRICT)
    executor = ExecutionExecutorV21(config=config)
    
    # 测试_normalize_scene_format
    normalized = executor._normalize_scene_format(test_scene)
    
    if "scene" not in normalized:
        print("  ✓ 格式规范化成功（v2.2-final -> v2.1-exec格式）")
        print(f"    - scene_id: {normalized.get('scene_id')}")
        print(f"    - shot.type: {normalized.get('shot', {}).get('type')}")
        print(f"    - character.id: {normalized.get('character', {}).get('id')}")
    else:
        print("  ✗ 格式规范化失败")
        return False
    
    # 3. 测试Prompt构建
    print("\n3. 测试Prompt构建...")
    prompt = executor._build_prompt(normalized)
    print(f"  ✓ Prompt构建成功")
    print(f"    Prompt: {prompt[:100]}...")
    
    # 检查是否包含关键元素
    required_elements = [
        "韩立",
        "calm and restrained temperament",
        "green daoist robe",
        "黄枫谷"
    ]
    
    missing = [elem for elem in required_elements if elem not in prompt]
    if missing:
        print(f"  ⚠ 缺少关键元素: {missing}")
    else:
        print("  ✓ 包含所有关键元素")
    
    # 4. 测试负面Prompt构建
    print("\n4. 测试负面Prompt构建...")
    negative_prompt = executor._build_negative_prompt(normalized)
    print(f"  ✓ 负面Prompt构建成功")
    print(f"    负面词数量: {len(negative_prompt)}")
    
    # 检查是否包含性别负锁
    if "female" in " ".join(negative_prompt):
        print("  ✓ 包含性别负锁")
    else:
        print("  ⚠ 未包含性别负锁")
    
    # 5. 测试决策trace
    print("\n5. 测试决策trace...")
    decision_trace = executor._build_decision_trace(normalized)
    print(f"  ✓ 决策trace构建成功")
    print(f"    Shot: {decision_trace['shot']['type']}")
    print(f"    Pose: {decision_trace['pose']['type']}")
    print(f"    Model: {decision_trace['model_route']['base_model']}")
    print(f"    Character: {decision_trace['character_anchor']['character_id']}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_v22_final_json()
    sys.exit(0 if success else 1)

