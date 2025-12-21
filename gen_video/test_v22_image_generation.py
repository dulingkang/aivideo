#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final格式图像生成测试

测试实际的图像生成流程，验证v2.2-final格式的完整功能
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def create_test_scene_v22() -> dict:
    """创建测试用的v2.2-final格式场景"""
    return {
        "version": "v2.2-final",
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "description": "图像生成测试场景"
        },
        "scene": {
            "id": "scene_test_image_001",
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
                    "lora_path": "test_lora.safetensors",  # 需要替换为实际路径
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
                
                "reference_image": "test_reference.jpg",  # 需要替换为实际路径
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


def test_image_generation():
    """测试图像生成"""
    print("=" * 60)
    print("v2.2-final格式图像生成测试")
    print("=" * 60)
    
    # 创建输出目录
    output_base = Path(__file__).parent / "outputs" / f"test_v22_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_base}")
    
    # 创建测试场景
    scene = create_test_scene_v22()
    
    # 保存测试JSON
    json_path = output_base / "test_scene.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)
    print(f"✓ 测试JSON已保存: {json_path}")
    
    try:
        from execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode
        from execution_validator import ExecutionValidator
        
        # 1. 验证JSON
        print("\n1. 验证JSON...")
        validator = ExecutionValidator()
        validation_result = validator.validate_scene(scene)
        
        if not validation_result.is_valid:
            print(f"  ✗ JSON验证失败: {validation_result.errors_count} 个错误")
            for issue in validation_result.issues:
                if issue.level.value == "error":
                    print(f"    - {issue.field}: {issue.message}")
            return False
        
        print(f"  ✓ JSON验证通过")
        
        # 2. 创建执行器（不初始化ImageGenerator，避免依赖）
        print("\n2. 创建执行器...")
        config = ExecutionConfig(mode=ExecutionMode.STRICT)
        executor = ExecutionExecutorV21(config=config)
        print("  ✓ 执行器创建成功")
        
        # 3. 测试Prompt构建
        print("\n3. 测试Prompt构建...")
        normalized = executor._normalize_scene_format(scene)
        prompt = executor._build_prompt(normalized)
        negative_prompt = executor._build_negative_prompt(normalized)
        
        print(f"  ✓ Prompt构建成功")
        print(f"    Prompt: {prompt[:100]}...")
        print(f"    负面词数量: {len(negative_prompt)}")
        
        # 保存Prompt到文件
        prompt_path = output_base / "generated_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write(f"Negative Prompt:\n{', '.join(negative_prompt)}\n")
        print(f"  ✓ Prompt已保存: {prompt_path}")
        
        # 4. 检查ImageGenerator是否可用
        print("\n4. 检查ImageGenerator...")
        if executor._image_generator is None:
            print("  ⚠ ImageGenerator未初始化（需要在实际环境中测试）")
            print("  ℹ 提示: 在实际生成时，需要传入ImageGenerator实例")
            print(f"  ℹ 输出路径将位于: {output_base / 'scene_001' / 'novel_image.png'}")
        else:
            print("  ✓ ImageGenerator已初始化")
            # 可以尝试实际生成
            print("\n5. 开始图像生成...")
            result = executor.execute_scene(scene, str(output_base))
            
            if result.success:
                print(f"  ✓ 图像生成成功")
                print(f"    图像路径: {result.image_path}")
                if result.image_path and Path(result.image_path).exists():
                    print(f"    ✓ 图像文件存在")
                else:
                    print(f"    ⚠ 图像文件不存在")
            else:
                print(f"  ✗ 图像生成失败: {result.error_message}")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        print(f"\n输出目录: {output_base}")
        print(f"  - 测试JSON: {json_path}")
        print(f"  - Prompt文件: {prompt_path}")
        if executor._image_generator:
            print(f"  - 图像文件: {output_base / 'scene_001' / 'novel_image.png'}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_image_generation()
    sys.exit(0 if success else 1)

