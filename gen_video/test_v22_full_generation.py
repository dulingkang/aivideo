#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final格式完整生成测试

测试实际的图像生成流程，需要ImageGenerator实例
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
            "description": "完整生成测试场景"
        },
        "scene": {
            "id": "scene_test_full_001",
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
                    "lora_path": "",  # 需要实际路径
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
                
                "reference_image": "",  # 需要实际路径
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
                "base_template": "{{character.name}}, {{character.anchor_patches.temperament_anchor}}, {{character.anchor_patches.explicit_lock_words}}, standing in {{environment.location}}, {{environment.atmosphere}}, cinematic lighting, high detail, epic atmosphere"
            },
            
            "generation_params": {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "seed": -1
            }
        }
    }


def test_full_generation():
    """测试完整生成流程"""
    print("=" * 60)
    print("v2.2-final格式完整生成测试")
    print("=" * 60)
    
    # 创建输出目录
    output_base = Path(__file__).parent / "outputs" / f"test_v22_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_base}")
    print(f"生成的图片将保存在: {output_base / 'scene_001' / 'novel_image.png'}")
    
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
        
        # 2. 尝试初始化ImageGenerator
        print("\n2. 初始化ImageGenerator...")
        image_generator = None
        
        try:
            # 尝试从generate_novel_video导入
            from generate_novel_video import NovelVideoGenerator
            import yaml
            
            # 查找config文件
            config_path = Path(__file__).parent.parent / "config.yaml"
            if not config_path.exists():
                config_path = Path(__file__).parent / "config.yaml"
            
            if config_path.exists():
                print(f"  ℹ 找到配置文件: {config_path}")
                generator = NovelVideoGenerator(str(config_path))
                image_generator = generator.image_generator
                print("  ✓ ImageGenerator初始化成功")
            else:
                print(f"  ⚠ 未找到配置文件: {config_path}")
                print("  ℹ 将跳过实际图像生成")
        except Exception as e:
            print(f"  ⚠ ImageGenerator初始化失败: {e}")
            print("  ℹ 将跳过实际图像生成")
        
        # 3. 创建执行器
        print("\n3. 创建执行器...")
        config = ExecutionConfig(mode=ExecutionMode.STRICT)
        executor = ExecutionExecutorV21(
            config=config,
            image_generator=image_generator,
            video_generator=None,
            tts_generator=None
        )
        print("  ✓ 执行器创建成功")
        
        # 4. 测试Prompt构建
        print("\n4. 测试Prompt构建...")
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
        
        # 5. 如果ImageGenerator可用，尝试实际生成
        if image_generator:
            print("\n5. 开始图像生成...")
            try:
                result = executor.execute_scene(scene, str(output_base))
                
                if result.success:
                    print(f"  ✓ 图像生成成功")
                    print(f"    图像路径: {result.image_path}")
                    if result.image_path and Path(result.image_path).exists():
                        print(f"    ✓ 图像文件存在")
                        print(f"    文件大小: {Path(result.image_path).stat().st_size / 1024:.2f} KB")
                    else:
                        print(f"    ⚠ 图像文件不存在")
                else:
                    print(f"  ✗ 图像生成失败: {result.error_message}")
            except Exception as e:
                print(f"  ✗ 图像生成异常: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n5. 跳过图像生成（ImageGenerator未初始化）")
            print(f"  ℹ 输出路径将位于: {output_base / 'scene_001' / 'novel_image.png'}")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        print(f"\n输出目录: {output_base}")
        print(f"  - 测试JSON: {json_path}")
        print(f"  - Prompt文件: {prompt_path}")
        if image_generator:
            print(f"  - 图像文件: {output_base / 'scene_001' / 'novel_image.png'}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_generation()
    sys.exit(0 if success else 1)

