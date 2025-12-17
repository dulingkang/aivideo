#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考强度调优测试脚本

测试不同参考强度下的生成效果，找到最佳平衡点
"""

import os
import sys
from pathlib import Path
from PIL import Image
import yaml

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_image_generator import EnhancedImageGenerator

def test_reference_strength_range():
    """测试不同镜头类型下的参考强度（使用 Execution Planner 自动计算）"""
    
    config_path = "config.yaml"
    
    # 测试场景
    test_scene = {
        "camera": {"shot": "medium", "angle": "eye_level"},
        "character": {
            "present": True,
            "emotion": "neutral",
            "description": "(Deep teal blue and light gray blue wide-sleeve traditional Chinese robe:1.5), (intricate gilded hollowed-out tangled branch patterns on shoulders and neckline:1.4), (flowing cloud dark patterns on robe fabric:1.3), (black cross-collar束腰 inner garment:1.2), a young Chinese male cultivator with (long black hair tied up with traditional hairpins:1.2)"
        },
        "environment": {
            "description": "fairy mountain landscape with floating palaces and clouds",
            "lighting": "cinematic lighting",
            "atmosphere": "misty ethereal atmosphere"
        }
    }
    
    # 参考图像
    ref_path = "/vepfs-dev/shawn/vid/fanren/gen_video/reference_image/hanli_mid.jpg"
    if not os.path.exists(ref_path):
        print(f"  ⚠️ 参考图像不存在: {ref_path}")
        return
    
    # 测试不同的镜头类型（Execution Planner 会自动计算参考强度）
    # 预期：wide=50%, medium=60%, close=75%
    shot_types = [
        ("wide", "远景"),
        ("medium", "中景"),
        ("close", "近景")
    ]
    
    print("=" * 60)
    print("参考强度调优测试（基于镜头类型）")
    print("=" * 60)
    print("预期参考强度：远景=50%, 中景=60%, 近景=75%")
    print("=" * 60)
    
    output_dir = Path("outputs/reference_strength_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import torch
    import gc
    
    for i, (shot_type, shot_name) in enumerate(shot_types):
        print(f"\n测试镜头类型: {shot_name} ({shot_type}) ({i+1}/{len(shot_types)})")
        print("-" * 60)
        
        # 每次测试都重新创建生成器，确保完全干净的显存状态
        generator = None
        try:
            # 在创建新生成器前，先清理显存
            if i > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            
            # 创建新的生成器实例
            generator = EnhancedImageGenerator(config_path)
            
            # 设置镜头类型
            test_scene["camera"]["shot"] = shot_type
            
            # 先获取策略和 prompt，打印出来用于调试
            strategy = generator.planner.analyze_scene(
                scene=test_scene,
                character_profiles=generator.character_profiles
            )
            prompt = generator.planner.build_weighted_prompt(test_scene, strategy)
            print(f"\n  📝 完整 Prompt:")
            print(f"  {prompt}")
            print(f"\n  📊 策略信息:")
            print(f"    镜头类型: {shot_type}")
            print(f"    参考强度: {strategy.reference_strength}%")
            print(f"    生成模式: {strategy.mode.value}")
            print(f"    解耦生成: {strategy.use_decoupled_pipeline}")
            
            # 生成图像（Execution Planner 会自动计算参考强度）
            image = generator.generate_scene(
                scene=test_scene,
                face_reference=ref_path
            )
            
            if image:
                # 获取实际使用的参考强度（从 Execution Planner）
                from execution_planner_v3 import ExecutionPlannerV3
                planner = ExecutionPlannerV3(config_path)
                actual_strength = planner.get_reference_strength_for_scene(test_scene)
                
                # 保存结果
                output_path = output_dir / f"shot_{shot_type}_strength_{actual_strength:02d}.png"
                image.save(output_path)
                print(f"  ✅ 已保存: {output_path}")
                print(f"  📊 实际参考强度: {actual_strength}%")
            else:
                print(f"  ❌ 生成失败")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 彻底清理显存
            if generator is not None:
                print(f"  清理显存...")
                generator.unload_all()
                del generator
            
            # 额外清理：强制同步和清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()  # 再次清理
                
                # 检查显存状态
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  清理后显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
                
                # 如果显存仍然很高，警告
                if reserved > 5.0:
                    print(f"  ⚠️  警告: 显存仍然较高 ({reserved:.2f}GB)")
                    if reserved > 20.0:
                        print(f"  ⚠️  严重警告: 显存过高，建议重启进程后再继续")
                        break
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


def test_clothing_consistency():
    """测试服饰一致性增强效果"""
    
    generator = EnhancedImageGenerator("config.yaml")
    
    test_scene = {
        "camera": {"shot": "medium", "angle": "eye_level"},
        "character": {
            "present": True,
            "emotion": "neutral",
            "description": "(Deep teal blue and light gray blue wide-sleeve traditional Chinese robe:1.5), (intricate gilded hollowed-out tangled branch patterns on shoulders and neckline:1.4), (flowing cloud dark patterns on robe fabric:1.3), (black cross-collar束腰 inner garment:1.2), a young Chinese male cultivator with (long black hair tied up with traditional hairpins:1.2)"
        },
        "environment": {
            "description": "fairy mountain landscape with floating palaces and clouds",
            "lighting": "cinematic lighting",
            "atmosphere": "misty ethereal atmosphere"
        }
    }
    
    ref_path = "/vepfs-dev/shawn/vid/fanren/gen_video/reference_image/hanli_mid.jpg"
    
    print("=" * 60)
    print("服饰一致性增强测试")
    print("=" * 60)
    
    output_dir = Path("outputs/clothing_consistency_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试启用和禁用服饰增强
    for enhance_clothing in [False, True]:
        mode = "enhanced" if enhance_clothing else "normal"
        print(f"\n测试模式: {mode}")
        print("-" * 60)
        
        try:
            image = generator.generate_scene(
                scene=test_scene,
                face_reference=ref_path,
                enhance_clothing_consistency=enhance_clothing
            )
            
            if image:
                output_path = output_dir / f"clothing_{mode}.png"
                image.save(output_path)
                print(f"  ✅ 已保存: {output_path}")
            else:
                print(f"  ❌ 生成失败")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        
        generator.unload_all()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="参考强度调优测试")
    parser.add_argument("--strength", action="store_true", help="测试不同参考强度")
    parser.add_argument("--clothing", action="store_true", help="测试服饰一致性增强")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    
    args = parser.parse_args()
    
    if args.all or args.strength:
        test_reference_strength_range()
    
    if args.all or args.clothing:
        test_clothing_consistency()
    
    if not (args.all or args.strength or args.clothing):
        print("请指定测试类型：")
        print("  --strength: 测试不同参考强度")
        print("  --clothing: 测试服饰一致性增强")
        print("  --all: 运行所有测试")

