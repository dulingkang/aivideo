#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成测试脚本

测试多个场景下的生成稳定性和一致性
验证不同角度、不同表情的生成效果
检查批量生成时的显存管理
"""

import os
import sys
from pathlib import Path
from PIL import Image
import yaml
import time
import torch
import gc

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_image_generator import EnhancedImageGenerator

def test_batch_generation():
    """批量生成测试"""
    
    config_path = "config.yaml"
    
    # 参考图像
    ref_path = "/vepfs-dev/shawn/vid/fanren/gen_video/reference_image/hanli_mid.jpg"
    if not os.path.exists(ref_path):
        print(f"  ⚠️ 参考图像不存在: {ref_path}")
        return
    
    # 测试场景列表
    test_scenes = [
        {
            "name": "远景-仙山",
            "camera": {"shot": "wide", "angle": "eye_level"},
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
        },
        {
            "name": "中景-修炼",
            "camera": {"shot": "medium", "angle": "eye_level"},
            "character": {
                "present": True,
                "emotion": "calm",
                "description": "(Deep teal blue and light gray blue wide-sleeve traditional Chinese robe:1.5), (intricate gilded hollowed-out tangled branch patterns on shoulders and neckline:1.4), (flowing cloud dark patterns on robe fabric:1.3), (black cross-collar束腰 inner garment:1.2), a young Chinese male cultivator with (long black hair tied up with traditional hairpins:1.2)"
            },
            "environment": {
                "description": "meditation chamber with spiritual energy",
                "lighting": "soft warm lighting",
                "atmosphere": "peaceful and serene"
            }
        },
        {
            "name": "近景-战斗",
            "camera": {"shot": "close", "angle": "eye_level"},
            "character": {
                "present": True,
                "emotion": "determined",
                "action": "in combat stance, casting spell, energy gathering around hands, battle-ready",
                "description": "(Deep teal blue and light gray blue wide-sleeve traditional Chinese robe:1.5), (intricate gilded hollowed-out tangled branch patterns on shoulders and neckline:1.4), (flowing cloud dark patterns on robe fabric:1.3), (black cross-collar束腰 inner garment:1.2), a young Chinese male cultivator with (long black hair tied up with traditional hairpins:1.2)"
            },
            "environment": {
                "description": "battlefield with energy waves, magical combat, spell effects",
                "lighting": "dramatic lighting",
                "atmosphere": "intense and powerful"
            }
        },
        {
            "name": "远景-森林",
            "camera": {"shot": "wide", "angle": "eye_level"},
            "character": {
                "present": True,
                "emotion": "neutral",
                "description": "(Deep teal blue and light gray blue wide-sleeve traditional Chinese robe:1.5), (intricate gilded hollowed-out tangled branch patterns on shoulders and neckline:1.4), (flowing cloud dark patterns on robe fabric:1.3), (black cross-collar束腰 inner garment:1.2), a young Chinese male cultivator with (long black hair tied up with traditional hairpins:1.2)"
            },
            "environment": {
                "description": "ancient forest with towering trees and mystical fog",
                "lighting": "dappled sunlight",
                "atmosphere": "mysterious and ancient"
            }
        },
        {
            "name": "中景-对话",
            "camera": {"shot": "medium", "angle": "eye_level"},
            "character": {
                "present": True,
                "emotion": "serious",
                "description": "(Deep teal blue and light gray blue wide-sleeve traditional Chinese robe:1.5), (intricate gilded hollowed-out tangled branch patterns on shoulders and neckline:1.4), (flowing cloud dark patterns on robe fabric:1.3), (black cross-collar束腰 inner garment:1.2), a young Chinese male cultivator with (long black hair tied up with traditional hairpins:1.2)"
            },
            "environment": {
                "description": "traditional Chinese courtyard",
                "lighting": "natural daylight",
                "atmosphere": "formal and respectful"
            }
        }
    ]
    
    print("=" * 60)
    print("批量生成测试")
    print("=" * 60)
    print(f"测试场景数量: {len(test_scenes)}")
    print("=" * 60)
    
    output_dir = Path("outputs/batch_generation_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建生成器实例（复用，但每次生成后清理）
    generator = None
    results = []
    
    try:
        generator = EnhancedImageGenerator(config_path)
        
        for i, scene in enumerate(test_scenes, 1):
            print(f"\n[{i}/{len(test_scenes)}] 测试场景: {scene['name']}")
            print("-" * 60)
            
            start_time = time.time()
            
            try:
                # 检查显存状态
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"  生成前显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
                
                # 生成图像
                image = generator.generate_scene(
                    scene=scene,
                    face_reference=ref_path
                )
                
                if image:
                    # 保存结果
                    output_path = output_dir / f"{i:02d}_{scene['name']}.png"
                    image.save(output_path)
                    
                    elapsed = time.time() - start_time
                    
                    # 检查显存状态
                    if torch.cuda.is_available():
                        allocated_after = torch.cuda.memory_allocated() / 1024**3
                        reserved_after = torch.cuda.memory_reserved() / 1024**3
                        print(f"  生成后显存: 已分配={allocated_after:.2f}GB, 已保留={reserved_after:.2f}GB")
                    
                    results.append({
                        "name": scene['name'],
                        "status": "success",
                        "path": str(output_path),
                        "time": elapsed,
                        "memory_allocated": allocated_after if torch.cuda.is_available() else 0,
                        "memory_reserved": reserved_after if torch.cuda.is_available() else 0
                    })
                    
                    print(f"  ✅ 已保存: {output_path}")
                    print(f"  ⏱️  耗时: {elapsed:.1f}秒")
                else:
                    results.append({
                        "name": scene['name'],
                        "status": "failed",
                        "error": "生成返回 None"
                    })
                    print(f"  ❌ 生成失败: 返回 None")
                
                # 清理显存（每次生成后）
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
            except Exception as e:
                results.append({
                    "name": scene['name'],
                    "status": "error",
                    "error": str(e)
                })
                print(f"  ❌ 错误: {e}")
                import traceback
                traceback.print_exc()
                
                # 清理显存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
    finally:
        # 最终清理
        if generator is not None:
            generator.unload_all()
            del generator
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = len(results) - success_count
    
    print(f"\n总测试数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    
    if success_count > 0:
        avg_time = sum(r.get("time", 0) for r in results if r.get("status") == "success") / success_count
        print(f"平均耗时: {avg_time:.1f}秒")
        
        if torch.cuda.is_available():
            max_memory = max((r.get("memory_reserved", 0) for r in results if r.get("status") == "success"), default=0)
            print(f"峰值显存: {max_memory:.2f}GB")
    
    print(f"\n结果保存在: {output_dir}")
    print("=" * 60)
    
    # 保存结果到文件
    import json
    results_file = output_dir / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果详情已保存: {results_file}")


if __name__ == "__main__":
    test_batch_generation()

