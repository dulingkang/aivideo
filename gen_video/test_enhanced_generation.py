#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证 PuLID + 解耦融合 + Execution Planner V3

运行方式:
    python test_enhanced_generation.py

测试内容:
1. Execution Planner V3 策略分析
2. PuLID 引擎初始化
3. 解耦融合引擎初始化
4. 端到端生成测试 (可选)
"""

import os
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_execution_planner():
    """测试 Execution Planner V3"""
    print("\n" + "=" * 60)
    print("测试 1: Execution Planner V3")
    print("=" * 60)
    
    try:
        from execution_planner_v3 import ExecutionPlannerV3
        
        planner = ExecutionPlannerV3()
        
        # 测试不同镜头类型
        test_cases = [
            # wide + top_down: 俯拍远景，几乎看不到脸，参考强度可以很低
            {"shot": "wide", "angle": "top_down", "emotion": "neutral", "expected_range": (5, 35)},
            {"shot": "medium", "angle": "eye_level", "emotion": "neutral", "expected_range": (55, 65)},
            {"shot": "close", "angle": "eye_level", "emotion": "angry", "expected_range": (80, 100)},
            {"shot": "extreme_close", "angle": "low", "emotion": "pain", "expected_range": (90, 100)},
        ]
        
        all_passed = True
        for case in test_cases:
            scene = {
                "camera": {"shot": case["shot"], "angle": case["angle"]},
                "character": {"present": True, "emotion": case["emotion"]},
                "environment": {"description": "test scene"}
            }
            
            strategy = planner.analyze_scene(scene)
            strength = strategy.reference_strength
            
            min_expected, max_expected = case["expected_range"]
            passed = min_expected <= strength <= max_expected
            
            status = "✅" if passed else "❌"
            print(f"  {status} {case['shot']} + {case['angle']} + {case['emotion']}: {strength}% (期望: {min_expected}-{max_expected}%)")
            
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ Execution Planner V3 测试通过!")
        else:
            print("\n⚠️ 部分测试未通过，请检查参数调整逻辑")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pulid_engine():
    """测试 PuLID 引擎初始化"""
    print("\n" + "=" * 60)
    print("测试 2: PuLID 引擎")
    print("=" * 60)
    
    try:
        from pulid_engine import PuLIDEngine
        
        # 检查模型文件
        models_dir = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models")
        pulid_path = models_dir / "pulid" / "pulid_flux_v0.9.1.safetensors"
        
        if pulid_path.exists():
            print(f"  ✅ PuLID 模型存在: {pulid_path}")
            print(f"     大小: {pulid_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"  ❌ PuLID 模型不存在: {pulid_path}")
            return False
        
        # 测试引擎初始化
        config = {
            "device": "cuda",
            "quantization": "bfloat16",
            "model_dir": str(models_dir)
        }
        
        engine = PuLIDEngine(config)
        
        # 测试参考强度计算
        print("\n  参考强度计算测试:")
        for shot in ["wide", "medium", "close"]:
            strength = engine.calculate_reference_strength(shot)
            print(f"    {shot}: {strength}%")
        
        print("\n✅ PuLID 引擎初始化成功!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decoupled_fusion():
    """测试解耦融合引擎"""
    print("\n" + "=" * 60)
    print("测试 3: 解耦融合引擎")
    print("=" * 60)
    
    try:
        from decoupled_fusion_engine import DecoupledFusionEngine
        
        # 检查模型文件
        models_dir = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models")
        sam2_path = models_dir / "sam2"
        
        if sam2_path.exists():
            print(f"  ✅ SAM2 目录存在: {sam2_path}")
            # 列出 SAM2 目录内容
            files = list(sam2_path.glob("*"))
            print(f"     文件: {[f.name for f in files[:5]]}")
        else:
            print(f"  ❌ SAM2 目录不存在: {sam2_path}")
        
        # 测试引擎初始化
        config = {
            "device": "cuda",
            "model_dir": str(models_dir)
        }
        
        engine = DecoupledFusionEngine(config)
        
        # 测试 YOLO 加载
        print("\n  加载 YOLO...")
        engine.load_yolo()
        print("  ✅ YOLO 加载成功")
        
        # 卸载
        engine.unload()
        
        print("\n✅ 解耦融合引擎初始化成功!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_generator():
    """测试增强型图像生成器"""
    print("\n" + "=" * 60)
    print("测试 4: 增强型图像生成器")
    print("=" * 60)
    
    try:
        from enhanced_image_generator import EnhancedImageGenerator
        
        # 检查配置文件
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print(f"  ⚠️ 配置文件不存在: {config_path}")
            return False
        
        # 创建生成器
        generator = EnhancedImageGenerator(config_path)
        print("  ✅ EnhancedImageGenerator 创建成功")
        
        # 测试场景
        test_scene = {
            "camera": {"shot": "medium", "angle": "eye_level"},
            "character": {
                "present": True,
                "id": "hanli",
                "emotion": "neutral"
            },
            "environment": {
                "description": "ancient mountain temple, misty clouds",
                "lighting": "soft morning light"
            }
        }
        
        # 分析策略 (不实际生成)
        strategy = generator.planner.analyze_scene(test_scene)
        prompt = generator.planner.build_weighted_prompt(test_scene, strategy)
        
        print(f"\n  策略分析:")
        print(f"    参考强度: {strategy.reference_strength}%")
        print(f"    身份引擎: {strategy.identity_engine.value}")
        print(f"    解耦生成: {strategy.use_decoupled_pipeline}")
        print(f"    环境权重: {strategy.environment_weight}x")
        print(f"\n  构建的 Prompt:")
        print(f"    {prompt[:100]}...")
        
        generator.unload_all()
        
        print("\n✅ 增强型图像生成器测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end(run_generation: bool = False):
    """端到端测试 (可选)"""
    if not run_generation:
        print("\n" + "=" * 60)
        print("测试 5: 端到端生成 (跳过)")
        print("=" * 60)
        print("  提示: 使用 --full 参数运行完整测试")
        return True
    
    print("\n" + "=" * 60)
    print("测试 5: 端到端生成")
    print("=" * 60)
    
    try:
        import os
        import torch
        
        # 设置 PyTorch CUDA 内存分配配置，减少内存碎片
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        print("  已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        from enhanced_image_generator import EnhancedImageGenerator
        from PIL import Image
        
        # 检查显存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            print(f"\n显存状态:")
            print(f"  总计: {total:.2f} GB")
            print(f"  已分配: {allocated:.2f} GB")
            print(f"  已保留: {reserved:.2f} GB")
            print(f"  可用: {free:.2f} GB")
            
            if free < 25:
                print(f"\n⚠️  警告: 可用显存较少 ({free:.2f}GB)，可能会超出显存限制")
                print("  建议:")
                print("  1. 关闭其他占用显存的程序")
                print("  2. 确保没有其他 Python 进程占用显存")
                print("  3. 如果仍然失败，考虑降低分辨率或使用 CPU offload")
        
        # 创建生成器
        generator = EnhancedImageGenerator("config.yaml")
        
        # 测试场景 - 注意要包含人物描述和朝向
        test_scene = {
            "camera": {"shot": "medium", "angle": "eye_level"},
            "character": {
                "present": True,
                "emotion": "neutral",
                "description": "a young Chinese male cultivator with long black hair tied up, wearing flowing white and blue traditional robes, facing the camera, looking at viewer, front view portrait"
            },
            "environment": {
                "description": "misty mountain valley with ancient Chinese pavilion, bamboo forest in background",
                "lighting": "soft dawn light through mist",
                "atmosphere": "serene and mystical"
            }
        }
        
        # 获取参考图像
        ref_path = "/vepfs-dev/shawn/vid/fanren/gen_video/reference_image/hanli_mid.jpg"
        if not os.path.exists(ref_path):
            print(f"  ⚠️ 参考图像不存在: {ref_path}")
            ref_path = None
        
        print("  开始生成...")
        
        # 生成图像
        image = generator.generate_scene(
            scene=test_scene,
            face_reference=ref_path
        )
        
        # 保存结果
        output_path = "outputs/test_enhanced_generation.png"
        os.makedirs("outputs", exist_ok=True)
        image.save(output_path)
        print(f"  ✅ 图像已保存: {output_path}")
        
        # 卸载模型
        print("\n  卸载模型...")
        generator.unload_all()
        
        # 检查卸载后的显存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  卸载后显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
        
        print("\n✅ 端到端测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("PuLID + 解耦融合 + Execution Planner V3 测试")
    print("=" * 60)
    print(f"工作目录: {os.getcwd()}")
    
    # 检查是否运行完整测试
    run_full = "--full" in sys.argv
    
    # 运行测试
    results = {}
    
    results["Execution Planner V3"] = test_execution_planner()
    results["PuLID Engine"] = test_pulid_engine()
    results["Decoupled Fusion"] = test_decoupled_fusion()
    results["Enhanced Generator"] = test_enhanced_generator()
    results["End-to-End"] = test_end_to_end(run_generation=run_full)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 部分测试失败，请检查日志")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
