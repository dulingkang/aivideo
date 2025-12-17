#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试图像质量分析器
验证 ImageQualityAnalyzer 的各项功能
"""

import os
import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def test_analyzer_import():
    """测试导入"""
    print("\n" + "=" * 60)
    print("1. 测试模块导入")
    print("=" * 60)
    
    try:
        from utils.image_quality_analyzer import (
            ImageQualityAnalyzer,
            ImageQualityReport,
            FaceSimilarityResult,
            CompositionResult,
            TechnicalQualityResult,
            ShotType,
            QualityLevel,
            analyze_image
        )
        print("✅ 所有类和函数导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_synthetic_image():
    """测试合成图像分析"""
    print("\n" + "=" * 60)
    print("2. 测试合成图像分析")
    print("=" * 60)
    
    from utils.image_quality_analyzer import ImageQualityAnalyzer
    from PIL import Image
    import numpy as np
    
    # 创建一个合成测试图像 (渐变 + 噪点)
    width, height = 768, 1152
    
    # 创建渐变
    x = np.linspace(0, 255, width)
    y = np.linspace(0, 255, height)
    xv, yv = np.meshgrid(x, y)
    
    # RGB 渐变
    r = xv.astype(np.uint8)
    g = yv.astype(np.uint8)
    b = ((xv + yv) / 2).astype(np.uint8)
    
    # 组合成图像
    img_array = np.stack([r, g, b], axis=2)
    
    # 添加一些细节（提高清晰度）
    noise = np.random.randint(-20, 20, img_array.shape).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    test_image = Image.fromarray(img_array)
    
    # 分析
    analyzer = ImageQualityAnalyzer()
    try:
        report = analyzer.analyze(test_image)
        
        print(f"   图像尺寸: {report.image_size}")
        print(f"   综合评分: {report.overall_score:.1f}")
        print(f"   质量等级: {report.overall_level.value}")
        
        if report.technical:
            tech = report.technical
            print(f"   清晰度: {tech.sharpness:.1f} ({tech.sharpness_level.value})")
            print(f"   饱和度: {tech.saturation:.1f} ({tech.saturation_level.value})")
            print(f"   亮度: {tech.brightness:.1f} ({tech.brightness_level.value})")
            print(f"   对比度: {tech.contrast:.1f} ({tech.contrast_level.value})")
        
        print("✅ 合成图像分析成功")
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        analyzer.unload()


def test_real_image():
    """测试真实图像分析"""
    print("\n" + "=" * 60)
    print("3. 测试真实图像分析")
    print("=" * 60)
    
    from utils.image_quality_analyzer import ImageQualityAnalyzer
    
    # 查找测试图像
    test_dirs = [
        "outputs/reference_strength_tuning",
        "outputs/batch_test",
        "outputs/enhanced",
        "reference_image"
    ]
    
    test_image = None
    reference_image = None
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(test_dir, f)
                    break
        if test_image:
            break
    
    # 查找参考图像
    ref_paths = [
        "reference_image/hanli_mid.jpg",
        "reference_image/hanli.jpg",
        "character_references/hanli/front/neutral.jpg"
    ]
    for ref_path in ref_paths:
        if os.path.exists(ref_path):
            reference_image = ref_path
            break
    
    if not test_image:
        print("⚠️ 未找到测试图像，跳过真实图像测试")
        return True
    
    print(f"   测试图像: {test_image}")
    if reference_image:
        print(f"   参考图像: {reference_image}")
    
    # 分析
    analyzer = ImageQualityAnalyzer()
    try:
        report = analyzer.analyze(
            test_image,
            reference_image=reference_image,
            similarity_threshold=0.7
        )
        
        # 打印报告
        print(analyzer.format_report(report, verbose=True))
        
        # 保存 JSON
        json_path = Path(test_image).with_suffix('.quality.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        print(f"\n📁 JSON 报告已保存: {json_path}")
        
        print("✅ 真实图像分析成功")
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        analyzer.unload()


def test_batch_analysis():
    """测试批量分析"""
    print("\n" + "=" * 60)
    print("4. 测试批量分析")
    print("=" * 60)
    
    from utils.image_quality_analyzer import ImageQualityAnalyzer
    
    # 查找测试图像目录
    test_dirs = [
        "outputs/reference_strength_tuning",
        "outputs/batch_test"
    ]
    
    test_dir = None
    for td in test_dirs:
        if os.path.exists(td):
            test_dir = td
            break
    
    if not test_dir:
        print("⚠️ 未找到测试图像目录，跳过批量测试")
        return True
    
    # 查找所有图像
    images = []
    for f in os.listdir(test_dir):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(test_dir, f))
    
    if not images:
        print("⚠️ 测试目录中没有图像")
        return True
    
    print(f"   测试目录: {test_dir}")
    print(f"   图像数量: {len(images)}")
    
    # 批量分析
    analyzer = ImageQualityAnalyzer()
    results = []
    
    try:
        for img_path in images[:5]:  # 最多分析 5 张
            print(f"\n   分析: {Path(img_path).name}")
            report = analyzer.analyze(img_path)
            results.append({
                'file': Path(img_path).name,
                'score': report.overall_score,
                'level': report.overall_level.value,
                'sharpness': report.technical.sharpness if report.technical else 0,
                'saturation': report.technical.saturation if report.technical else 0
            })
            print(f"      评分: {report.overall_score:.1f} ({report.overall_level.value})")
        
        # 统计
        if results:
            avg_score = sum(r['score'] for r in results) / len(results)
            print(f"\n📊 批量分析统计:")
            print(f"   分析数量: {len(results)}")
            print(f"   平均评分: {avg_score:.1f}")
        
        print("✅ 批量分析成功")
        return True
        
    except Exception as e:
        print(f"❌ 批量分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        analyzer.unload()


def test_integration():
    """测试与 enhanced_image_generator 的集成"""
    print("\n" + "=" * 60)
    print("5. 测试集成（模拟）")
    print("=" * 60)
    
    try:
        # 模拟 enhanced_image_generator 中的质量验证
        from utils.image_quality_analyzer import ImageQualityAnalyzer, QualityLevel
        from PIL import Image
        import numpy as np
        
        # 创建模拟图像
        img_array = np.random.randint(50, 200, (1152, 768, 3), dtype=np.uint8)
        test_image = Image.fromarray(img_array)
        
        # 创建分析器
        analyzer = ImageQualityAnalyzer()
        
        # 分析
        report = analyzer.analyze(test_image)
        
        # 模拟日志输出
        print("   模拟质量验证日志:")
        print("   " + "=" * 50)
        print("   📊 图像质量分析结果")
        print("   " + "=" * 50)
        
        level_emoji = {
            QualityLevel.EXCELLENT: "🌟",
            QualityLevel.GOOD: "✅",
            QualityLevel.FAIR: "🟡",
            QualityLevel.POOR: "🟠",
            QualityLevel.BAD: "🔴"
        }
        emoji = level_emoji.get(report.overall_level, "❓")
        print(f"   🎯 综合评分: {report.overall_score:.1f}/100 {emoji}")
        
        if report.composition:
            print(f"   🎬 镜头类型: {report.composition.shot_type.value}")
        
        if report.technical:
            print(f"   📊 清晰度: {report.technical.sharpness:.1f}")
            print(f"      饱和度: {report.technical.saturation:.1f}")
        
        print("   " + "=" * 50)
        
        analyzer.unload()
        print("✅ 集成测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("图像质量分析器测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_analyzer_import),
        ("合成图像分析", test_synthetic_image),
        ("真实图像分析", test_real_image),
        ("批量分析", test_batch_analysis),
        ("集成测试", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ 测试 {name} 异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} - {name}")
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️ 部分测试失败，请检查日志")


if __name__ == "__main__":
    main()
