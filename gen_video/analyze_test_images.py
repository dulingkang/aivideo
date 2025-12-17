#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析参考强度调优测试生成的图片
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from decoupled_fusion_engine import DecoupledFusionEngine

def analyze_image_quality(image_path: Path, reference_path: Path, engine: DecoupledFusionEngine):
    """分析单张图片的质量"""
    print(f"\n{'='*60}")
    print(f"分析: {image_path.name}")
    print(f"{'='*60}")
    
    # 加载图片
    try:
        gen_image = Image.open(image_path).convert('RGB')
        ref_image = Image.open(reference_path).convert('RGB')
    except Exception as e:
        print(f"  ❌ 加载图片失败: {e}")
        return None
    
    # 获取图片尺寸
    w, h = gen_image.size
    print(f"  📐 图片尺寸: {w}x{h}")
    
    # 计算人脸相似度
    print(f"  🔍 计算人脸相似度...")
    try:
        passed, similarity = engine.verify_face_similarity(
            generated_image=gen_image,
            reference_image=ref_image,
            threshold=0.7
        )
        
        status = "✅ 通过" if passed else "❌ 未通过"
        print(f"  📊 人脸相似度: {similarity:.3f} (阈值: 0.7) {status}")
        
        # 相似度等级
        if similarity >= 0.8:
            level = "🟢 优秀"
        elif similarity >= 0.7:
            level = "🟡 良好"
        elif similarity >= 0.5:
            level = "🟠 一般"
        else:
            level = "🔴 较差"
        print(f"  📈 相似度等级: {level}")
        
    except Exception as e:
        print(f"  ⚠️  相似度计算失败: {e}")
        similarity = None
        passed = False
    
    # 分析图片构图（简单方法：检测人物在图片中的位置和大小）
    print(f"  🎨 分析构图...")
    try:
        # 转换为numpy数组
        img_array = np.array(gen_image)
        
        # 简单的构图分析：计算非背景区域（假设背景较亮或较暗）
        # 这里使用一个简单的启发式方法
        gray = np.mean(img_array, axis=2)
        
        # 计算中心区域的平均亮度（用于判断人物位置）
        center_y, center_x = h // 2, w // 2
        center_region = gray[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
        center_brightness = np.mean(center_region)
        
        # 计算边缘区域的平均亮度
        edge_region = np.concatenate([
            gray[:h//8, :].flatten(),  # 上边缘
            gray[-h//8:, :].flatten(),  # 下边缘
            gray[:, :w//8].flatten(),  # 左边缘
            gray[:, -w//8:].flatten()  # 右边缘
        ])
        edge_brightness = np.mean(edge_region)
        
        # 计算对比度（中心与边缘的差异）
        contrast = abs(center_brightness - edge_brightness)
        
        print(f"  📍 中心区域亮度: {center_brightness:.1f}")
        print(f"  📍 边缘区域亮度: {edge_brightness:.1f}")
        print(f"  📍 对比度: {contrast:.1f}")
        
        # 简单的构图判断
        if contrast < 20:
            composition = "可能为远景（人物较小，对比度低）"
        elif contrast > 50:
            composition = "可能为近景（人物较大，对比度高）"
        else:
            composition = "可能为中景（中等对比度）"
        
        print(f"  🎬 构图判断: {composition}")
        
    except Exception as e:
        print(f"  ⚠️  构图分析失败: {e}")
        composition = "无法判断"
    
    # 分析图片质量指标
    print(f"  🎯 图片质量指标:")
    try:
        # 计算图片的清晰度（使用拉普拉斯算子）
        import cv2
        gray_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray_cv, cv2.CV_64F).var()
        
        if laplacian_var > 100:
            sharpness = "🟢 清晰"
        elif laplacian_var > 50:
            sharpness = "🟡 一般"
        else:
            sharpness = "🔴 模糊"
        
        print(f"    清晰度: {laplacian_var:.1f} {sharpness}")
        
        # 计算颜色饱和度
        saturation = np.std(img_array.astype(float))
        if saturation > 50:
            sat_level = "🟢 饱和"
        elif saturation > 30:
            sat_level = "🟡 适中"
        else:
            sat_level = "🔴 低饱和"
        
        print(f"    饱和度: {saturation:.1f} {sat_level}")
        
    except Exception as e:
        print(f"    ⚠️  质量指标计算失败: {e}")
    
    return {
        'file': image_path.name,
        'similarity': similarity,
        'passed': passed,
        'composition': composition,
        'size': (w, h)
    }


def main():
    """主函数"""
    print("=" * 60)
    print("参考强度调优测试图片分析")
    print("=" * 60)
    
    # 图片目录
    image_dir = Path("outputs/reference_strength_tuning")
    reference_path = Path("reference_image/hanli_mid.jpg")
    
    if not image_dir.exists():
        print(f"❌ 图片目录不存在: {image_dir}")
        return
    
    if not reference_path.exists():
        print(f"❌ 参考图片不存在: {reference_path}")
        return
    
    # 初始化融合引擎（用于人脸相似度计算）
    print("\n初始化分析引擎...")
    try:
        import yaml
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 构建引擎配置
        decoupled_config = config.get("decoupled_fusion", {})
        engine_config = {
            "device": "cuda",
            "model_dir": os.path.dirname(decoupled_config.get("sam2_path", "/vepfs-dev/shawn/vid/fanren/gen_video/models/sam2")),
        }
        
        engine = DecoupledFusionEngine(engine_config)
    except Exception as e:
        print(f"❌ 初始化引擎失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 查找所有测试图片
    test_images = sorted(image_dir.glob("shot_*.png"))
    
    if not test_images:
        print(f"❌ 未找到测试图片")
        return
    
    print(f"\n找到 {len(test_images)} 张测试图片")
    
    # 分析每张图片
    results = []
    for img_path in test_images:
        result = analyze_image_quality(img_path, reference_path, engine)
        if result:
            results.append(result)
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("📊 分析总结")
    print("=" * 60)
    
    if results:
        print(f"\n共分析 {len(results)} 张图片:\n")
        
        for result in results:
            similarity_str = f"{result['similarity']:.3f}" if result['similarity'] is not None else "N/A"
            status = "✅" if result.get('passed', False) else "❌"
            print(f"  {status} {result['file']}")
            print(f"     相似度: {similarity_str}")
            print(f"     构图: {result.get('composition', 'N/A')}")
            print()
        
        # 计算平均相似度
        valid_similarities = [r['similarity'] for r in results if r['similarity'] is not None]
        if valid_similarities:
            avg_similarity = np.mean(valid_similarities)
            print(f"📈 平均相似度: {avg_similarity:.3f}")
            
            # 通过率
            passed_count = sum(1 for r in results if r.get('passed', False))
            pass_rate = passed_count / len(results) * 100
            print(f"📊 通过率: {passed_count}/{len(results)} ({pass_rate:.1f}%)")
    
    # 清理
    try:
        engine.unload()
    except:
        pass
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

