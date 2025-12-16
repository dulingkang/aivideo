#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版人脸一致性分析
使用 PIL 和基础图像特征进行比较
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict
import sys

def extract_face_region(img: Image.Image) -> np.ndarray:
    """
    提取人脸区域（假设人脸在中心区域）
    使用简单的中心裁剪来提取可能的人脸区域
    """
    width, height = img.size
    
    # 假设人脸在中心 40% 的区域
    crop_size = min(width, height) * 0.4
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    
    face_region = img.crop((left, top, right, bottom))
    
    # 缩放到固定尺寸以便比较
    face_region = face_region.resize((128, 128), Image.Resampling.LANCZOS)
    
    return np.array(face_region.convert('RGB'))

def calculate_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算两张图片的相似度
    使用多种方法：
    1. 直方图相似度
    2. 结构相似度（SSIM 简化版）
    3. 像素差异
    """
    # 方法1: 直方图相似度
    hist1 = np.histogram(img1.flatten(), bins=256, range=(0, 256))[0]
    hist2 = np.histogram(img2.flatten(), bins=256, range=(0, 256))[0]
    hist_sim = 1 - np.sum(np.abs(hist1 - hist2)) / (2 * img1.size)
    
    # 方法2: 平均像素差异
    pixel_diff = np.mean(np.abs(img1.astype(float) - img2.astype(float))) / 255.0
    pixel_sim = 1 - pixel_diff
    
    # 方法3: 颜色分布相似度
    color_sim = np.mean([
        np.corrcoef(img1[:, :, i].flatten(), img2[:, :, i].flatten())[0, 1]
        for i in range(3)
    ])
    color_sim = max(0, color_sim)  # 确保非负
    
    # 综合相似度
    similarity = (hist_sim * 0.3 + pixel_sim * 0.4 + color_sim * 0.3)
    
    return similarity

def analyze_images(image_dir: str, image_files: List[str] = None):
    """
    分析图片的人脸一致性
    """
    image_dir = Path(image_dir)
    
    if image_files is None:
        image_files = sorted([f.name for f in image_dir.glob("*.png")])
    
    print("=" * 70)
    print("🔍 生成图片人脸一致性分析")
    print("=" * 70)
    print(f"\n📁 图片目录: {image_dir}")
    print(f"📊 找到 {len(image_files)} 张图片\n")
    
    # 加载所有图片并提取人脸区域
    images_data = []
    for img_file in image_files:
        img_path = image_dir / img_file
        if not img_path.exists():
            continue
        
        try:
            img = Image.open(img_path)
            face_region = extract_face_region(img)
            
            images_data.append({
                'file': img_file,
                'image': img,
                'face_region': face_region,
                'size': img.size
            })
            
            print(f"✅ {img_file}: {img.size[0]}x{img.size[1]}")
            
        except Exception as e:
            print(f"❌ {img_file}: 加载失败 - {e}")
    
    if len(images_data) < 2:
        print("\n⚠️  至少需要 2 张图片才能进行相似度分析")
        return
    
    # 比较所有图片对
    print("\n" + "=" * 70)
    print("📊 人脸区域相似度分析")
    print("=" * 70)
    print("\n比较结果（相似度 0-1，1 表示完全相同）：\n")
    
    similarities = []
    for i in range(len(images_data)):
        for j in range(i + 1, len(images_data)):
            sim = calculate_similarity(
                images_data[i]['face_region'],
                images_data[j]['face_region']
            )
            
            similarities.append({
                'file1': images_data[i]['file'],
                'file2': images_data[j]['file'],
                'similarity': sim
            })
            
            sim_percent = sim * 100
            if sim_percent >= 70:
                status = "✅ 高度一致"
            elif sim_percent >= 50:
                status = "⚠️  部分一致"
            else:
                status = "❌ 不一致"
            
            print(f"{images_data[i]['file'][:30]:30} vs {images_data[j]['file'][:30]:30}")
            print(f"  相似度: {sim_percent:5.1f}% ({status})")
            print()
    
    # 统计结果
    if similarities:
        avg_sim = sum(s['similarity'] for s in similarities) / len(similarities) * 100
        max_sim = max(s['similarity'] for s in similarities) * 100
        min_sim = min(s['similarity'] for s in similarities) * 100
        
        print("=" * 70)
        print("📈 统计结果")
        print("=" * 70)
        print(f"  平均相似度: {avg_sim:.1f}%")
        print(f"  最高相似度: {max_sim:.1f}%")
        print(f"  最低相似度: {min_sim:.1f}%")
        print()
        
        # 结论
        if avg_sim >= 70:
            print("✅ 结论: 人脸一致性良好，LoRA 训练效果不错")
        elif avg_sim >= 50:
            print("⚠️  结论: 人脸一致性一般")
            print("   建议:")
            print("   - 增加训练数据（20 → 30-50 张）")
            print("   - 增加训练步数（2000 → 3000+ 步）")
            print("   - 调整 lora_alpha（当前 1.0，可尝试 1.2-1.5）")
        else:
            print("❌ 结论: 人脸一致性较差")
            print("   建议:")
            print("   - 检查训练数据质量（人脸清晰度、角度多样性）")
            print("   - 重新训练 LoRA，增加数据量和训练步数")
            print("   - 考虑使用 InstantID 进行人脸固定")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = "outputs/api/images"
    
    if len(sys.argv) > 2:
        image_files = sys.argv[2:]
    else:
        image_files = None
    
    analyze_images(image_dir, image_files)


