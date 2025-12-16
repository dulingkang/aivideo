#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析生成图片的人脸一致性
使用 face_recognition 或 PIL 进行基础分析
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple
import sys

def analyze_images(image_dir: str, image_files: List[str] = None):
    """
    分析图片的人脸一致性
    
    Args:
        image_dir: 图片目录
        image_files: 要分析的图片文件列表（如果为 None，分析所有图片）
    """
    image_dir = Path(image_dir)
    
    if image_files is None:
        # 分析所有 PNG 图片
        image_files = sorted([f.name for f in image_dir.glob("*.png")])
    
    print("=" * 60)
    print("🔍 分析生成图片的人脸一致性")
    print("=" * 60)
    print(f"\n📁 图片目录: {image_dir}")
    print(f"📊 找到 {len(image_files)} 张图片\n")
    
    # 基础分析：图片尺寸、文件大小
    results = []
    for img_file in image_files:
        img_path = image_dir / img_file
        if not img_path.exists():
            continue
        
        try:
            img = Image.open(img_path)
            file_size = img_path.stat().st_size / 1024  # KB
            
            results.append({
                'file': img_file,
                'size': img.size,
                'mode': img.mode,
                'file_size_kb': file_size,
                'aspect_ratio': img.size[0] / img.size[1] if img.size[1] > 0 else 0
            })
            
            print(f"📷 {img_file}")
            print(f"   尺寸: {img.size[0]}x{img.size[1]}")
            print(f"   文件大小: {file_size:.1f} KB")
            print(f"   宽高比: {img.size[0]/img.size[1]:.2f}")
            print()
            
        except Exception as e:
            print(f"❌ 无法读取 {img_file}: {e}")
    
    # 尝试使用 face_recognition 进行人脸检测（如果可用）
    try:
        import face_recognition
        print("\n" + "=" * 60)
        print("🔍 使用 face_recognition 进行人脸分析")
        print("=" * 60)
        
        face_encodings = []
        for img_file in image_files:
            img_path = image_dir / img_file
            if not img_path.exists():
                continue
            
            try:
                # 加载图片
                image = face_recognition.load_image_file(str(img_path))
                
                # 检测人脸位置
                face_locations = face_recognition.face_locations(image)
                
                if len(face_locations) == 0:
                    print(f"⚠️  {img_file}: 未检测到人脸")
                elif len(face_locations) > 1:
                    print(f"⚠️  {img_file}: 检测到 {len(face_locations)} 个人脸（应该只有 1 个）")
                else:
                    # 提取人脸编码
                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                    face_encodings.append({
                        'file': img_file,
                        'encoding': face_encoding,
                        'location': face_locations[0]
                    })
                    print(f"✅ {img_file}: 检测到 1 个人脸")
                    
            except Exception as e:
                print(f"❌ {img_file}: 人脸检测失败 - {e}")
        
        # 比较人脸相似度
        if len(face_encodings) >= 2:
            print("\n" + "=" * 60)
            print("📊 人脸相似度分析")
            print("=" * 60)
            
            similarities = []
            for i in range(len(face_encodings)):
                for j in range(i + 1, len(face_encodings)):
                    encoding1 = face_encodings[i]['encoding']
                    encoding2 = face_encodings[j]['encoding']
                    
                    # 计算欧氏距离（越小越相似）
                    distance = face_recognition.face_distance([encoding1], encoding2)[0]
                    similarity = 1 - distance  # 转换为相似度（0-1，1 表示完全相同）
                    
                    similarities.append({
                        'file1': face_encodings[i]['file'],
                        'file2': face_encodings[j]['file'],
                        'distance': distance,
                        'similarity': similarity
                    })
            
            # 显示相似度结果
            print(f"\n共比较 {len(similarities)} 对图片：\n")
            for sim in similarities:
                similarity_percent = sim['similarity'] * 100
                if similarity_percent >= 70:
                    status = "✅ 高度一致"
                elif similarity_percent >= 50:
                    status = "⚠️  部分一致"
                else:
                    status = "❌ 不一致"
                
                print(f"{sim['file1']} vs {sim['file2']}")
                print(f"  相似度: {similarity_percent:.1f}% ({status})")
                print(f"  距离: {sim['distance']:.4f}")
                print()
            
            # 计算平均相似度
            avg_similarity = sum(s['similarity'] for s in similarities) / len(similarities) * 100
            print(f"📊 平均相似度: {avg_similarity:.1f}%")
            
            if avg_similarity >= 70:
                print("✅ 结论: 人脸一致性良好")
            elif avg_similarity >= 50:
                print("⚠️  结论: 人脸一致性一般，可能需要更多训练数据或调整训练参数")
            else:
                print("❌ 结论: 人脸一致性较差，建议重新训练或增加训练数据")
        
    except ImportError:
        print("\n⚠️  face_recognition 库未安装，跳过人脸相似度分析")
        print("   安装方法: pip install face_recognition")
        print("\n💡 基础分析完成，但无法进行人脸相似度比较")
    except Exception as e:
        print(f"\n⚠️  人脸分析出错: {e}")
        print("   基础分析完成，但无法进行人脸相似度比较")


if __name__ == "__main__":
    # 默认分析 outputs/api/images 目录
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = "outputs/api/images"
    
    # 如果指定了特定图片
    if len(sys.argv) > 2:
        image_files = sys.argv[2:]
    else:
        image_files = None
    
    analyze_images(image_dir, image_files)

