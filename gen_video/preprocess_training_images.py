#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练图片预处理脚本
统一所有图片为 1024x1024（1:1 正方形）
"""

from PIL import Image
from pathlib import Path
import argparse


def preprocess_images(
    input_dir: str,
    output_dir: str,
    target_size: tuple = (1024, 1024),
    background_color: tuple = (255, 255, 255),
    crop_bottom: int = 0,
    crop_right: int = 0
):
    """
    预处理训练图片：统一尺寸为指定大小
    
    Args:
        input_dir: 原始图片目录
        output_dir: 输出目录
        target_size: 目标尺寸 (width, height)，默认 (1024, 1024)
        background_color: 背景颜色 (R, G, B)，默认白色
        crop_bottom: 裁剪底部像素数（用于去除水印），默认 0
        crop_right: 裁剪右侧像素数（用于去除水印），默认 0
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}
    
    processed_count = 0
    skipped_count = 0
    
    print(f"📁 输入目录: {input_path}")
    print(f"📁 输出目录: {output_path}")
    print(f"📐 目标尺寸: {target_size[0]}x{target_size[1]}")
    print(f"🎨 背景颜色: {background_color}")
    if crop_bottom > 0 or crop_right > 0:
        print(f"✂️  裁剪水印: 底部 {crop_bottom}px, 右侧 {crop_right}px")
    print("-" * 60)
    
    for img_file in sorted(input_path.iterdir()):
        if img_file.suffix not in image_extensions:
            continue
        
        print(f"处理: {img_file.name}", end=" ... ")
        
        try:
            # 打开图片
            img = Image.open(img_file)
            
            # 转换为 RGB（如果是 RGBA 或其他模式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 获取原始尺寸
            orig_width, orig_height = img.size
            target_width, target_height = target_size
            
            # 裁剪水印区域（如果指定）
            if crop_bottom > 0 or crop_right > 0:
                # 计算裁剪区域：从左上角开始，裁剪掉右下角的水印
                crop_width = orig_width - crop_right if crop_right > 0 else orig_width
                crop_height = orig_height - crop_bottom if crop_bottom > 0 else orig_height
                
                # 确保裁剪尺寸有效
                crop_width = max(1, crop_width)
                crop_height = max(1, crop_height)
                
                # 裁剪图片（从左上角开始，保留主体部分）
                img = img.crop((0, 0, crop_width, crop_height))
                orig_width, orig_height = img.size
                
                if crop_bottom > 0 or crop_right > 0:
                    print(f"   已裁剪水印: {crop_width}x{crop_height}", end=" ... ")
            
            # 如果已经是目标尺寸，直接复制
            if orig_width == target_width and orig_height == target_height:
                output_file = output_path / img_file.name
                img.save(output_file, quality=95)
                print(f"✅ 已复制（已是目标尺寸）")
                processed_count += 1
                continue
            
            # 计算缩放比例（保持宽高比）
            scale = min(target_width / orig_width, target_height / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # 缩放图片（使用高质量重采样）
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 创建目标尺寸的背景
            img_final = Image.new('RGB', target_size, background_color)
            
            # 计算居中位置
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # 将缩放后的图片粘贴到中心
            img_final.paste(img_resized, (x_offset, y_offset))
            
            # 保存（保持原文件名）
            output_file = output_path / img_file.name
            img_final.save(output_file, quality=95, optimize=True)
            
            print(f"✅ 已处理 ({orig_width}x{orig_height} → {target_width}x{target_height})")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            skipped_count += 1
    
    print("-" * 60)
    print(f"✅ 预处理完成！")
    print(f"   成功处理: {processed_count} 张")
    if skipped_count > 0:
        print(f"   跳过/失败: {skipped_count} 张")
    print(f"   输出目录: {output_path}")
    print(f"   所有图片已统一为 {target_size[0]}x{target_size[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理训练图片：统一尺寸为 1024x1024")
    parser.add_argument(
        "--input",
        type=str,
        default="train_data/host_person_raw",
        help="原始图片目录（默认: train_data/host_person_raw）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_data/host_person",
        help="输出目录（默认: train_data/host_person）"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        metavar=("WIDTH", "HEIGHT"),
        help="目标尺寸（默认: 1024 1024）"
    )
    parser.add_argument(
        "--bg",
        type=int,
        nargs=3,
        default=[255, 255, 255],
        metavar=("R", "G", "B"),
        help="背景颜色 RGB（默认: 255 255 255 白色）"
    )
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=0,
        help="裁剪底部像素数（用于去除水印，如豆包水印，默认: 0）"
    )
    parser.add_argument(
        "--crop-right",
        type=int,
        default=0,
        help="裁剪右侧像素数（用于去除水印，如豆包水印，默认: 0）"
    )
    
    args = parser.parse_args()
    
    preprocess_images(
        input_dir=args.input,
        output_dir=args.output,
        target_size=tuple(args.size),
        background_color=tuple(args.bg),
        crop_bottom=args.crop_bottom,
        crop_right=args.crop_right
    )

