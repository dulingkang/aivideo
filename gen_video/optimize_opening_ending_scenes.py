#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化开始和结束场景：
1. 设置共享图片路径，避免重复生成
2. 在图片上叠加标题文字
"""

import json
from pathlib import Path
import os

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    try:
        import cv2
        import numpy as np
        HAS_CV2 = True
        HAS_PIL = False
    except ImportError:
        HAS_PIL = False
        HAS_CV2 = False
        print("警告: 未安装PIL或OpenCV，无法创建带标题的图片")

# 共享图片路径
SHARED_OPENING_IMAGE = "gen_video/shared_images/opening_template.png"
SHARED_ENDING_IMAGE = "gen_video/shared_images/ending_template.png"

def create_shared_image_dirs():
    """创建共享图片目录"""
    shared_dir = Path("gen_video/shared_images")
    shared_dir.mkdir(parents=True, exist_ok=True)
    return shared_dir

def create_title_image_cv2(text: str, episode: int = None, width: int = 1024, height: int = 1024) -> Path:
    """使用OpenCV创建带标题的图片"""
    import cv2
    import numpy as np
    
    # 创建渐变背景
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 渐变效果
    for y in range(height):
        r = int(20 + (60 - 20) * y / height)
        g = int(15 + (40 - 15) * y / height)
        b = int(40 + (80 - 40) * y / height)
        img[y, :] = [b, g, r]  # BGR格式
    
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (100, 220, 255)  # BGR格式，金色
    thickness = 3
    
    if episode:
        title_text = f"第{episode}集"
        subtitle_text = text
    else:
        title_text = text
        subtitle_text = ""
    
    # 获取文字大小
    (text_width, text_height), baseline = cv2.getTextSize(title_text, font, font_scale, thickness)
    
    # 计算位置（居中）
    text_x = (width - text_width) // 2
    text_y = height // 2 - 50 if subtitle_text else height // 2
    
    # 绘制文字阴影
    cv2.putText(img, title_text, (text_x + 3, text_y + 3), font, font_scale, (0, 0, 0), thickness)
    # 绘制文字
    cv2.putText(img, title_text, (text_x, text_y), font, font_scale, color, thickness)
    
    if subtitle_text:
        # 副标题
        sub_scale = font_scale * 1.2
        (sub_width, sub_height), _ = cv2.getTextSize(subtitle_text, font, sub_scale, thickness)
        sub_x = (width - sub_width) // 2
        sub_y = height // 2 + 100
        
        cv2.putText(img, subtitle_text, (sub_x + 3, sub_y + 3), font, sub_scale, (0, 0, 0), thickness)
        cv2.putText(img, subtitle_text, (sub_x, sub_y), font, sub_scale, (255, 255, 255), thickness)
    
    # 保存图片
    shared_dir = create_shared_image_dirs()
    output_path = shared_dir / f"opening_ep{episode}.png" if episode else shared_dir / "opening_template.png"
    cv2.imwrite(str(output_path), img)
    print(f"创建标题图片: {output_path}")
    
    return output_path

def create_title_image(text: str, episode: int = None, width: int = 1024, height: int = 1024) -> Path:
    """
    创建带标题的图片（用于开始场景）
    
    Args:
        text: 标题文字
        episode: 集数（可选）
        width: 图片宽度
        height: 图片高度
    
    Returns:
        生成的图片路径
    """
    if HAS_CV2:
        return create_title_image_cv2(text, episode, width, height)
    
    if not HAS_PIL:
        print("警告: 无法创建带标题的图片，请安装PIL或OpenCV")
        shared_dir = create_shared_image_dirs()
        return shared_dir / "opening_template.png"
    
    # 使用PIL创建
    shared_dir = create_shared_image_dirs()
    
    # 创建渐变背景（仙侠风格）
    img = Image.new('RGB', (width, height), color=(20, 15, 40))  # 深蓝紫色背景
    
    # 绘制渐变效果
    draw = ImageDraw.Draw(img)
    for y in range(height):
        # 从深到浅的渐变
        r = int(20 + (60 - 20) * y / height)
        g = int(15 + (40 - 15) * y / height)
        b = int(40 + (80 - 40) * y / height)
        draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))
    
    # 尝试加载字体（如果系统有中文字体）
    try:
        # 尝试常见的中文字体路径
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size=60)
                    break
                except:
                    continue
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 绘制标题文字
    if episode:
        title_text = f"第{episode}集"
        subtitle_text = text
    else:
        title_text = text
        subtitle_text = ""
    
    # 计算文字位置（居中）
    if subtitle_text:
        # 主标题（集数）
        bbox = draw.textbbox((0, 0), title_text, font=font)
        title_width = bbox[2] - bbox[0]
        title_height = bbox[3] - bbox[1]
        
        # 副标题
        sub_bbox = draw.textbbox((0, 0), subtitle_text, font=font)
        sub_width = sub_bbox[2] - sub_bbox[0]
        sub_height = sub_bbox[3] - sub_bbox[1]
        
        # 绘制主标题（上方，较小）
        title_x = (width - title_width) // 2
        title_y = height // 2 - title_height - 30
        
        # 绘制副标题（下方，较大）
        sub_x = (width - sub_width) // 2
        sub_y = height // 2 + 30
        
        # 绘制文字阴影效果
        shadow_offset = 3
        draw.text((title_x + shadow_offset, title_y + shadow_offset), title_text, 
                 fill=(0, 0, 0, 180), font=font)
        draw.text((title_x, title_y), title_text, 
                 fill=(255, 220, 100), font=font)  # 金色
        
        # 副标题
        sub_font_size = int(font.size * 1.2)
        try:
            sub_font = ImageFont.truetype(font.path if hasattr(font, 'path') else font_paths[0], size=sub_font_size)
        except:
            sub_font = font
        
        draw.text((sub_x + shadow_offset, sub_y + shadow_offset), subtitle_text, 
                 fill=(0, 0, 0, 180), font=sub_font)
        draw.text((sub_x, sub_y), subtitle_text, 
                 fill=(255, 255, 255), font=sub_font)  # 白色
    else:
        bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        
        # 绘制文字阴影
        shadow_offset = 3
        draw.text((text_x + shadow_offset, text_y + shadow_offset), title_text, 
                 fill=(0, 0, 0, 180), font=font)
        draw.text((text_x, text_y), title_text, 
                 fill=(255, 220, 100), font=font)
    
    # 保存图片
    output_path = shared_dir / f"opening_ep{episode}.png" if episode else shared_dir / "opening_template.png"
    img.save(output_path)
    print(f"创建标题图片: {output_path}")
    
    return output_path

def optimize_opening_ending_scenes(json_path: Path, create_images: bool = False):
    """优化开始和结束场景的图片路径"""
    print(f"处理: {json_path.name}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    episode = data.get("episode", "")
    title = data.get("title", "")
    scenes = data.get("scenes", [])
    
    updated = False
    
    # 处理开始场景（id=0）
    if scenes and scenes[0].get("id") == 0:
        opening_scene = scenes[0]
        
        # 如果create_images为True，生成带标题的图片
        if create_images:
            title_image_path = create_title_image(title, episode)
            opening_scene["image_path"] = str(title_image_path)
            updated = True
            print(f"  为开始场景创建标题图片: {title_image_path}")
        else:
            # 设置共享图片路径
            if not opening_scene.get("image_path"):
                shared_path = Path(SHARED_OPENING_IMAGE)
                if shared_path.exists():
                    opening_scene["image_path"] = str(shared_path)
                    updated = True
                    print(f"  设置开始场景使用共享图片")
                else:
                    print(f"  警告: 共享图片不存在: {shared_path}")
    
    # 处理结束场景（id=999）
    if scenes and scenes[-1].get("id") == 999:
        ending_scene = scenes[-1]
        
        # 设置共享图片路径
        if not ending_scene.get("image_path"):
            shared_path = Path(SHARED_ENDING_IMAGE)
            if shared_path.exists():
                ending_scene["image_path"] = str(shared_path)
                updated = True
                print(f"  设置结束场景使用共享图片")
            else:
                # 如果没有共享图片，创建默认结束图片
                if create_images:
                    shared_dir = create_shared_image_dirs()
                    ending_image = create_title_image("下一集再见", episode=None, width=1024, height=1024)
                    ending_scene["image_path"] = str(ending_image)
                    updated = True
                    print(f"  创建结束场景图片: {ending_image}")
    
    if updated:
        data["scenes"] = scenes
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 已更新")
    else:
        print(f"  - 无需更新")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="优化开始和结束场景")
    parser.add_argument("--create-images", action="store_true", 
                       help="创建带标题的图片（否则只设置共享路径）")
    parser.add_argument("--episode", type=int, 
                       help="指定集数（仅用于创建图片）")
    args = parser.parse_args()
    
    base_dir = Path("lingjie/scenes")
    
    if args.episode:
        # 只处理指定集数
        json_path = base_dir / f"{args.episode}.json"
        if json_path.exists():
            optimize_opening_ending_scenes(json_path, args.create_images)
        else:
            print(f"文件不存在: {json_path}")
    else:
        # 处理所有集数
        for i in range(1, 12):
            json_path = base_dir / f"{i}.json"
            if json_path.exists():
                optimize_opening_ending_scenes(json_path, args.create_images)
            else:
                print(f"文件不存在: {json_path}")

if __name__ == "__main__":
    main()

