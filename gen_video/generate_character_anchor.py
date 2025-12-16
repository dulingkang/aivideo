#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建角色人设锚点图（Scene 0）

这是整个视频的"DNA"，所有后续场景都必须引用这张图。

⚡ 核心规则：直接复制参考图，不生成
- hanli_anchor.png = hanli_mid.jpg（直接复制）
- 这是工业界最常见做法，确保 100% 相似度
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional
from image_generator import ImageGenerator


def create_character_anchor(
    generator: ImageGenerator,
    character_id: str,
    output_dir: Path = None,
) -> Optional[Path]:
    """
    创建角色人设锚点图（直接复制参考图，不生成）
    
    Args:
        generator: 图像生成器实例（用于获取配置）
        character_id: 角色ID（如 "hanli"）
        output_dir: 输出目录
    
    Returns:
        创建的人设锚点图路径
    """
    print(f"\n{'='*60}")
    print(f"创建角色人设锚点图: {character_id}")
    print(f"{'='*60}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = Path("gen_video/character_anchors")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{character_id}_anchor.png"
    
    # 如果已存在，询问是否覆盖
    if output_path.exists():
        print(f"  ⚠ 人设锚点图已存在: {output_path}")
        print(f"  ℹ 跳过创建（如需重新创建，请先删除该文件）")
        return output_path
    
    # 构建标准人设锚点 prompt
    # ⚡ 关键修复：简化 prompt，减少对 InstantID 人脸相似度的干扰
    # 特征：中景、平视、站立、微侧、自然姿态、凡人修仙传风格
    # 注意：prompt 越简单，InstantID 的人脸相似度越高
    prompt = (
        "medium shot, eye level, front view, "
        "young male cultivator, "
        "standing naturally, slight side angle, "
        "simple dark green cultivator robe, "
        "long black hair, calm expression, "
        "Chinese xianxia anime style"
    )
    
    negative_prompt = (
        "low quality, blurry, distorted, deformed, bad anatomy, "
        "multiple people, crowd, group, "
        "exaggerated expression, overacting, "
        "western style, european features"
    )
    
    print(f"  📝 Prompt: {prompt}")
    print(f"  📝 Negative Prompt: {negative_prompt}")
    
    # ⚡ 核心规则：直接复制参考图，不生成
    # 优先级：配置中的 face_image_path > hanli_mid.jpg > hanli_mid.png
    reference_path = None
    if character_id == "hanli":
        # 优先级 1：配置中的 face_image_path
        face_image_path = generator.image_config.get("face_image_path")
        if face_image_path and Path(face_image_path).exists():
            reference_path = Path(face_image_path)
            print(f"  ✓ 使用配置中的参考图: {reference_path.name}")
        else:
            # 优先级 2：hanli_mid.jpg 或 hanli_mid.png
            default_path_jpg = Path("gen_video/reference_image/hanli_mid.jpg")
            if default_path_jpg.exists():
                reference_path = default_path_jpg
                print(f"  ✓ 使用参考图: {reference_path.name} (.jpg)")
            elif default_path_png.exists():
                reference_path = default_path_png
                print(f"  ✓ 使用参考图: {reference_path.name} (.png)")
            else:
                print(f"  ❌ 错误：未找到参考图")
                print(f"  ℹ 请确保以下文件之一存在：")
                print(f"     - {face_image_path if face_image_path else '配置中的 face_image_path'}")
                print(f"     - gen_video/reference_image/hanli_mid.jpg")
                print(f"     - gen_video/reference_image/hanli_mid.png")
                return None
    
    if not reference_path or not reference_path.exists():
        print(f"  ❌ 错误：参考图不存在: {reference_path}")
        return None
    
    try:
        # ⚡ 核心规则：直接复制参考图，不生成
        print(f"  🎯 直接复制参考图作为人设锚点图（工业界标准做法）...")
        print(f"     源文件: {reference_path}")
        print(f"     目标文件: {output_path}")
        
        # 复制文件
        shutil.copy2(reference_path, output_path)
        
        print(f"  ✅ 人设锚点图已创建: {output_path}")
        print(f"  ℹ 所有后续场景将引用此图作为形象锚点（100% 相似度）")
        return output_path
        
    except Exception as e:
        print(f"  ❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="创建角色人设锚点图（Scene 0）- 直接复制参考图")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--character", type=str, default="hanli", help="角色ID（默认：hanli）")
    parser.add_argument("--output-dir", type=str, help="输出目录（默认：gen_video/character_anchors）")
    
    args = parser.parse_args()
    
    # 初始化图像生成器（仅用于读取配置）
    print("初始化图像生成器（读取配置）...")
    generator = ImageGenerator(args.config)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("gen_video/character_anchors")
    
    # 创建人设锚点图（直接复制参考图）
    anchor_path = create_character_anchor(
        generator=generator,
        character_id=args.character,
        output_dir=output_dir,
    )
    
    if anchor_path:
        print(f"\n✅ 人设锚点图创建成功: {anchor_path}")
        print(f"  ℹ 所有后续场景将引用此图作为形象锚点（100% 相似度）")
    else:
        print(f"\n❌ 人设锚点图创建失败")


if __name__ == "__main__":
    main()

