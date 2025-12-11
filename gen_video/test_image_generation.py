#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试图像生成脚本
只生成图像，不生成视频和音频
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from image_generator import ImageGenerator
import argparse


def main():
    parser = argparse.ArgumentParser(description="测试图像生成")
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="JSON 脚本文件路径（例如: ../lingjie/1.json）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录（可选，默认使用配置中的 image_output 目录）"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的图像"
    )
    
    args = parser.parse_args()
    
    # 解析脚本路径
    script_path = Path(args.script)
    if not script_path.is_absolute():
        # 如果是相对路径，尝试从当前目录或 gen_video 目录查找
        script_path = (Path(__file__).parent / script_path).resolve()
        if not script_path.exists():
            # 尝试从上一级目录查找
            script_path = (Path(__file__).parent.parent / args.script).resolve()
    
    if not script_path.exists():
        print(f"❌ 错误: 脚本文件未找到: {args.script}")
        print(f"   尝试的路径: {script_path}")
        return 1
    
    print(f"📄 脚本文件: {script_path}")
    
    # 解析配置文件路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path(__file__).parent / config_path).resolve()
    
    if not config_path.exists():
        print(f"❌ 错误: 配置文件未找到: {config_path}")
        return 1
    
    print(f"⚙️  配置文件: {config_path}")
    
    # 创建图像生成器
    print("\n🔧 正在初始化图像生成器...")
    try:
        image_generator = ImageGenerator(str(config_path))
        print("✅ 图像生成器初始化成功")
    except Exception as e:
        print(f"❌ 图像生成器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 使用配置中的输出目录，基于脚本文件名
        script_name = script_path.stem
        output_dir = Path(image_generator.image_config.get("image_output", "outputs/images")) / f"lingjie_{script_name}_test"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 生成图像
    print(f"\n🎨 开始生成图像...")
    print(f"   脚本: {script_path.name}")
    print(f"   输出: {output_dir}")
    print("=" * 80)
    
    try:
        generated_paths = image_generator.generate_from_script(
            str(script_path),
            output_dir=str(output_dir),
            overwrite=args.overwrite,
            update_script=True,  # 更新 JSON 文件中的 image_path
        )
        
        print("\n" + "=" * 80)
        print(f"✅ 图像生成完成！")
        print(f"   共生成 {len(generated_paths)} 张图像")
        print(f"\n生成的图像:")
        for i, path in enumerate(generated_paths, 1):
            print(f"   {i}. {path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 图像生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())



















