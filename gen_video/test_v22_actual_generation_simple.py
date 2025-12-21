#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final格式实际图像生成测试（简化版）

直接调用generate_novel_video.py的generate方法
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="v2.2-final格式实际图像生成测试")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="schemas/scene_v22_real_example.json",
        help="JSON文件路径（默认: schemas/scene_v22_real_example.json）"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认: outputs/test_v22_actual_YYYYMMDD_HHMMSS）"
    )
    
    args = parser.parse_args()
    
    json_path = Path(__file__).parent / args.json_path
    if not json_path.exists():
        print(f"✗ JSON文件不存在: {json_path}")
        sys.exit(1)
    
    # 加载JSON
    print("=" * 60)
    print("v2.2-final格式实际图像生成测试")
    print("=" * 60)
    print(f"\n使用JSON文件: {json_path}")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
        print(f"✓ JSON文件加载成功")
    except Exception as e:
        print(f"✗ JSON文件加载失败: {e}")
        sys.exit(1)
    
    # 创建输出目录
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(__file__).parent / "outputs" / f"test_v22_actual_{json_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_base}")
    
    # 调用generate_novel_video.py的generate方法
    try:
        from generate_novel_video import NovelVideoGenerator
        
        # 查找config文件
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent / "config.yaml"
        
        if not config_path.exists():
            print(f"✗ 未找到配置文件: {config_path}")
            sys.exit(1)
        
        print(f"\n✓ 找到配置文件: {config_path}")
        print("🚀 初始化NovelVideoGenerator...")
        
        generator = NovelVideoGenerator(str(config_path))
        
        print("\n" + "=" * 60)
        print("开始生成")
        print("=" * 60)
        
        scene_id = scene.get("scene", {}).get("scene_id", 1)
        print(f"场景ID: {scene_id}")
        print(f"Shot: {scene.get('scene', {}).get('shot', {}).get('type')}")
        print(f"Pose: {scene.get('scene', {}).get('pose', {}).get('type')}")
        print(f"Model: {scene.get('scene', {}).get('model_route', {}).get('base_model')}")
        
        import time
        start_time = time.time()
        
        print("\n🚀 调用generate方法...")
        # generate方法的参数是scene，不是scene_data
        # 注意：generate方法会自动检测v2.2-final格式，无需use_v21_exec参数
        result = generator.generate(
            scene=scene,  # 使用scene参数
            output_dir=str(output_base)
        )
        
        elapsed = time.time() - start_time
        
        if result and result.get("success", False):
            print(f"\n✓ 生成成功 (耗时: {elapsed:.2f}秒)")
            
            # 查找生成的图片
            image_path = None
            possible_paths = [
                output_base / f"scene_{scene_id:03d}" / "novel_image.png",
                output_base / "scene_001" / "novel_image.png",
                output_base / f"scene_{scene_id}" / "novel_image.png",
            ]
            
            for p in possible_paths:
                if p.exists():
                    image_path = p
                    break
            
            if image_path:
                file_size = image_path.stat().st_size / 1024
                print(f"  ✓ 图像文件: {image_path}")
                print(f"    文件大小: {file_size:.2f} KB")
                
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    print(f"    图片尺寸: {img.size[0]}x{img.size[1]}")
                    print(f"    图片模式: {img.mode}")
                except Exception as e:
                    print(f"    ⚠ 无法读取图片信息: {e}")
            else:
                print(f"  ⚠ 未找到图像文件")
                print(f"    尝试的路径:")
                for p in possible_paths:
                    print(f"      - {p}")
        else:
            error_msg = result.get("error", "未知错误") if result else "生成返回None"
            print(f"\n✗ 生成失败: {error_msg}")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        print(f"\n输出目录: {output_base}")
        if image_path:
            print(f"图像文件: {image_path}")
        
    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        print("\n💡 提示: 可能需要激活conda环境或安装依赖")
        print("   例如: conda activate <env_name>")
        print("   或者: pip install torch torchvision")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 生成异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

