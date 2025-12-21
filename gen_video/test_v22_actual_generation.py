#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final格式实际图像生成测试

使用真实的ImageGenerator生成图片
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def test_actual_generation(json_path: str):
    """实际生成图像测试"""
    print("=" * 60)
    print("v2.2-final格式实际图像生成测试")
    print("=" * 60)
    print(f"\n使用JSON文件: {json_path}")
    
    # 加载JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
        print(f"✓ JSON文件加载成功")
    except Exception as e:
        print(f"✗ JSON文件加载失败: {e}")
        return False
    
    # 创建输出目录
    json_file = Path(json_path)
    output_base = Path(__file__).parent / "outputs" / f"test_v22_actual_{json_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_base}")
    print(f"生成的图片将保存在: {output_base / 'scene_001' / 'novel_image.png'}")
    
    try:
        # 使用generate_novel_video.py的方式，避免导入问题
        from generate_novel_video import NovelVideoGenerator
        import yaml
        
        # 查找config文件
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent / "config.yaml"
        
        if not config_path.exists():
            print(f"  ✗ 未找到配置文件: {config_path}")
            print(f"  ℹ 尝试的路径:")
            print(f"    - {Path(__file__).parent.parent / 'config.yaml'}")
            print(f"    - {Path(__file__).parent / 'config.yaml'}")
            return False
        
        print(f"  ℹ 找到配置文件: {config_path}")
        
        # 1. 初始化生成器
        print("\n" + "=" * 60)
        print("步骤1: 初始化生成器")
        print("=" * 60)
        print("  🚀 初始化NovelVideoGenerator...")
        generator = NovelVideoGenerator(str(config_path))
        print("  ✓ NovelVideoGenerator初始化成功")
        
        # 2. 验证JSON
        print("\n" + "=" * 60)
        print("步骤2: JSON验证")
        print("=" * 60)
        
        # 检查版本
        version = scene.get("version", "")
        if version == "v2.2-final":
            print(f"  ✓ 检测到v2.2-final格式")
        else:
            print(f"  ⚠ 版本: {version}")
        
        # 3. 使用generate方法生成
        print("\n" + "=" * 60)
        print("步骤3: 开始生成")
        print("=" * 60)
        
        scene_id = scene.get("scene", {}).get("scene_id", 1)
        print(f"  场景ID: {scene_id}")
        print(f"  Shot: {scene.get('scene', {}).get('shot', {}).get('type')}")
        print(f"  Pose: {scene.get('scene', {}).get('pose', {}).get('type')}")
        print(f"  Model: {scene.get('scene', {}).get('model_route', {}).get('base_model')}")
        
        try:
            import time
            start_time = time.time()
            
            # 使用generate方法
            print("  🚀 调用generate方法...")
            result = generator.generate(
                scene_data=scene,
                output_dir=str(output_base),
                use_v21_exec=True  # 使用v2.1执行器
            )
            
            elapsed = time.time() - start_time
            
            if result and result.get("success", False):
                print(f"  ✓ 生成成功 (耗时: {elapsed:.2f}秒)")
                
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
                    print(f"    ✓ 图像文件存在: {image_path}")
                    print(f"    文件大小: {file_size:.2f} KB")
                    
                    # 显示图片信息
                    try:
                        from PIL import Image
                        img = Image.open(image_path)
                        print(f"    图片尺寸: {img.size[0]}x{img.size[1]}")
                        print(f"    图片模式: {img.mode}")
                    except Exception as e:
                        print(f"    ⚠ 无法读取图片信息: {e}")
                else:
                    print(f"    ⚠ 未找到图像文件，可能路径:")
                    for p in possible_paths:
                        print(f"      - {p}")
            else:
                error_msg = result.get("error", "未知错误") if result else "生成返回None"
                print(f"  ✗ 生成失败: {error_msg}")
                return False
                
        except Exception as e:
            print(f"  ✗ 生成异常: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 保存测试结果
        print("\n" + "=" * 60)
        print("步骤4: 保存测试结果")
        print("=" * 60)
        
        # 复制原始JSON
        json_output_path = output_base / json_file.name
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 测试JSON已保存: {json_output_path}")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        print(f"\n输出目录: {output_base}")
        print(f"  - 测试JSON: {json_output_path}")
        if image_path:
            print(f"  - 图像文件: {image_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


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
    
    args = parser.parse_args()
    
    json_path = Path(__file__).parent / args.json_path
    if not json_path.exists():
        print(f"✗ JSON文件不存在: {json_path}")
        print(f"\n可用的JSON文件:")
        json_files = list(Path(__file__).parent.glob("schemas/scene_v22*.json"))
        for f in json_files:
            print(f"  - {f}")
        sys.exit(1)
    
    success = test_actual_generation(str(json_path))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

