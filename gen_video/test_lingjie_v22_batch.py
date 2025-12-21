#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试 lingjie v2.2-final 场景生成
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def test_scene(generator, scene_file: Path, output_base: Path, scene_index: int, total: int):
    """测试单个场景"""
    print("\n" + "=" * 60)
    print(f"场景 {scene_index}/{total}: {scene_file.name}")
    print("=" * 60)
    
    # 加载JSON
    try:
        with open(scene_file, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
    except Exception as e:
        print(f"✗ JSON加载失败: {e}")
        return {"success": False, "error": f"JSON加载失败: {e}"}
    
    scene_id = scene_data.get("scene", {}).get("scene_id", scene_index - 1)
    shot_type = scene_data.get("scene", {}).get("shot", {}).get("type", "unknown")
    pose_type = scene_data.get("scene", {}).get("pose", {}).get("type", "unknown")
    model = scene_data.get("scene", {}).get("model_route", {}).get("base_model", "unknown")
    
    print(f"场景ID: {scene_id}")
    print(f"Shot: {shot_type}")
    print(f"Pose: {pose_type}")
    print(f"Model: {model}")
    
    # 生成
    start_time = time.time()
    try:
        result = generator.generate(
            scene=scene_data,
            output_dir=str(output_base)
        )
        
        elapsed = time.time() - start_time
        
        if result and result.get("success", False):
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
                print(f"✓ 生成成功 (耗时: {elapsed:.2f}秒)")
                print(f"  ✓ 图像文件: {image_path}")
                print(f"    文件大小: {file_size:.2f} KB")
                
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    print(f"    图片尺寸: {img.size[0]}x{img.size[1]}")
                except Exception as e:
                    print(f"    ⚠ 无法读取图片信息: {e}")
            else:
                print(f"✓ 生成成功，但未找到图像文件")
            
            return {
                "success": True,
                "scene_id": scene_id,
                "elapsed_seconds": round(elapsed, 2),
                "image_path": str(image_path) if image_path else None
            }
        else:
            error_msg = result.get("error", "未知错误") if result else "生成返回None"
            print(f"✗ 生成失败: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "scene_id": scene_id,
                "elapsed_seconds": round(elapsed, 2)
            }
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ 生成异常: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"异常: {str(e)}",
            "scene_id": scene_id,
            "elapsed_seconds": round(elapsed, 2)
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量测试 lingjie v2.2-final 场景生成")
    parser.add_argument(
        "--scenes-dir",
        default="../lingjie/v22",
        help="v2.2-final JSON 场景目录（默认: ../lingjie/v22）"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认: outputs/test_lingjie_v22_YYYYMMDD_HHMMSS）"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=5,
        help="最大测试场景数（默认: 5）"
    )
    
    args = parser.parse_args()
    
    # 查找JSON文件
    scenes_dir = Path(__file__).parent.parent / args.scenes_dir
    if not scenes_dir.exists():
        print(f"✗ 场景目录不存在: {scenes_dir}")
        sys.exit(1)
    
    # 查找所有场景文件
    scene_files = sorted(scenes_dir.glob("scene_*_v22.json"))
    if not scene_files:
        print(f"✗ 未找到场景文件: {scenes_dir / 'scene_*_v22.json'}")
        sys.exit(1)
    
    if args.max_scenes:
        scene_files = scene_files[:args.max_scenes]
    
    print("=" * 60)
    print("批量测试 lingjie v2.2-final 场景生成")
    print("=" * 60)
    print(f"\n找到 {len(scene_files)} 个场景文件:")
    for i, f in enumerate(scene_files, 1):
        print(f"  {i}. {f.name}")
    
    # 创建输出目录
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(__file__).parent / "outputs" / f"test_lingjie_v22_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_base}")
    
    # 初始化生成器
    try:
        from generate_novel_video import NovelVideoGenerator
        
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            print(f"✗ 未找到配置文件: {config_path}")
            sys.exit(1)
        
        print(f"\n✓ 找到配置文件: {config_path}")
        print("🚀 初始化NovelVideoGenerator...")
        
        generator = NovelVideoGenerator(str(config_path))
        
        print("\n" + "=" * 60)
        print("开始批量生成")
        print("=" * 60)
        
        # 批量测试
        results = []
        total_start = time.time()
        
        for i, scene_file in enumerate(scene_files, 1):
            result = test_scene(
                generator=generator,
                scene_file=scene_file,
                output_base=output_base,
                scene_index=i,
                total=len(scene_files)
            )
            results.append(result)
        
        total_elapsed = time.time() - total_start
        
        # 统计结果
        success_count = sum(1 for r in results if r.get("success", False))
        fail_count = len(results) - success_count
        
        print("\n" + "=" * 60)
        print("批量生成完成")
        print("=" * 60)
        print(f"\n总计: {len(results)}")
        print(f"成功: {success_count} ({success_count/len(results)*100:.1f}%)")
        print(f"失败: {fail_count}")
        print(f"总耗时: {total_elapsed:.2f}秒")
        print(f"平均耗时: {total_elapsed/len(results):.2f}秒/场景")
        
        # 保存报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_scenes": len(results),
            "success_count": success_count,
            "fail_count": fail_count,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "average_elapsed_seconds": round(total_elapsed / len(results), 2),
            "results": results
        }
        
        report_json_path = output_base / "batch_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存: {report_json_path}")
        
    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        print("\n💡 提示: 可能需要激活conda环境")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 批量测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

