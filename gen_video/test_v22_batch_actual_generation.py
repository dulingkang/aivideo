#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final格式批量实际图像生成测试

测试多个场景的连续生成
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def test_single_scene(
    generator,
    json_path: Path,
    output_base: Path,
    scene_index: int,
    total_scenes: int
) -> Dict[str, Any]:
    """测试单个场景"""
    print("\n" + "=" * 60)
    print(f"场景 {scene_index}/{total_scenes}: {json_path.name}")
    print("=" * 60)
    
    # 加载JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
    except Exception as e:
        return {
            "success": False,
            "error": f"JSON加载失败: {e}",
            "scene_id": None,
            "json_path": str(json_path)
        }
    
    scene_id = scene.get("scene", {}).get("scene_id", scene_index)
    shot_type = scene.get("scene", {}).get("shot", {}).get("type", "unknown")
    pose_type = scene.get("scene", {}).get("pose", {}).get("type", "unknown")
    model = scene.get("scene", {}).get("model_route", {}).get("base_model", "unknown")
    
    print(f"场景ID: {scene_id}")
    print(f"Shot: {shot_type}")
    print(f"Pose: {pose_type}")
    print(f"Model: {model}")
    
    # 生成
    start_time = time.time()
    try:
        result = generator.generate(
            scene=scene,
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
            
            image_info = {}
            if image_path:
                file_size = image_path.stat().st_size / 1024
                image_info = {
                    "path": str(image_path),
                    "size_kb": round(file_size, 2),
                    "exists": True
                }
                
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    image_info["width"] = img.size[0]
                    image_info["height"] = img.size[1]
                    image_info["mode"] = img.mode
                except Exception as e:
                    image_info["read_error"] = str(e)
            else:
                image_info = {
                    "exists": False,
                    "searched_paths": [str(p) for p in possible_paths]
                }
            
            print(f"✓ 生成成功 (耗时: {elapsed:.2f}秒)")
            if image_info.get("exists"):
                print(f"  ✓ 图像文件: {image_info['path']}")
                print(f"    文件大小: {image_info.get('size_kb', 0)} KB")
                print(f"    图片尺寸: {image_info.get('width', 0)}x{image_info.get('height', 0)}")
            
            return {
                "success": True,
                "scene_id": scene_id,
                "json_path": str(json_path),
                "shot_type": shot_type,
                "pose_type": pose_type,
                "model": model,
                "elapsed_seconds": round(elapsed, 2),
                "image": image_info
            }
        else:
            error_msg = result.get("error", "未知错误") if result else "生成返回None"
            print(f"✗ 生成失败: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "scene_id": scene_id,
                "json_path": str(json_path),
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
            "json_path": str(json_path),
            "elapsed_seconds": round(elapsed, 2)
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="v2.2-final格式批量实际图像生成测试")
    parser.add_argument(
        "--json-dir",
        default="schemas",
        help="JSON文件目录（默认: schemas）"
    )
    parser.add_argument(
        "--json-pattern",
        default="scene_v22_real_example*.json",
        help="JSON文件匹配模式（默认: scene_v22_real_example*.json）"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认: outputs/test_v22_batch_YYYYMMDD_HHMMSS）"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="最大测试场景数（默认: 全部）"
    )
    
    args = parser.parse_args()
    
    # 查找JSON文件
    json_dir = Path(__file__).parent / args.json_dir
    if not json_dir.exists():
        print(f"✗ JSON目录不存在: {json_dir}")
        sys.exit(1)
    
    json_files = sorted(json_dir.glob(args.json_pattern))
    if not json_files:
        print(f"✗ 未找到匹配的JSON文件: {json_dir / args.json_pattern}")
        sys.exit(1)
    
    if args.max_scenes:
        json_files = json_files[:args.max_scenes]
    
    print("=" * 60)
    print("v2.2-final格式批量实际图像生成测试")
    print("=" * 60)
    print(f"\n找到 {len(json_files)} 个JSON文件:")
    for i, f in enumerate(json_files, 1):
        print(f"  {i}. {f.name}")
    
    # 创建输出目录
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(__file__).parent / "outputs" / f"test_v22_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_base}")
    
    # 初始化生成器
    try:
        from generate_novel_video import NovelVideoGenerator
        
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
        print("开始批量生成")
        print("=" * 60)
        
        # 批量测试
        results = []
        total_start = time.time()
        
        for i, json_path in enumerate(json_files, 1):
            result = test_single_scene(
                generator=generator,
                json_path=json_path,
                output_base=output_base,
                scene_index=i,
                total_scenes=len(json_files)
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
        report_md_path = output_base / "batch_report.md"
        
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成Markdown报告
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write("# v2.2-final批量生成测试报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**总计**: {len(results)} 个场景\n\n")
            f.write(f"**成功**: {success_count} ({success_count/len(results)*100:.1f}%)\n\n")
            f.write(f"**失败**: {fail_count}\n\n")
            f.write(f"**总耗时**: {total_elapsed:.2f}秒\n\n")
            f.write(f"**平均耗时**: {total_elapsed/len(results):.2f}秒/场景\n\n")
            
            f.write("## 详细结果\n\n")
            f.write("| 序号 | 场景ID | Shot | Pose | Model | 状态 | 耗时(秒) | 图像 |\n")
            f.write("|------|--------|------|------|-------|------|----------|------|\n")
            
            for i, r in enumerate(results, 1):
                status = "✓ 成功" if r.get("success") else "✗ 失败"
                elapsed = r.get("elapsed_seconds", 0)
                scene_id = r.get("scene_id", "N/A")
                shot = r.get("shot_type", "N/A")
                pose = r.get("pose_type", "N/A")
                model = r.get("model", "N/A")
                
                image_info = r.get("image", {})
                if image_info.get("exists"):
                    image_str = f"✓ {image_info.get('width', 0)}x{image_info.get('height', 0)}"
                else:
                    image_str = "✗ 未找到"
                
                f.write(f"| {i} | {scene_id} | {shot} | {pose} | {model} | {status} | {elapsed} | {image_str} |\n")
            
            f.write("\n## 失败详情\n\n")
            for i, r in enumerate(results, 1):
                if not r.get("success"):
                    f.write(f"### 场景 {i}: {r.get('json_path', 'N/A')}\n\n")
                    f.write(f"- **错误**: {r.get('error', '未知错误')}\n\n")
        
        print(f"\n报告已保存:")
        print(f"  JSON: {report_json_path}")
        print(f"  Markdown: {report_md_path}")
        
    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        print("\n💡 提示: 可能需要激活conda环境或安装依赖")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 批量测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

