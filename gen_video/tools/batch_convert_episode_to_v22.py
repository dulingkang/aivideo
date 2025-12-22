#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换 episode 目录下的 v2 JSON 文件到 v2.2-final 格式

用法:
    python3 batch_convert_episode_to_v22.py --episode-dir ../lingjie/episode --output-dir ../lingjie/v22
    python3 batch_convert_episode_to_v22.py --episode-dir ../lingjie/episode --output-dir ../lingjie/v22 --episode 1
    python3 batch_convert_episode_to_v22.py --episode-dir ../lingjie/episode --output-dir ../lingjie/v22 --max-scenes 5
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from utils.v2_to_v22_converter import convert_file as convert_v2_file, convert_v2_to_v22
from utils.v1_to_v22_converter import convert_file as convert_v1_file, convert_v1_to_v22


def detect_json_format(json_path: Path) -> Optional[str]:
    """
    检测 JSON 文件的格式
    
    Returns:
        "v2": v2 格式
        "v1": V1 格式（可以直接转换为 v2.2-final）
        None: 无法识别
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 检查是否有 scenes 数组
        if "scenes" not in data:
            return None
        
        scenes = data.get("scenes", [])
        if not scenes:
            return None
        
        # 检查第一个场景的格式
        first_scene = scenes[0]
        
        # v2 格式特征：有 version 字段且值为 "v2"
        if first_scene.get("version") == "v2":
            return "v2"
        
        # v1 格式特征：有 id 字段但没有 version 字段，且有 mood/lighting/action 等字段
        if "id" in first_scene and "version" not in first_scene:
            # 检查是否有 V1 特有的字段
            if "mood" in first_scene or "lighting" in first_scene or "action" in first_scene:
                return "v1"
        
        # 如果已经有 scene_id 和 episode_id，可能是 v2
        if "scene_id" in first_scene and "episode_id" in first_scene:
            return "v2"
        
        return None
        
    except Exception as e:
        print(f"⚠ 无法读取文件 {json_path}: {e}")
        return None


def find_episode_files(episode_dir: Path, episode_num: Optional[int] = None) -> List[Path]:
    """
    查找 episode 目录下的 JSON 文件
    
    Args:
        episode_dir: episode 目录路径
        episode_num: 可选的 episode 编号（如 1, 2, 3...）
    
    Returns:
        JSON 文件路径列表
    """
    files = []
    
    # 如果指定了 episode 编号，优先查找对应的文件
    if episode_num is not None:
        # 优先查找 .v2.json 文件
        v2_file = episode_dir / f"{episode_num}.v2.json"
        if v2_file.exists():
            files.append(v2_file)
            return files
        
        # 其次查找普通 .json 文件
        json_file = episode_dir / f"{episode_num}.json"
        if json_file.exists():
            files.append(json_file)
            return files
    
    # 否则查找所有 JSON 文件
    for json_file in sorted(episode_dir.glob("*.json")):
        # 跳过已经转换过的文件
        if json_file.name.endswith("_v22.json") or json_file.name == "all_scenes_v22.json":
            continue
        
        files.append(json_file)
    
    return files


def convert_episode_file(
    input_file: Path,
    output_dir: Path,
    max_scenes: Optional[int] = None,
    episode_num: Optional[int] = None
) -> bool:
    """
    转换单个 episode JSON 文件
    
    Args:
        input_file: 输入的 JSON 文件路径
        output_dir: 输出目录
        max_scenes: 最大转换场景数
        episode_num: episode 编号（用于创建子目录）
    
    Returns:
        是否成功
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {input_file.name}")
    print(f"{'='*60}")
    
    # 检测格式
    format_type = detect_json_format(input_file)
    
    if format_type is None:
        print(f"⚠ 无法识别文件格式，跳过: {input_file.name}")
        return False
    
    if format_type == "v1":
        print(f"✓ 检测到 V1 格式，将直接转换为 v2.2-final 格式")
    
    if format_type != "v2":
        print(f"⚠ 未知格式，跳过: {input_file.name}")
        return False
    
    print(f"✓ 检测到 v2 格式")
    
    # 确定输出目录
    if episode_num is not None:
        # 如果指定了 episode 编号，创建子目录
        episode_output_dir = output_dir / f"episode_{episode_num}"
    else:
        # 从文件名提取 episode 编号
        episode_num = None
        try:
            # 尝试从文件名提取数字（如 "1.v2.json" -> 1）
            name_parts = input_file.stem.split(".")
            for part in name_parts:
                if part.isdigit():
                    episode_num = int(part)
                    break
        except:
            pass
        
        if episode_num is not None:
            episode_output_dir = output_dir / f"episode_{episode_num}"
        else:
            episode_output_dir = output_dir
    
    episode_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换
    try:
        if format_type == "v1":
            # 使用 V1 到 v2.2-final 的直接转换
            output_files = convert_v1_file(
                input_path=str(input_file),
                output_dir=str(episode_output_dir),
                max_scenes=max_scenes
            )
        elif format_type == "v2":
            # 使用 V2 到 v2.2-final 的转换
            output_files = convert_v2_file(
                input_path=str(input_file),
                output_dir=str(episode_output_dir),
                max_scenes=max_scenes
            )
        else:
            print(f"✗ 未知格式，无法转换")
            return False
        
        print(f"✓ 转换成功: {len(output_files)} 个场景文件")
        return True
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 验证转换后的文件
    if output_files:
        print(f"\n{'='*60}")
        print("验证转换后的文件...")
        print(f"{'='*60}")
        
        try:
            import sys
            from pathlib import Path
            utils_path = Path(__file__).parent.parent / "utils"
            if str(utils_path) not in sys.path:
                sys.path.insert(0, str(utils_path))
            
            from conversion_validator import validate_directory
            
            output_dir_path = Path(output_dir)
            episode_subdir = output_dir_path / f"episode_{episode_num}"
            
            if episode_subdir.exists():
                results = validate_directory(episode_subdir, pattern="scene_*_v22.json", verbose=True)
                all_valid = all(results.values())
                
                if not all_valid:
                    print(f"\n⚠ 警告: 部分文件验证失败，请检查上述错误")
                    return False
                else:
                    print(f"\n✓ 所有文件验证通过")
        except Exception as e:
            print(f"\n⚠ 验证工具运行失败: {e}")
            print("  转换已完成，但建议手动验证文件")


def main():
    parser = argparse.ArgumentParser(
        description="批量转换 episode 目录下的 v2 JSON 文件到 v2.2-final 格式"
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default="../lingjie/episode",
        help="episode 目录路径（默认: ../lingjie/episode）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../lingjie/v22",
        help="输出目录路径（默认: ../lingjie/v22）"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="指定要转换的 episode 编号（如 1, 2, 3...），不指定则转换所有"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="每个 episode 最大转换场景数（默认: 全部）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检测文件，不实际转换"
    )
    
    args = parser.parse_args()
    
    # 解析路径
    script_dir = Path(__file__).parent.parent
    episode_dir = (script_dir / args.episode_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    if not episode_dir.exists():
        print(f"✗ Episode 目录不存在: {episode_dir}")
        sys.exit(1)
    
    print("="*60)
    print("批量转换 episode JSON 文件到 v2.2-final 格式")
    print("="*60)
    print(f"\n输入目录: {episode_dir}")
    print(f"输出目录: {output_dir}")
    if args.episode:
        print(f"Episode 编号: {args.episode}")
    if args.max_scenes:
        print(f"最大场景数: {args.max_scenes}")
    if args.dry_run:
        print("⚠ 仅检测模式（dry-run）")
    
    # 查找文件
    episode_files = find_episode_files(episode_dir, args.episode)
    
    if not episode_files:
        print(f"\n✗ 未找到可转换的 JSON 文件")
        sys.exit(1)
    
    print(f"\n找到 {len(episode_files)} 个文件:")
    for f in episode_files:
        format_type = detect_json_format(f)
        format_str = format_type if format_type else "未知"
        print(f"  - {f.name} ({format_str})")
    
    if args.dry_run:
        print("\n✓ 检测完成（dry-run 模式，未实际转换）")
        return
    
    # 转换
    print(f"\n开始转换...")
    success_count = 0
    fail_count = 0
    
    for episode_file in episode_files:
        # 提取 episode 编号
        episode_num = None
        if args.episode:
            episode_num = args.episode
        else:
            try:
                name_parts = episode_file.stem.split(".")
                for part in name_parts:
                    if part.isdigit():
                        episode_num = int(part)
                        break
            except:
                pass
        
        success = convert_episode_file(
            input_file=episode_file,
            output_dir=output_dir,
            max_scenes=args.max_scenes,
            episode_num=episode_num
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 总结
    print("\n" + "="*60)
    print("转换完成")
    print("="*60)
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(episode_files)}")
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()

