#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
转换后 JSON 验证工具 - 便捷脚本
"""
import sys
from pathlib import Path

# 添加 utils 路径
utils_path = Path(__file__).parent.parent / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from conversion_validator import validate_file, validate_directory

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="验证转换后的 v2.2-final JSON 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 验证单个文件
  python3 validate_converted_json.py ../lingjie/v22/episode_1/scene_001_v22.json
  
  # 验证整个目录
  python3 validate_converted_json.py ../lingjie/v22/episode_1
  
  # 验证指定模式的文件
  python3 validate_converted_json.py ../lingjie/v22/episode_1 --pattern "scene_*_v22.json"
        """
    )
    parser.add_argument(
        "path",
        type=str,
        help="JSON 文件路径或目录路径"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="scene_*_v22.json",
        help="文件匹配模式（仅目录模式，默认: scene_*_v22.json）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只显示结果"
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"✗ 路径不存在: {path}")
        sys.exit(1)
    
    if path.is_file():
        # 验证单个文件
        is_valid = validate_file(path, verbose=not args.quiet)
        sys.exit(0 if is_valid else 1)
    elif path.is_dir():
        # 验证目录
        results = validate_directory(path, pattern=args.pattern, verbose=not args.quiet)
        all_valid = all(results.values())
        sys.exit(0 if all_valid else 1)
    else:
        print(f"✗ 无效路径: {path}")
        sys.exit(1)

