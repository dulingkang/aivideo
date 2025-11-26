#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将视频文件缩放到 1920x1080
保持音频不变，使用高质量缩放算法
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    """获取视频分辨率"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split('x'))
        return width, height
    except Exception as e:
        print(f"❌ 无法获取视频分辨率: {video_path} - {e}")
        return None, None


def resize_video_to_1080p(
    input_path: Path,
    output_path: Path,
    target_width: int = 1920,
    target_height: int = 1080,
    crf: int = 23,
    preset: str = "medium",
) -> bool:
    """
    将视频缩放到 1920x1080
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_width: 目标宽度
        target_height: 目标高度
        crf: 质量参数（18-28，越小质量越好）
        preset: 编码预设（ultrafast, fast, medium, slow, veryslow）
    """
    # 检查输入文件
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return False
    
    # 检查当前分辨率
    width, height = get_video_resolution(input_path)
    if width is None or height is None:
        return False
    
    if width == target_width and height == target_height:
        print(f"⏭️  跳过: {input_path.name} 已经是 {target_width}x{target_height}")
        return True
    
    print(f"📹 处理: {input_path.name}")
    print(f"   当前分辨率: {width}x{height}")
    print(f"   目标分辨率: {target_width}x{target_height}")
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建 ffmpeg 命令
    # 使用 lanczos 算法进行高质量缩放
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-i", str(input_path),
        "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",  # 直接复制音频，不重新编码
        "-movflags", "+faststart",  # 优化网络播放
        str(output_path),
    ]
    
    try:
        print(f"   开始转换...")
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 检查输出文件大小
        input_size = input_path.stat().st_size / (1024 * 1024)  # MB
        output_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"   ✅ 完成")
        print(f"   文件大小: {input_size:.1f}MB → {output_size:.1f}MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 转换失败: {e}")
        if e.stderr:
            print(f"   错误信息: {e.stderr.decode('utf-8', errors='ignore')[:200]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量将视频缩放到 1920x1080")
    parser.add_argument(
        "--input", "-i",
        required=True,
        nargs="+",
        help="输入视频文件（可多个，支持通配符）",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="输出目录（默认：覆盖原文件，添加 _1080p 后缀）",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="视频质量参数（18-28，默认23）",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        help="编码预设（默认：medium）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要处理的文件，不实际转换",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="直接覆盖原文件（危险操作，请谨慎使用）",
    )
    
    args = parser.parse_args()
    
    # 收集所有输入文件
    import glob
    input_files = []
    for pattern in args.input:
        matched = glob.glob(pattern, recursive=True)
        input_files.extend([Path(f) for f in matched if Path(f).is_file()])
    
    if not input_files:
        print("❌ 未找到任何视频文件")
        return 1
    
    print(f"找到 {len(input_files)} 个视频文件")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 预览模式（不会实际转换）")
        print()
        for video_file in input_files:
            width, height = get_video_resolution(video_file)
            if width and height:
                print(f"  {video_file.name}: {width}x{height}")
        return 0
    
    # 处理每个文件
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for video_file in input_files:
        if args.in_place:
            # 直接覆盖原文件
            output_file = video_file
            temp_file = video_file.with_suffix('.tmp' + video_file.suffix)
            actual_output = temp_file
        elif args.output_dir:
            # 输出到指定目录
            output_dir = Path(args.output_dir)
            output_file = output_dir / video_file.name
            actual_output = output_file
        else:
            # 添加 _1080p 后缀
            output_file = video_file.with_stem(video_file.stem + "_1080p")
            actual_output = output_file
        
        if resize_video_to_1080p(
            video_file,
            actual_output,
            crf=args.crf,
            preset=args.preset,
        ):
            if args.in_place and actual_output.exists():
                # 替换原文件
                video_file.unlink()
                actual_output.rename(video_file)
                print(f"   ✅ 已覆盖原文件")
            success_count += 1
        else:
            if actual_output.exists() and actual_output != video_file:
                actual_output.unlink()  # 删除失败的文件
            fail_count += 1
        print()
    
    # 统计结果
    print("=" * 60)
    print(f"处理完成:")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {fail_count} 个")
    if skip_count > 0:
        print(f"  跳过: {skip_count} 个（已是目标分辨率）")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

