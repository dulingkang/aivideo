#!/usr/bin/env python3
"""
批量处理视频静音
将原始场景视频处理成静音版本，供后续快速拼接使用
"""

import argparse
import subprocess
from pathlib import Path
from typing import List
import sys

def get_video_duration(video_path: Path) -> float:
    """获取视频时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

def mute_video(input_path: Path, output_path: Path):
    """
    将视频静音（去掉音频轨道）
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径（静音版本）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-c:v', 'libx264',  # 重新编码视频
        '-preset', 'fast',  # 快速编码
        '-crf', '23',       # 质量控制
        '-an',              # 去掉音频（静音）
        '-y',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def process_directory(input_dir: Path, output_dir: Path, pattern: str = "*.mp4", 
                     skip_existing: bool = True, dry_run: bool = False):
    """
    处理目录中的所有视频文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（静音版本）
        pattern: 文件匹配模式（默认: *.mp4）
        skip_existing: 跳过已存在的文件
        dry_run: 仅显示将要处理的文件，不实际处理
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 查找所有视频文件
    video_files = list(input_dir.glob(pattern))
    
    if not video_files:
        print(f"⚠️  没有找到匹配 {pattern} 的视频文件: {input_dir}")
        return
    
    print(f"=" * 60)
    print(f"批量静音处理")
    print(f"=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(video_files)} 个视频文件")
    print()
    
    if dry_run:
        print("🔍 预览模式（不会实际处理）:")
        print()
    
    processed = 0
    skipped = 0
    failed = 0
    
    for i, video_file in enumerate(sorted(video_files), 1):
        # 计算相对路径，保持目录结构
        relative_path = video_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        
        # 检查是否已存在
        if skip_existing and output_file.exists():
            duration = get_video_duration(output_file)
            print(f"[{i}/{len(video_files)}] ⏭️  跳过（已存在）: {relative_path} ({duration:.2f}秒)")
            skipped += 1
            continue
        
        if dry_run:
            duration = get_video_duration(video_file)
            print(f"[{i}/{len(video_files)}] 📝 将处理: {relative_path} ({duration:.2f}秒)")
            continue
        
        # 处理视频
        duration = get_video_duration(video_file)
        print(f"[{i}/{len(video_files)}] 🔄 处理: {relative_path} ({duration:.2f}秒)", end=" ... ")
        sys.stdout.flush()
        
        if mute_video(video_file, output_file):
            output_duration = get_video_duration(output_file)
            print(f"✅ 完成 ({output_duration:.2f}秒)")
            processed += 1
        else:
            print(f"❌ 失败")
            failed += 1
    
    print()
    print("=" * 60)
    if dry_run:
        print(f"预览完成: {len(video_files)} 个文件")
    else:
        print(f"处理完成:")
        print(f"  ✅ 成功: {processed}")
        print(f"  ⏭️  跳过: {skipped}")
        print(f"  ❌ 失败: {failed}")
        print(f"  📁 总计: {len(video_files)}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='批量处理视频静音')
    parser.add_argument('--input', '-i', required=True,
                       help='输入目录（包含视频文件）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录（静音版本）')
    parser.add_argument('--pattern', default='*.mp4',
                       help='文件匹配模式（默认: *.mp4）')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='不跳过已存在的文件（强制重新处理）')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式：只显示将要处理的文件，不实际处理')
    
    args = parser.parse_args()
    
    process_directory(
        Path(args.input),
        Path(args.output),
        pattern=args.pattern,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run
    )

if __name__ == '__main__':
    main()

