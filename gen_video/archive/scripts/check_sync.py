#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查字幕和音频同步，以及视频拼接是否正确
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import json

def get_media_duration(media_path: str) -> float:
    """获取媒体文件时长（秒）"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1', media_path],
            capture_output=True,
            text=True,
            check=True
        )
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"⚠ 无法获取 {media_path} 的时长: {e}")
        return 0.0

def parse_srt_time(time_str: str) -> float:
    """解析SRT时间格式 (HH:MM:SS,mmm) 为秒"""
    try:
        time_part, ms_part = time_str.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        total_seconds = h * 3600 + m * 60 + s + ms / 1000.0
        return total_seconds
    except Exception:
        return 0.0

def check_subtitle_sync(subtitle_path: str, audio_path: str, video_path: str = None):
    """检查字幕和音频的同步情况"""
    print("=" * 60)
    print("检查字幕和音频同步")
    print("=" * 60)
    
    # 1. 检查文件是否存在
    if not os.path.exists(subtitle_path):
        print(f"✗ 字幕文件不存在: {subtitle_path}")
        return False
    
    if not os.path.exists(audio_path):
        print(f"✗ 音频文件不存在: {audio_path}")
        return False
    
    # 2. 获取音频时长
    audio_duration = get_media_duration(audio_path)
    print(f"\n音频时长: {audio_duration:.2f} 秒")
    
    # 3. 解析字幕文件
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    # 解析SRT格式
    subtitle_blocks = srt_content.strip().split('\n\n')
    if not subtitle_blocks or subtitle_blocks[0].strip() == '':
        print("⚠ 字幕文件为空")
        return False
    
    first_start = None
    last_end = None
    total_subtitle_duration = 0
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # 解析时间轴 (格式: 00:00:00,000 --> 00:00:00,000)
        time_line = lines[1]
        if '-->' in time_line:
            start_str, end_str = time_line.split('-->')
            start_time = parse_srt_time(start_str.strip())
            end_time = parse_srt_time(end_str.strip())
            
            if first_start is None:
                first_start = start_time
            last_end = end_time
            
            duration = end_time - start_time
            total_subtitle_duration += duration
    
    print(f"字幕时间范围: {first_start:.2f}s - {last_end:.2f}s")
    print(f"字幕总时长: {last_end - first_start:.2f} 秒")
    print(f"字幕内容总时长: {total_subtitle_duration:.2f} 秒")
    
    # 4. 检查同步
    duration_diff = abs(audio_duration - (last_end - first_start))
    print(f"\n时长差异: {duration_diff:.2f} 秒")
    
    if duration_diff < 0.5:
        print("✓ 字幕和音频时长基本同步（差异 < 0.5秒）")
        sync_ok = True
    elif duration_diff < 2.0:
        print("⚠ 字幕和音频时长有轻微差异（差异 < 2秒）")
        sync_ok = True
    else:
        print(f"✗ 字幕和音频时长差异较大（差异 {duration_diff:.2f}秒）")
        sync_ok = False
    
    # 5. 检查字幕是否从0开始
    if first_start is not None and first_start > 1.0:
        print(f"⚠ 警告: 字幕不是从0开始，起始时间: {first_start:.2f}秒")
    
    # 6. 检查字幕是否覆盖完整音频
    if last_end < audio_duration - 1.0:
        print(f"⚠ 警告: 字幕未覆盖完整音频，字幕结束时间: {last_end:.2f}秒，音频时长: {audio_duration:.2f}秒")
    
    return sync_ok

def check_video_concat(video_clips: list, final_video: str = None):
    """检查视频片段能否正确拼接"""
    print("\n" + "=" * 60)
    print("检查视频拼接")
    print("=" * 60)
    
    if not video_clips:
        print("✗ 视频片段列表为空")
        return False
    
    # 1. 检查所有视频文件是否存在
    missing_files = []
    total_duration = 0
    video_info = []
    
    for i, video_path in enumerate(video_clips, 1):
        abs_path = os.path.abspath(video_path)
        if not os.path.exists(abs_path):
            missing_files.append(abs_path)
            continue
        
        duration = get_media_duration(abs_path)
        total_duration += duration
        video_info.append({
            'index': i,
            'path': abs_path,
            'duration': duration
        })
        print(f"  片段 {i}: {os.path.basename(abs_path)} ({duration:.2f}秒)")
    
    if missing_files:
        print(f"\n✗ 以下视频文件不存在:")
        for f in missing_files:
            print(f"    {f}")
        return False
    
    print(f"\n视频片段总数: {len(video_clips)}")
    print(f"视频总时长: {total_duration:.2f} 秒 ({total_duration/60:.1f} 分钟)")
    
    # 2. 检查最终视频（如果提供）
    if final_video and os.path.exists(final_video):
        final_duration = get_media_duration(final_video)
        print(f"\n最终视频时长: {final_duration:.2f} 秒 ({final_duration/60:.1f} 分钟)")
        
        duration_diff = abs(final_duration - total_duration)
        if duration_diff < 0.5:
            print("✓ 最终视频时长与片段总时长一致（差异 < 0.5秒）")
        elif duration_diff < 2.0:
            print(f"⚠ 最终视频时长与片段总时长有轻微差异: {duration_diff:.2f}秒")
        else:
            print(f"✗ 最终视频时长与片段总时长差异较大: {duration_diff:.2f}秒")
    
    # 3. 检查视频格式是否一致
    print("\n检查视频格式一致性...")
    try:
        import ffmpeg
        formats = []
        for video_path in video_clips[:3]:  # 只检查前3个
            probe = ffmpeg.probe(video_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if video_stream:
                codec = video_stream.get('codec_name', 'unknown')
                width = video_stream.get('width', 0)
                height = video_stream.get('height', 0)
                fps = eval(video_stream.get('r_frame_rate', '0/1'))
                formats.append({
                    'codec': codec,
                    'resolution': f"{width}x{height}",
                    'fps': fps
                })
                print(f"  {os.path.basename(video_path)}: {codec}, {width}x{height}, {fps:.2f}fps")
        
        # 检查格式是否一致
        if len(set(f['codec'] for f in formats)) > 1:
            print("⚠ 警告: 视频编码格式不一致，可能影响拼接")
        if len(set(f['resolution'] for f in formats)) > 1:
            print("⚠ 警告: 视频分辨率不一致，可能影响拼接")
        if len(set(f['fps'] for f in formats)) > 1:
            print("⚠ 警告: 视频帧率不一致，可能影响拼接")
    except Exception as e:
        print(f"⚠ 无法检查视频格式: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="检查字幕和音频同步，以及视频拼接")
    parser.add_argument("--subtitle", "-s", help="字幕文件路径 (.srt)")
    parser.add_argument("--audio", "-a", help="音频文件路径")
    parser.add_argument("--video", "-v", help="最终视频文件路径（可选）")
    parser.add_argument("--clips", "-c", nargs="+", help="视频片段路径列表")
    parser.add_argument("--output-dir", "-o", help="输出目录（自动查找字幕和音频）")
    
    args = parser.parse_args()
    
    # 如果提供了输出目录，自动查找文件
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not args.subtitle:
            subtitle_path = output_dir / "subtitle.srt"
            if subtitle_path.exists():
                args.subtitle = str(subtitle_path)
        if not args.audio:
            audio_path = output_dir / "audio.wav"
            if audio_path.exists():
                args.audio = str(audio_path)
        if not args.video:
            # 查找最终视频（.mp4文件）
            video_files = list(output_dir.glob("*.mp4"))
            if video_files:
                # 优先查找不带_upx2等后缀的
                final_videos = [v for v in video_files if '_upx2' not in v.stem and '_upscaled' not in v.stem]
                if final_videos:
                    args.video = str(final_videos[0])
                else:
                    args.video = str(video_files[0])
        if not args.clips:
            # 查找视频片段目录
            clips_dir = output_dir / "videos"
            if clips_dir.exists():
                args.clips = sorted([str(f) for f in clips_dir.glob("*.mp4")])
    
    # 检查字幕和音频同步
    if args.subtitle and args.audio:
        check_subtitle_sync(args.subtitle, args.audio, args.video)
    
    # 检查视频拼接
    if args.clips:
        check_video_concat(args.clips, args.video)
    
    if not args.subtitle and not args.clips:
        print("错误: 需要提供 --subtitle 和 --audio，或 --clips，或 --output-dir")
        parser.print_help()

if __name__ == "__main__":
    main()

