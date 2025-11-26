#!/usr/bin/env python3
"""
提取关键帧：从每个镜头片段中提取首帧/中帧用于标注
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def extract_keyframe(video_path, output_path, time_offset='0', frame_number=None):
    """
    从视频中提取关键帧
    
    Args:
        video_path: 视频路径
        output_path: 输出图片路径
        time_offset: 时间偏移（秒），或 'start'/'middle'/'end'
        frame_number: 直接指定帧号（如果指定则忽略time_offset）
    """
    cmd = ['ffmpeg', '-i', str(video_path), '-y']
    
    if frame_number is not None:
        cmd.extend(['-vf', f'select=eq(n\\,{frame_number})', '-vsync', '0'])
    elif time_offset == 'start':
        cmd.extend(['-ss', '0.5', '-vframes', '1'])  # 开头0.5秒
    elif time_offset == 'middle':
        # 获取视频时长，取中间帧
        duration_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
        ]
        result = subprocess.run(duration_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            mid_time = duration / 2
            cmd.extend(['-ss', str(mid_time), '-vframes', '1'])
        else:
            cmd.extend(['-ss', '1', '-vframes', '1'])
    elif time_offset == 'end':
        cmd.extend(['-sseof', '-1', '-vframes', '1'])
    else:
        cmd.extend(['-ss', str(time_offset), '-vframes', '1'])
    
    cmd.extend(['-q:v', '2', str(output_path)])  # 高质量JPEG
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def extract_scene_keyframes(scene_dir, output_dir, positions=['start', 'middle']):
    """
    从场景目录中的所有视频提取关键帧
    
    Args:
        scene_dir: 场景视频目录
        output_dir: 关键帧输出目录
        positions: 提取位置列表 ['start', 'middle', 'end']
    """
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene_files = sorted(scene_dir.glob('*.mp4'))
    print(f"找到 {len(scene_files)} 个场景视频")
    
    for scene_file in scene_files:
        scene_name = scene_file.stem  # 例如 scene_000
        
        for pos in positions:
            keyframe_name = f"{scene_name}_{pos}.jpg"
            keyframe_path = output_dir / keyframe_name
            
            if extract_keyframe(scene_file, keyframe_path, time_offset=pos):
                print(f"  ✓ {scene_name} - {pos}: {keyframe_path}")
            else:
                print(f"  ✗ 提取失败: {scene_name} - {pos}")
    
    print(f"✓ 关键帧提取完成: {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='从场景视频中提取关键帧')
    parser.add_argument('--input', '-i', required=True, help='场景视频目录')
    parser.add_argument('--output', '-o', required=True, help='关键帧输出目录')
    parser.add_argument('--positions', '-p', nargs='+', 
                       default=['start', 'middle'],
                       choices=['start', 'middle', 'end'],
                       help='提取位置')
    
    args = parser.parse_args()
    
    scene_dir = Path(args.input)
    if not scene_dir.exists():
        print(f"错误: 输入目录不存在: {scene_dir}")
        return 1
    
    if extract_scene_keyframes(scene_dir, args.output, args.positions):
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())

