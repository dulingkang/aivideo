#!/usr/bin/env python3
"""
提取视频测试片段：用于测试裁剪参数
从视频中间提取几秒，保留字幕区域
"""

import subprocess
import argparse
from pathlib import Path

def extract_test_clip(input_path, output_path, duration=5, start_time=None):
    """
    提取视频测试片段
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        duration: 提取时长（秒），默认5秒
        start_time: 开始时间（秒），如果为None则从中间开始
    """
    # 获取视频总时长
    cmd_probe = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    result = subprocess.run(cmd_probe, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    
    # 如果没有指定开始时间，从中间开始
    if start_time is None:
        start_time = max(0, (total_duration - duration) / 2)
    
    print(f"视频总时长: {total_duration:.2f} 秒")
    print(f"提取片段: {start_time:.2f} 秒 - {start_time + duration:.2f} 秒 (共 {duration} 秒)")
    
    # 提取片段（不进行任何处理，保留原始质量）
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-ss', str(start_time),
        '-t', str(duration),
        '-c', 'copy',  # 直接复制流，不重新编码（快速）
        '-y',
        str(output_path)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    
    print(f"✓ 测试片段已保存: {output_path}")
    return True

def get_video_info(input_path):
    """获取视频信息（分辨率、时长等）"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration',
        '-of', 'json',
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        import json
        data = json.loads(result.stdout)
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            return {
                'width': stream.get('width'),
                'height': stream.get('height'),
                'duration': stream.get('duration')
            }
    return None

def main():
    parser = argparse.ArgumentParser(
        description='提取视频测试片段（用于测试裁剪参数）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从中间提取5秒
  python3 extract_test_clip.py -i input.mp4 -o test_clip.mp4
  
  # 从第10秒开始提取3秒
  python3 extract_test_clip.py -i input.mp4 -o test_clip.mp4 --start 10 --duration 3
  
  # 提取4K视频的测试片段
  python3 extract_test_clip.py -i 4k_video.mp4 -o test_4k.mp4
        """
    )
    parser.add_argument('--input', '-i', required=True, help='输入视频路径')
    parser.add_argument('--output', '-o', required=True, help='输出视频路径')
    parser.add_argument('--duration', '-d', type=float, default=5,
                       help='提取时长（秒），默认5秒')
    parser.add_argument('--start', '-s', type=float, default=None,
                       help='开始时间（秒），如果不指定则从中间开始')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return 1
    
    # 获取视频信息
    info = get_video_info(input_path)
    if info:
        print(f"视频分辨率: {info['width']}x{info['height']}")
        if info['duration']:
            print(f"视频时长: {float(info['duration']):.2f} 秒")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if extract_test_clip(input_path, output_path, args.duration, args.start):
        print(f"\n提示: 现在可以用这个测试片段来调整裁剪参数")
        print(f"例如: python3 clean_video.py -i {output_path} -o test_cropped.mp4 --crop W:H:X:Y")
        return 0
    else:
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())

