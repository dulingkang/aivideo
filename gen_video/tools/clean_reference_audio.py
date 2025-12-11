#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理参考音频
去除静音、降噪、标准化音量，提升声音克隆质量
"""

import argparse
import subprocess
from pathlib import Path


def clean_audio(input_path: Path, output_path: Path, 
                remove_silence: bool = True,
                denoise: bool = True,
                normalize: bool = True,
                target_sr: int = 16000):
    """
    清理音频文件
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        remove_silence: 是否去除静音
        denoise: 是否降噪
        normalize: 是否标准化音量
        target_sr: 目标采样率（默认16kHz，CosyVoice2要求）
    """
    print(f"清理音频: {input_path.name}")
    print(f"输出: {output_path.name}")
    
    # 构建 FFmpeg 命令
    filters = []
    
    # 1. 去除静音（使用 silenceremove）
    if remove_silence:
        # 去除开头和结尾的静音
        # -50dB 阈值，0.5秒持续时间
        filters.append("silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB:detection=peak")
        filters.append("silenceremove=stop_periods=-1:stop_silence=0.1:stop_threshold=-50dB:detection=peak")
        print("  ✓ 启用静音去除")
    
    # 2. 降噪（使用 highpass 和 lowpass 滤波器）
    if denoise:
        # 高通滤波器：去除低频噪音（200Hz以下）
        # 低通滤波器：去除高频噪音（8000Hz以上，保留人声范围）
        filters.append("highpass=f=200")
        filters.append("lowpass=f=8000")
        print("  ✓ 启用降噪")
    
    # 3. 标准化音量（使用 loudnorm）
    if normalize:
        # 标准化到 -16 LUFS（广播标准）
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        print("  ✓ 启用音量标准化")
    
    # 构建完整命令
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-ar', str(target_sr),  # 设置采样率
        '-ac', '1',  # 单声道
        '-acodec', 'pcm_s16le',  # 16-bit PCM
    ]
    
    # 添加滤镜
    if filters:
        cmd.extend(['-af', ','.join(filters)])
    
    cmd.append(str(output_path))
    
    print(f"\n执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 清理失败: {result.stderr}")
        return False
    
    # 检查输出文件
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"\n✅ 清理完成: {output_path}")
        print(f"   文件大小: {file_size / 1024:.2f} KB")
        
        # 显示音频信息
        info_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=sample_rate,channels,duration',
            '-of', 'default=noprint_wrappers=1',
            str(output_path)
        ]
        info_result = subprocess.run(info_cmd, capture_output=True, text=True)
        if info_result.returncode == 0:
            print("   音频信息:")
            for line in info_result.stdout.strip().split('\n'):
                if line:
                    print(f"     {line}")
        
        return True
    else:
        print(f"❌ 输出文件未生成")
        return False


def extract_clear_segment(input_path: Path, output_path: Path,
                          start_time: float = None,
                          duration: float = None,
                          target_sr: int = 16000):
    """
    提取音频中最清晰的部分
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        start_time: 开始时间（秒），None 表示从开头
        duration: 持续时间（秒），None 表示到结尾
        target_sr: 目标采样率
    """
    print(f"提取清晰片段: {input_path.name}")
    
    cmd = ['ffmpeg', '-y', '-i', str(input_path)]
    
    if start_time:
        cmd.extend(['-ss', str(start_time)])
    
    if duration:
        cmd.extend(['-t', str(duration)])
    
    cmd.extend([
        '-ar', str(target_sr),
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        str(output_path)
    ])
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 提取失败: {result.stderr}")
        return False
    
    if output_path.exists():
        print(f"✅ 提取完成: {output_path}")
        return True
    else:
        print(f"❌ 输出文件未生成")
        return False


def main():
    parser = argparse.ArgumentParser(description='清理参考音频，提升声音克隆质量')
    parser.add_argument('--input', type=str, required=True,
                       help='输入音频文件路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出音频文件路径')
    parser.add_argument('--no-silence-remove', action='store_true',
                       help='不去除静音')
    parser.add_argument('--no-denoise', action='store_true',
                       help='不降噪')
    parser.add_argument('--no-normalize', action='store_true',
                       help='不标准化音量')
    parser.add_argument('--extract', action='store_true',
                       help='提取清晰片段模式（需要指定 --start 和 --duration）')
    parser.add_argument('--start', type=float, default=None,
                       help='提取开始时间（秒）')
    parser.add_argument('--duration', type=float, default=None,
                       help='提取持续时间（秒）')
    parser.add_argument('--target-sr', type=int, default=16000,
                       help='目标采样率（默认16000，CosyVoice2要求）')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return 1
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.extract:
        # 提取模式
        success = extract_clear_segment(
            input_path, output_path,
            start_time=args.start,
            duration=args.duration,
            target_sr=args.target_sr
        )
    else:
        # 清理模式
        success = clean_audio(
            input_path, output_path,
            remove_silence=not args.no_silence_remove,
            denoise=not args.no_denoise,
            normalize=not args.no_normalize,
            target_sr=args.target_sr
        )
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

