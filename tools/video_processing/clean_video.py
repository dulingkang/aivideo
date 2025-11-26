#!/usr/bin/env python3
"""
原番清洗脚本：去除字幕和LOGO
支持自动检测字幕区域并裁剪，轻度遮挡LOGO
注意：建议在原始视频上统一清洗，而不是在切分后的片段上
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def calculate_aspect_crop(crop_region, target_aspect=16/9):
    """
    计算等比例裁剪参数：从已裁剪的区域中，左右再裁一点，保持目标宽高比
    
    Args:
        crop_region: 原始裁剪区域 "w:h:x:y"，例如 "1920:980:0:0"
        target_aspect: 目标宽高比，默认16:9
    
    Returns:
        新的裁剪区域 "w:h:x:y"
    """
    parts = crop_region.split(':')
    if len(parts) != 4:
        return crop_region
    
    w, h, x, y = map(int, parts)
    
    # 计算保持目标宽高比时的新宽度
    new_w = int(h * target_aspect)
    
    # 如果新宽度小于原宽度，需要左右裁剪
    if new_w < w:
        crop_x = (w - new_w) // 2  # 左右各裁掉一半
        return f"{new_w}:{h}:{crop_x}:{y}"
    else:
        # 如果新宽度大于原宽度，说明高度需要调整（这种情况不应该发生）
        return crop_region

def detect_subtitle_region(video_path, threshold=0.3):
    """
    使用ffprobe检测字幕区域（简单方法：检测底部固定区域）
    返回 (width, height, x, y) 用于crop
    """
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) >= 2:
        width = int(lines[0])
        height = int(lines[1])
        # 假设字幕在底部120-150px区域，裁剪掉
        crop_height = height - 150
        return f"{width}:{crop_height}:0:0"
    return None

def build_scale_filter(scale_to, mode='stretch'):
    """生成缩放相关滤镜"""
    if not scale_to:
        return []
    if mode == 'letterbox':
        # 等比例缩放并补边
        return [
            f'scale={scale_to}:force_original_aspect_ratio=decrease',
            f'pad={scale_to}:(ow-iw)/2:(oh-ih)/2'
        ]
    # 默认直接拉伸
    return [f'scale={scale_to}']


def parse_time(time_str):
    """解析时间字符串，支持秒数或HH:MM:SS格式，返回秒数"""
    if not time_str:
        return None
    
    # 如果包含冒号，按HH:MM:SS格式解析
    if ':' in str(time_str):
        parts = str(time_str).split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
    # 否则直接当作秒数
    try:
        return float(time_str)
    except:
        return None

def clean_video(input_path, output_path, crop_region=None, delogo_config=None,
                scale_to=None, scale_mode='stretch', auto_aspect=False, audio_path=None,
                trim_start=None, trim_end=None, duration=None, mute=False):
    """
    清洗视频：裁剪字幕区域，轻度遮挡LOGO，可缩放回原始分辨率
    
    Args:
        crop_region: 裁剪区域 "w:h:x:y"，例如 "1920:980:0:0" (裁掉底部100px)
        delogo_config: LOGO区域 "x:y:w:h"，例如 "1650:40:200:150"
        scale_to: 缩放目标分辨率 "w:h"，例如 "1920:1080"，默认不缩放
        audio_path: 外部音频文件路径（如果视频没有音频轨道）
        trim_start: 跳过开头的时间，格式 "HH:MM:SS" 或秒数（如 "5" 或 "00:00:05"）
        trim_end: 跳过结尾的时间，格式 "HH:MM:SS" 或秒数（如 "30" 或 "00:00:30"），从视频末尾往前去掉指定时长
        duration: 输出视频的总时长，格式 "HH:MM:SS" 或秒数（如 "600"），从开头开始截取指定时长（优先级高于trim_end）
        
    注意：
        - crop参数中 y=0 表示从顶部保留，裁掉底部
        - 例如 1920:980:0:0 表示保留1920x980，裁掉底部100px
        - delogo会模糊指定区域，适合遮挡LOGO
        - 如果指定scale_to，会在最后将视频缩放到目标分辨率
        - 如果指定audio_path，会使用外部音频文件替换视频中的音频
        - 如果指定trim_start，会跳过视频开头的指定时间（如跳过下集预告、序幕等）
        - 如果指定trim_end或duration，会控制输出视频的结束时间
        - duration优先级高于trim_end：如果同时指定，使用duration
    """
    # 解析时间参数
    trim_start_seconds = parse_time(trim_start) if trim_start else None
    trim_end_seconds = parse_time(trim_end) if trim_end else None
    duration_seconds = parse_time(duration) if duration else None
    
    # 如果指定了duration，计算实际的输出时长
    # 如果指定了trim_end，需要先获取视频总时长
    output_duration = None
    if duration_seconds:
        output_duration = duration_seconds
        print(f"输出视频时长: {duration_seconds}秒（从开头开始）")
    elif trim_end_seconds:
        # 获取视频总时长
        cmd_probe = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(input_path)
        ]
        result_probe = subprocess.run(cmd_probe, capture_output=True, text=True)
        try:
            total_duration = float(result_probe.stdout.strip())
            # 计算实际输出时长：总时长 - trim_end - trim_start
            output_duration = total_duration - trim_end_seconds
            if trim_start_seconds:
                output_duration -= trim_start_seconds
            if output_duration <= 0:
                print(f"警告: 计算的输出时长({output_duration:.2f}秒) <= 0，请检查trim_start和trim_end参数")
                output_duration = None
            else:
                print(f"跳过结尾: {trim_end_seconds}秒，输出时长: {output_duration:.2f}秒")
        except:
            print(f"警告: 无法获取视频总时长，将忽略trim_end参数")
    
    cmd = ['ffmpeg']
    
    # 如果指定了trim_start，在输入文件之前添加-ss参数（更高效，会跳过解码）
    if trim_start_seconds:
        cmd.extend(['-ss', str(trim_start_seconds)])
        print(f"跳过开头: {trim_start_seconds}秒")
    
    cmd.extend(['-i', str(input_path)])
    
    # 如果有外部音频文件，添加音频输入
    if audio_path:
        if not Path(audio_path).exists():
            print(f"警告: 音频文件不存在: {audio_path}")
            audio_path = None
        else:
            # 如果指定了trim_start，外部音频也需要跳过相同的时间
            if trim_start_seconds:
                cmd.extend(['-ss', str(trim_start_seconds)])
            cmd.extend(['-i', str(audio_path)])
            print(f"使用外部音频: {audio_path}")
    
    cmd.append('-y')  # -y 应该在所有输入之后
    
    # 如果指定了输出时长，添加-t参数（在输入之后，输出之前）
    if output_duration:
        cmd.extend(['-t', str(output_duration)])
    
    # 构建滤镜链
    # 注意：delogo应该在crop之前执行，这样坐标不需要调整
    filters = []
    
    # 遮挡LOGO（使用delogo模糊）- 在裁剪之前执行，坐标按原始视频计算
    if delogo_config:
        filters.append(f'delogo={delogo_config}')
    
    # 裁剪字幕区域（从底部裁掉）
    if crop_region:
        # 如果启用自动等比例裁剪，先计算等比例裁剪参数
        if auto_aspect:
            crop_region = calculate_aspect_crop(crop_region)
            print(f"自动等比例裁剪: {crop_region}")
        filters.append(f'crop={crop_region}')
    
    # 缩放（可选）
    filters.extend(build_scale_filter(scale_to, scale_mode))
    
    # 如果有滤镜，添加到命令
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    # 输出设置 - 根据文件扩展名自动选择编码器
    output_ext = output_path.suffix.lower()
    
    # 流映射：如果有外部音频，需要明确映射视频和音频流
    if mute:
        # 静音模式：去掉音频轨道
        cmd.append('-an')
        print("静音模式：去除音频轨道")
    elif audio_path:
        # 映射视频流（第一个输入的视频）和音频流（第二个输入的音频）
        cmd.extend(['-map', '0:v:0'])  # 第一个输入（视频）的视频流
        cmd.extend(['-map', '1:a:0'])  # 第二个输入（音频）的音频流
        # 音频编码设置
        if output_ext == '.webm':
            cmd.extend(['-c:a', 'libopus'])
        else:
            cmd.extend(['-c:a', 'aac', '-b:a', '192k'])  # MP4使用AAC编码
    else:
        # 没有外部音频，使用视频中的音频（如果存在）
        if output_ext == '.webm':
            cmd.extend(['-c:a', 'libopus'])
        else:
            cmd.extend(['-c:a', 'copy'])  # 音频不变
    
    # 视频编码设置
    if output_ext == '.webm':
        # WebM格式使用VP9编码
        cmd.extend([
            '-c:v', 'libvpx-vp9',
            '-crf', '30',  # VP9的CRF范围是0-63，30是高质量
            '-b:v', '0',  # 使用CRF模式
        ])
    else:
        # 默认使用H.264 (MP4)
        cmd.extend([
            '-c:v', 'libx264',
            '-crf', '18',  # 高质量
            '-preset', 'slow',
        ])
    
    cmd.append(str(output_path))
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    
    print(f"✓ 清洗完成: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='清洗原番视频（去字幕/LOGO）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 推荐（后续用超分）：裁剪+等比例裁剪，不缩放，保持1742x980，最后用RealESRGAN超分
  python3 clean_video.py -i input.mp4 -o output.mp4 \\
    --crop 1920:980:0:0 --auto-aspect --delogo 1650:40:200:150
  
  # 中等分辨率（适合超分）：裁剪+等比例裁剪，缩放到1600x900，最后超分到1080p
  python3 clean_video.py -i input.mp4 -o output.mp4 \\
    --crop 1920:980:0:0 --auto-aspect --delogo 1650:40:200:150 --scale-medium
  
  # 直接到1080p（不用超分时）：裁剪+等比例裁剪，直接缩放到1920x1080
  python3 clean_video.py -i input.mp4 -o output.mp4 \\
    --crop 1920:980:0:0 --auto-aspect --delogo 1650:40:200:150 --scale 1920:1080
  
  # 自动检测字幕区域（可能不准确）
  python3 clean_video.py -i input.mp4 -o output.mp4 --auto-detect
  
  # 使用外部音频文件（视频没有音频轨道时）
  python3 clean_video.py -i input.mp4 -o output.mp4 --audio input_audio.webm
  # 或者不指定--audio，会自动查找 input_audio.webm/mp3/wav 等

注意:
  - crop参数格式: w:h:x:y
    * w=宽度, h=保留的高度, x=0(不左右裁), y=0(从顶部保留)
    * 例如 1920:980:0:0 表示保留1920x980，裁掉底部100px
  - delogo参数格式: x:y:w:h (LOGO的坐标和尺寸)
  - 建议在原始视频上统一清洗，而不是在切分后的片段上
  - 如果视频没有音频轨道，使用--audio指定外部音频文件
  - 如果不指定--audio，会自动查找同目录下带_audio后缀的文件（如 input_audio.webm）
        """
    )
    parser.add_argument('--input', '-i', required=True, help='输入视频路径')
    parser.add_argument('--output', '-o', required=True, help='输出视频路径')
    parser.add_argument('--crop', help='手动指定裁剪区域 (w:h:x:y)，例如 1920:980:0:0')
    parser.add_argument('--delogo', help='LOGO区域 (x:y:w:h)，例如 1650:40:200:150')
    parser.add_argument('--scale', help='缩放目标分辨率 (w:h)，例如 1920:1080，默认不缩放')
    parser.add_argument('--scale-mode', choices=['stretch', 'letterbox'], default='stretch',
                        help='缩放模式: stretch(拉伸) 或 letterbox(等比例+补边)')
    parser.add_argument('--auto-aspect', action='store_true',
                       help='自动等比例裁剪：从已裁剪区域左右再裁一点，保持16:9比例（推荐，避免黑边）')
    parser.add_argument('--scale-medium', action='store_true',
                       help='缩放到中等分辨率（1600x900），适合后续超分处理')
    parser.add_argument('--auto-detect', action='store_true', help='自动检测字幕区域（可能不准确）')
    parser.add_argument('--audio', help='外部音频文件路径（如果视频没有音频轨道）。如果不指定，会自动查找同目录下带_audio后缀的文件')
    parser.add_argument('--trim-start', help='跳过开头的时间，格式 "HH:MM:SS" 或秒数（如 "5" 或 "00:00:05"），用于跳过下集预告、序幕等')
    parser.add_argument('--trim-end', help='跳过结尾的时间，格式 "HH:MM:SS" 或秒数（如 "30" 或 "00:00:30"），从视频末尾往前去掉指定时长')
    parser.add_argument('--duration', help='输出视频的总时长，格式 "HH:MM:SS" 或秒数（如 "600"），从开头开始截取指定时长（优先级高于--trim-end）')
    parser.add_argument('--mute', action='store_true', help='去除音频轨道（静音），用于后续统一添加BGM和旁白')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return 1
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    crop_region = args.crop
    if args.auto_detect and not crop_region:
        print("自动检测字幕区域...")
        crop_region = detect_subtitle_region(input_path)
        if crop_region:
            print(f"检测到裁剪区域: {crop_region}")
    
    # 如果指定了中等分辨率，覆盖scale参数
    scale_to = args.scale
    if args.scale_medium:
        scale_to = "1600:900"
        print("使用中等分辨率: 1600x900（适合后续超分）")
    
    # 处理音频文件：如果未指定，尝试自动查找_audio后缀的文件
    audio_path = args.audio
    if not audio_path:
        # 自动查找：例如 input.mp4 -> input_audio.mp3 或 input_audio.webm
        input_stem = input_path.stem
        input_dir = input_path.parent
        # 尝试常见的音频扩展名
        for ext in ['.webm', '.mp3', '.wav', '.m4a', '.aac']:
            audio_candidate = input_dir / f"{input_stem}_audio{ext}"
            if audio_candidate.exists():
                audio_path = str(audio_candidate)
                print(f"自动检测到音频文件: {audio_path}")
                break
    
    if clean_video(input_path, output_path, crop_region, args.delogo,
                   scale_to, args.scale_mode, args.auto_aspect, audio_path, 
                   args.trim_start, args.trim_end, args.duration, args.mute):
        print(f"✓ 清洗成功: {output_path}")
        return 0
    else:
        print(f"✗ 清洗失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())
