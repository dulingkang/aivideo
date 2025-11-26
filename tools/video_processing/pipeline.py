#!/usr/bin/env python3
"""
主流程脚本：整合所有步骤，自动化处理原番视频
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

def run_step(step_name, cmd_args, check=True):
    """运行单个处理步骤"""
    print(f"\n{'='*60}")
    print(f"步骤: {step_name}")
    print(f"{'='*60}")
    
    # 实时输出，不捕获（这样可以看到详细的错误信息）
    result = subprocess.run(cmd_args)
    
    if check and result.returncode != 0:
        print(f"\n✗ 步骤失败: {step_name} (退出码: {result.returncode})")
        print(f"  提示: 查看上方的错误信息以了解失败原因")
        return False
    
    print(f"\n✓ 步骤完成: {step_name}")
    return True

def process_episode(episode_id, raw_video_path, base_output_dir, 
                   clean=True, split=True, transcribe=True, align=True,
                   extract=True, describe=True,
                   skip_existing=False, crop_region=None, delogo_config=None,
                   scale_to=None, scale_mode='stretch', auto_aspect=False, scale_medium=False,
                   audio_path=None, subtitle_method='whisperx', trim_start=None,
                   trim_end=None, duration=None, mute=False):
    """
    处理单集视频的完整流程
    
    Args:
        episode_id: 集号（例如 170）
        raw_video_path: 原始视频路径
        base_output_dir: 基础输出目录
        clean: 是否清洗
        split: 是否切分
        extract: 是否提取关键帧
        describe: 是否生成描述
        skip_existing: 跳过已存在的文件
    """
    base_output_dir = Path(base_output_dir)
    episode_dir = base_output_dir / f"episode_{episode_id}"
    
    # 步骤0: 提取字幕（必须在清洗之前，因为清洗会裁剪掉字幕）
    subtitle_json = None
    if transcribe:
        subtitle_json = episode_dir / "subtitles.json"
        if skip_existing and subtitle_json.exists():
            print(f"跳过已存在的字幕文件: {subtitle_json}")
            subtitle_json = str(subtitle_json)
        else:
            # 从原始视频提取字幕（清洗前）
            if subtitle_method == 'ocr':
                # 使用OCR提取字幕
                print("使用OCR方法提取字幕（从原始视频）...")
                transcribe_result = run_step(
                    "OCR提取字幕",
                    ['python3', str(Path(__file__).parent / 'extract_subtitles_ocr.py'),
                     '--input', str(raw_video_path),
                     '--output', str(subtitle_json),
                     '--method', 'easyocr',  # 使用easyocr避免PaddlePaddle版本冲突
                     '--fps', '1.0'],
                    check=False
                )
            else:
                # 使用WhisperX语音识别
                transcribe_result = run_step(
                    "转写字幕",
                    ['python3', str(Path(__file__).parent / 'transcribe_subtitles.py'),
                     '--input', str(raw_video_path),
                     '--output', str(subtitle_json),
                     '--language', 'zh',
                     '--model', 'medium'],
                    check=False  # 不阻止后续流程
                )
            
            if transcribe_result and subtitle_json.exists():
                subtitle_json = str(subtitle_json)
            else:
                print("⚠ 警告: 字幕提取失败，继续处理（将只使用图像描述）")
                print("  提示: 可以稍后手动运行提取，或使用 --skip-transcribe 跳过此步骤")
                subtitle_json = None
    
    # 步骤1: 清洗视频
    if clean:
        clean_output = episode_dir / "clean" / f"episode_{episode_id}_clean.mp4"
        if skip_existing and clean_output.exists():
            print(f"跳过已存在的清洗视频: {clean_output}")
        else:
            clean_output.parent.mkdir(parents=True, exist_ok=True)
            clean_cmd = ['python3', str(Path(__file__).parent / 'clean_video.py'),
                        '--input', str(raw_video_path),
                        '--output', str(clean_output)]
            
            if crop_region:
                clean_cmd.extend(['--crop', crop_region])
            if delogo_config:
                clean_cmd.extend(['--delogo', delogo_config])
            if scale_medium:
                clean_cmd.append('--scale-medium')
            elif scale_to:
                clean_cmd.extend(['--scale', scale_to,
                                  '--scale-mode', scale_mode])
            if auto_aspect:
                clean_cmd.append('--auto-aspect')
            if audio_path:
                clean_cmd.extend(['--audio', str(audio_path)])
            if trim_start:
                clean_cmd.extend(['--trim-start', str(trim_start)])
            if trim_end:
                clean_cmd.extend(['--trim-end', str(trim_end)])
            if duration:
                clean_cmd.extend(['--duration', str(duration)])
            if mute:
                clean_cmd.append('--mute')
            if not crop_region and not delogo_config:
                clean_cmd.append('--auto-detect')
            
            if not run_step("清洗视频", clean_cmd):
                return False
        video_to_split = clean_output
    else:
        video_to_split = raw_video_path
    
    # 步骤2: 切分镜头
    if split:
        scenes_dir = episode_dir / "scenes"
        if skip_existing and scenes_dir.exists() and list(scenes_dir.glob('*.mp4')):
            print(f"跳过已存在的场景目录: {scenes_dir}")
        else:
            if not run_step(
                "切分镜头",
                ['python3', str(Path(__file__).parent / 'split_scenes.py'),
                 '--input', str(video_to_split),
                 '--output', str(scenes_dir)]
            ):
                return False
    else:
        scenes_dir = episode_dir / "scenes"
    
    # 步骤3: 对齐字幕到镜头（使用步骤0从原始视频提取的字幕）
    aligned_subtitle_json = None
    if align and subtitle_json:
        scene_list_json = episode_dir / "scene_list.json"
        aligned_subtitle_json = episode_dir / "aligned_subtitles.json"
        
        if skip_existing and aligned_subtitle_json.exists():
            print(f"跳过已存在的对齐字幕: {aligned_subtitle_json}")
            aligned_subtitle_json = str(aligned_subtitle_json)
        elif scene_list_json.exists():
            align_cmd = ['python3', str(Path(__file__).parent / 'align_subtitles.py'),
                 '--subtitles', subtitle_json,
                 '--scenes', str(scene_list_json),
                 '--output', str(aligned_subtitle_json)]
            # 如果有trim_start，传递时间偏移参数
            if trim_start:
                # 解析时间偏移（支持秒数或HH:MM:SS格式）
                def parse_time_offset(time_str):
                    """解析时间字符串，返回秒数"""
                    if not time_str:
                        return None
                    if ':' in str(time_str):
                        parts = str(time_str).split(':')
                        if len(parts) == 3:
                            hours, minutes, seconds = map(float, parts)
                            return hours * 3600 + minutes * 60 + seconds
                        elif len(parts) == 2:
                            minutes, seconds = map(float, parts)
                            return minutes * 60 + seconds
                    try:
                        return float(time_str)
                    except:
                        return None
                
                offset_seconds = parse_time_offset(trim_start)
                if offset_seconds:
                    align_cmd.extend(['--time-offset', str(offset_seconds)])
            
            if not run_step("对齐字幕到镜头", align_cmd):
                print("警告: 字幕对齐失败，继续处理（将只使用图像描述）")
                aligned_subtitle_json = None
            else:
                aligned_subtitle_json = str(aligned_subtitle_json)
        else:
            print("警告: 场景列表不存在，跳过字幕对齐")
            aligned_subtitle_json = None
    
    # 步骤5: 提取关键帧
    if extract:
        keyframes_dir = episode_dir / "keyframes"
        if skip_existing and keyframes_dir.exists() and list(keyframes_dir.glob('*.jpg')):
            print(f"跳过已存在的关键帧目录: {keyframes_dir}")
        else:
            if not run_step(
                "提取关键帧",
                ['python3', str(Path(__file__).parent / 'extract_keyframes.py'),
                 '--input', str(scenes_dir),
                 '--output', str(keyframes_dir)]
            ):
                return False
    else:
        keyframes_dir = episode_dir / "keyframes"
    
    # 步骤6: 生成描述和embedding（整合字幕）
    if describe:
        metadata_json = episode_dir / "scene_metadata.json"
        if skip_existing and metadata_json.exists():
            print(f"跳过已存在的metadata: {metadata_json}")
        else:
            describe_cmd = ['python3', str(Path(__file__).parent / 'describe_scenes.py'),
                           '--input', str(keyframes_dir),
                           '--output', str(metadata_json)]
            
            if aligned_subtitle_json:
                describe_cmd.extend(['--subtitles', aligned_subtitle_json])
                print("  将整合字幕信息到描述中")
            
            if not run_step("生成场景描述", describe_cmd):
                return False
    
    print(f"\n{'='*60}")
    print(f"✓ 集 {episode_id} 处理完成")
    print(f"  输出目录: {episode_dir}")
    print(f"{'='*60}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='原番视频处理主流程')
    parser.add_argument('--episode', '-e', required=True, type=int, help='集号')
    parser.add_argument('--input', '-i', required=True, help='原始视频路径')
    parser.add_argument('--output', '-o', default='./processed', help='输出基础目录')
    parser.add_argument('--skip-clean', action='store_true', help='跳过清洗步骤')
    parser.add_argument('--skip-split', action='store_true', help='跳过切分步骤')
    parser.add_argument('--skip-transcribe', action='store_true', help='跳过后缀转写步骤')
    parser.add_argument('--skip-align', action='store_true', help='跳过字幕对齐步骤')
    parser.add_argument('--skip-extract', action='store_true', help='跳过关键帧提取')
    parser.add_argument('--skip-describe', action='store_true', help='跳过描述生成')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已存在的文件')
    parser.add_argument('--crop', help='裁剪区域 (w:h:x:y)，例如 1920:980:0:0')
    parser.add_argument('--delogo', help='LOGO区域 (x:y:w:h)，例如 1650:40:200:150')
    parser.add_argument('--scale', help='缩放目标分辨率 (w:h)，例如 1920:1080，默认不缩放（建议最后用超分）')
    parser.add_argument('--scale-mode', choices=['stretch', 'letterbox'], default='stretch',
                        help='缩放模式: stretch(拉伸) 或 letterbox(等比例+补边)')
    parser.add_argument('--scale-medium', action='store_true',
                       help='缩放到中等分辨率（1600x900），适合后续超分处理')
    parser.add_argument('--auto-aspect', action='store_true',
                       help='自动等比例裁剪：从已裁剪区域左右再裁一点，保持16:9比例（推荐）')
    parser.add_argument('--audio', help='外部音频文件路径（如果视频没有音频轨道）。如果不指定，会自动查找同目录下带_audio后缀的文件')
    parser.add_argument('--trim-start', help='跳过开头的时间，格式 "HH:MM:SS" 或秒数（如 "5" 或 "00:00:05"），用于跳过下集预告、序幕等')
    parser.add_argument('--trim-end', help='跳过结尾的时间，格式 "HH:MM:SS" 或秒数（如 "30" 或 "00:00:30"），从视频末尾往前去掉指定时长')
    parser.add_argument('--duration', help='输出视频的总时长，格式 "HH:MM:SS" 或秒数（如 "600"），从开头开始截取指定时长（优先级高于--trim-end）')
    parser.add_argument('--subtitle-method', choices=['whisperx', 'ocr'], default='whisperx',
                       help='字幕提取方法：whisperx(语音识别) 或 ocr(文字识别，推荐用于有硬编码字幕的视频)')
    parser.add_argument('--mute', action='store_true', 
                       help='清洗视频时去除音频轨道（静音），用于后续统一添加BGM和旁白')
    
    args = parser.parse_args()
    
    raw_video = Path(args.input)
    if not raw_video.exists():
        print(f"错误: 视频文件不存在: {raw_video}")
        return 1
    
    if process_episode(
        args.episode,
        raw_video,
        args.output,
        clean=not args.skip_clean,
        split=not args.skip_split,
        transcribe=not args.skip_transcribe,
        align=not args.skip_align,
        extract=not args.skip_extract,
        describe=not args.skip_describe,
        skip_existing=args.skip_existing,
        crop_region=args.crop,
        delogo_config=args.delogo,
        scale_to=args.scale,
        scale_mode=args.scale_mode,
        auto_aspect=args.auto_aspect,
        scale_medium=args.scale_medium,
        audio_path=args.audio,
        subtitle_method=args.subtitle_method,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        duration=args.duration,
        mute=args.mute
    ):
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())

