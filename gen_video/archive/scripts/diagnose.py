#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频生成流程诊断脚本
快速检查配置和生成文件，定位问题
"""

import yaml
from pathlib import Path
import sys
import os

def check_config():
    """检查配置文件"""
    print("=" * 60)
    print("配置检查")
    print("=" * 60)
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ 无法读取 config.yaml: {e}")
        return None
    
    # TTS 配置
    print("\n[1] TTS 配置:")
    tts = cfg.get('tts', {})
    engine = tts.get('engine', 'unknown')
    print(f"  引擎: {engine}", end='')
    if engine == 'chattts':
        print(" ✓")
    elif engine == 'piper':
        print(" ✗ 应该是 chattts（piper 是男声）")
    else:
        print(f" ⚠️ 未知引擎")
    
    print(f"  语速: {tts.get('speed', 1.0)}")
    style = tts.get('style_prompt', '')
    print(f"  Style Prompt: {style[:60]}...")
    if '女性' in style or '女声' in style:
        print("    ✓ 包含女性声音描述")
    else:
        print("    ✗ 未包含女性声音描述")
    
    # BGM 配置
    print("\n[2] BGM 配置:")
    bgm = cfg.get('composition', {}).get('bgm', {})
    enabled = bgm.get('enabled', False)
    print(f"  启用: {enabled}", end='')
    if enabled:
        print(" ✓")
    else:
        print(" ✗ 应该启用")
    
    if enabled:
        default = bgm.get('tracks', {}).get('default', {})
        if default:
            bgm_path = default.get('path', '')
            if bgm_path:
                p = Path(bgm_path)
                if not p.is_absolute():
                    p = Path('.') / p
                exists = p.exists()
                print(f"  默认BGM: {p}")
                print(f"    存在: {'✓' if exists else '✗ 文件不存在'}")
    
    print(f"  BGM 音量: {bgm.get('volume', 0.3)}")
    print(f"  配音音量: {cfg.get('composition', {}).get('audio_volume', 1.0)}")
    
    return cfg

def check_files(output_name='lingjie_ep5_auto_v2'):
    """检查生成的文件"""
    print("\n" + "=" * 60)
    print("生成文件检查")
    print("=" * 60)
    
    output_dir = Path('outputs') / output_name
    
    # 音频文件
    audio_file = output_dir / 'audio.wav'
    print(f"\n[1] 音频文件: {audio_file}")
    if audio_file.exists():
        try:
            import soundfile as sf
            data, sr = sf.read(str(audio_file))
            duration = len(data) / sr
            channels = 1 if data.ndim == 1 else data.shape[1]
            print(f"  ✓ 存在")
            print(f"    时长: {duration:.2f} 秒")
            print(f"    采样率: {sr} Hz", end='')
            if sr == 24000:
                print(" ✓ (ChatTTS 默认)")
            else:
                print(f" ⚠️ (ChatTTS 默认是 24000)")
            print(f"    声道: {channels}", end='')
            if channels == 1:
                print(" ✓ (ChatTTS 默认单声道)")
            else:
                print(f" ⚠️ (ChatTTS 默认单声道)")
            print(f"    文件大小: {audio_file.stat().st_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"  ✗ 无法读取: {e}")
    else:
        print(f"  ✗ 不存在")
    
    # 字幕文件
    subtitle_file = output_dir / 'subtitle.srt'
    print(f"\n[2] 字幕文件: {subtitle_file}")
    if subtitle_file.exists():
        size = subtitle_file.stat().st_size
        print(f"  ✓ 存在 ({size} 字节)")
        try:
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    print(f"    行数: {len(lines)}")
                    # 检查是否是占位字幕
                    content = ''.join(lines)
                    if '占位' in content or len(lines) < 10:
                        print("    ⚠️ 可能是占位字幕（时间轴估算）")
                    else:
                        print("    ✓ 可能是实际识别的字幕")
        except Exception as e:
            print(f"  ✗ 无法读取: {e}")
    else:
        print(f"  ✗ 不存在")
    
    # 视频文件
    video_file = output_dir / f'{output_name}.mp4'
    print(f"\n[3] 视频文件: {video_file}")
    if video_file.exists():
        size_mb = video_file.stat().st_size / 1024 / 1024
        print(f"  ✓ 存在 ({size_mb:.1f} MB)")
        
        # 检查视频中的音频流
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-hide_banner', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=codec_name,sample_rate,channels', '-of', 'default=noprint_wrappers=1',
                 str(video_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("    音频流信息:")
                for line in result.stdout.strip().split('\n'):
                    if line:
                        print(f"      {line}")
            else:
                print("    ⚠️ 无法获取音频流信息")
        except Exception as e:
            print(f"    ⚠️ 检查音频流失败: {e}")
    else:
        print(f"  ✗ 不存在")

def check_tts_engine():
    """检查 TTS 引擎实际加载情况"""
    print("\n" + "=" * 60)
    print("TTS 引擎检查")
    print("=" * 60)
    
    try:
        from tts_generator import TTSGenerator
        tts = TTSGenerator('config.yaml')
        print(f"配置的引擎: {tts.engine}")
        print(f"实际使用的函数: {tts.synthesize_func.__name__}")
        
        if tts.engine == 'chattts':
            if hasattr(tts, 'chattts'):
                print("✓ ChatTTS 已加载")
            else:
                print("✗ ChatTTS 未加载")
        elif tts.engine == 'piper':
            print("⚠️ 使用的是 Piper（默认男声）")
            print("  建议改为 chattts 以获得女声")
    except Exception as e:
        print(f"✗ 无法检查 TTS 引擎: {e}")
        import traceback
        traceback.print_exc()

def main():
    output_name = sys.argv[1] if len(sys.argv) > 1 else 'lingjie_ep5_auto_v2'
    
    print("\n" + "=" * 60)
    print("视频生成流程诊断")
    print("=" * 60)
    print(f"输出目录: outputs/{output_name}\n")
    
    cfg = check_config()
    check_files(output_name)
    check_tts_engine()
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
    print("\n建议:")
    print("1. 如果 TTS 引擎是 piper，请检查 config.yaml 中 tts.engine 是否为 chattts")
    print("2. 如果音频不存在，请重新运行步骤 4（生成配音）")
    print("3. 如果视频中没有配音，请检查合成步骤的日志")
    print("4. 如果 BGM 不存在，请检查 background_music/ 目录下的文件")

if __name__ == '__main__':
    main()

