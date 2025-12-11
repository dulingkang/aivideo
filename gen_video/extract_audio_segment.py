#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取音频前N秒片段
用于 CosyVoice2 zero_shot 模式，需要较短的参考音频（5-15秒）
"""

import argparse
import librosa
import soundfile as sf
from pathlib import Path


def extract_audio_segment(input_path: str, output_path: str, duration: float = 5.0, target_sr: int = 16000):
    """
    提取音频前N秒片段并转换为16kHz单声道
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        duration: 提取时长（秒），默认5秒
        target_sr: 目标采样率，默认16kHz（CosyVoice2要求）
    """
    print(f"提取音频片段: {input_path} -> {output_path}")
    print(f"  提取时长: {duration} 秒")
    print(f"  目标采样率: {target_sr} Hz")
    
    # 加载音频
    try:
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        print(f"  原始采样率: {sr} Hz")
        
        # 转换为单声道
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
            print("  转换为单声道")
        
        # 重采样到目标采样率
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            print(f"  重采样到 {target_sr} Hz")
        
        # 提取前N秒
        max_samples = int(target_sr * duration)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            print(f"  提取前 {duration} 秒")
        else:
            print(f"  ⚠️  音频总时长 {len(audio)/target_sr:.2f} 秒，小于 {duration} 秒")
        
        # 标准化音量
        max_val = abs(audio).max()
        if max_val > 0:
            if max_val > 1.0:
                audio = audio / max_val
                print("  归一化音频数值范围到 [-1, 1]")
            elif max_val < 0.1:
                audio = audio / max_val * 0.95
                print(f"  放大音频音量（原始最大值: {max_val:.4f}）")
        
        # 保存为16kHz WAV格式（PCM_16）
        sf.write(output_path, audio, target_sr, format='WAV', subtype='PCM_16')
        
        # 显示文件信息
        file_size = Path(output_path).stat().st_size / 1024  # KB
        actual_duration = len(audio) / target_sr
        print(f"✓ 音频已保存: {output_path}")
        print(f"  文件大小: {file_size:.2f} KB")
        print(f"  实际时长: {actual_duration:.2f} 秒")
        print(f"  采样率: {target_sr} Hz")
        
        return True
    except Exception as e:
        print(f"✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="提取音频前N秒片段")
    parser.add_argument("--input", type=str, required=True, help="输入音频路径")
    parser.add_argument("--output", type=str, help="输出音频路径（默认：input_5s.wav）")
    parser.add_argument("--duration", type=float, default=5.0, help="提取时长（秒），默认5秒")
    parser.add_argument("--sample-rate", type=int, default=16000, help="目标采样率，默认16kHz")
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_5s.wav"
    
    # 提取音频片段
    success = extract_audio_segment(
        args.input,
        str(output_path),
        duration=args.duration,
        target_sr=args.sample_rate
    )
    
    if success:
        print("\n✓ 处理完成！")
        print(f"参考音频已准备好: {output_path}")
        print("\n使用方法:")
        print(f"1. 将参考音频路径添加到 config.yaml:")
        print(f"   prompt_speech: \"{output_path}\"")
        print("2. 确保 prompt_text 与音频前5秒的内容完全匹配")
    else:
        print("\n✗ 处理失败")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

