#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备参考音频脚本
用于预处理参考音频，确保格式正确
"""

import os
import sys
import argparse
import soundfile as sf
import librosa
import numpy as np


def process_audio(input_path: str, output_path: str, target_sr: int = 24000):
    """
    处理参考音频
    
    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        target_sr: 目标采样率（默认24kHz）
    """
    print(f"处理音频: {input_path} -> {output_path}")
    
    # 加载音频
    try:
        audio, sr = librosa.load(input_path, sr=None, mono=False)
    except Exception as e:
        print(f"错误: 无法加载音频: {e}")
        return False
    
    print(f"原始采样率: {sr} Hz")
    print(f"原始声道数: {audio.shape[0] if len(audio.shape) > 1 else 1}")
    
    # 转换为单声道
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
        print("转换为单声道")
    
    # 重采样到目标采样率
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        print(f"重采样到 {target_sr} Hz")
    
    # 标准化音量（可选）
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95  # 保留一些余量
        print("标准化音量")
    
    # 保存音频
    try:
        sf.write(output_path, audio, target_sr, format='WAV', subtype='PCM_16')
        print(f"✓ 音频已保存: {output_path}")
        
        # 显示文件信息
        file_size = os.path.getsize(output_path) / 1024  # KB
        duration = len(audio) / target_sr
        print(f"文件大小: {file_size:.2f} KB")
        print(f"时长: {duration:.2f} 秒")
        
        return True
    except Exception as e:
        print(f"错误: 无法保存音频: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="处理参考音频")
    parser.add_argument("--input", type=str, required=True, help="输入音频路径")
    parser.add_argument("--output", type=str, help="输出音频路径（默认：input_processed.wav）")
    parser.add_argument("--sample-rate", type=int, default=24000, help="目标采样率（默认24000）")
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_processed.wav"
    
    # 处理音频
    success = process_audio(args.input, str(output_path), args.sample_rate)
    
    if success:
        print("\n✓ 处理完成！")
        print(f"参考音频已准备好: {output_path}")
        print("\n使用方法:")
        print(f"1. 将参考音频路径添加到 config.yaml:")
        print(f"   reference_audio: \"{output_path}\"")
        print("2. 运行 TTS 生成器将自动使用参考音频")
    else:
        print("\n✗ 处理失败")
        sys.exit(1)


if __name__ == "__main__":
    from pathlib import Path
    main()

