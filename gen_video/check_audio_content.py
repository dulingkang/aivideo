#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查音频文件信息，帮助确认 prompt_text
"""

import sys
from pathlib import Path

def check_audio_info(audio_path: str):
    """检查音频文件信息"""
    print(f"检查音频文件: {audio_path}")
    print("=" * 60)
    
    try:
        # 尝试使用 torchaudio
        try:
            import torchaudio
            data, sr = torchaudio.load(audio_path)
            duration = data.shape[-1] / sr
            print(f"✓ 采样率: {sr} Hz")
            print(f"✓ 时长: {duration:.2f} 秒")
            print(f"✓ 形状: {data.shape}")
            print(f"✓ 声道数: {data.shape[0]}")
        except ImportError:
            # 如果没有 torchaudio，尝试使用 soundfile
            try:
                import soundfile as sf
                data, sr = sf.read(audio_path)
                duration = len(data) / sr
                print(f"✓ 采样率: {sr} Hz")
                print(f"✓ 时长: {duration:.2f} 秒")
                if len(data.shape) > 1:
                    print(f"✓ 形状: {data.shape}, 声道数: {data.shape[1]}")
                else:
                    print(f"✓ 单声道")
            except ImportError:
                print("⚠️  需要安装 torchaudio 或 soundfile 来检查音频信息")
                print("   安装命令: pip install torchaudio 或 pip install soundfile")
                return
    
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        return
    
    print("\n" + "=" * 60)
    print("重要提示:")
    print("1. 请播放此音频文件，确认前5秒的实际内容")
    print("2. prompt_text 必须与音频内容完全匹配（包括标点符号）")
    print("3. prompt_text 应该只匹配前5秒的内容（约15-25字符）")
    print("4. 如果音频内容是 '欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念。'")
    print("   那么 prompt_text 应该是: '欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念。'")
    print("5. 请根据实际音频内容更新 config.yaml 中的 prompt_text")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "assets/prompts/haoran_prompt_5s.wav"
    
    check_audio_info(audio_path)

