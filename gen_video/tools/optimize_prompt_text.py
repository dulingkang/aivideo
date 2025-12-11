#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化 prompt_text
根据音频时长和内容，生成合适的 prompt_text
"""

import argparse
import subprocess
from pathlib import Path


def get_audio_duration(audio_path: Path) -> float:
    """获取音频时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'stream=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0


def estimate_text_length(duration: float, chars_per_second: float = 3.5) -> int:
    """
    根据音频时长估算文本长度
    
    Args:
        duration: 音频时长（秒）
        chars_per_second: 每秒字符数（中文约3.5字/秒）
    
    Returns:
        估算的文本长度（字符数）
    """
    return int(duration * chars_per_second)


def suggest_prompt_text(current_text: str, duration: float) -> str:
    """
    根据音频时长建议合适的 prompt_text
    
    Args:
        current_text: 当前文本
        duration: 音频时长（秒）
    
    Returns:
        建议的文本
    """
    estimated_length = estimate_text_length(duration)
    
    # 如果当前文本太长，建议使用前N个字符
    if len(current_text) > estimated_length * 1.5:
        # 文本太长，建议使用前部分
        suggested = current_text[:estimated_length]
        return suggested
    elif len(current_text) < estimated_length * 0.5:
        # 文本太短，保持原样
        return current_text
    else:
        # 文本长度合适
        return current_text


def main():
    parser = argparse.ArgumentParser(description='优化 prompt_text，使其与音频时长匹配')
    parser.add_argument('--audio', type=str, required=True,
                       help='参考音频文件路径')
    parser.add_argument('--text', type=str, required=True,
                       help='当前的 prompt_text')
    parser.add_argument('--output', type=str, default=None,
                       help='输出建议的 prompt_text 到文件（可选）')
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ 音频文件不存在: {audio_path}")
        return 1
    
    # 获取音频时长
    duration = get_audio_duration(audio_path)
    if duration == 0.0:
        print(f"⚠️  无法获取音频时长")
        return 1
    
    print("="*60)
    print("prompt_text 优化建议")
    print("="*60)
    print(f"\n音频文件: {audio_path.name}")
    print(f"音频时长: {duration:.2f} 秒")
    print(f"当前文本: {args.text}")
    print(f"当前文本长度: {len(args.text)} 字符")
    
    # 估算合适的文本长度
    estimated_length = estimate_text_length(duration)
    print(f"\n估算的合适长度: {estimated_length} 字符（按3.5字/秒计算）")
    
    # 建议的文本
    suggested_text = suggest_prompt_text(args.text, duration)
    
    print(f"\n建议的 prompt_text:")
    print(f"  {suggested_text}")
    print(f"  长度: {len(suggested_text)} 字符")
    
    # 计算匹配度
    length_ratio = len(suggested_text) / estimated_length if estimated_length > 0 else 0
    if 0.8 <= length_ratio <= 1.2:
        print(f"  ✅ 长度匹配度良好（{length_ratio:.2f}）")
    elif length_ratio > 1.2:
        print(f"  ⚠️  文本可能过长（{length_ratio:.2f}），建议缩短")
    else:
        print(f"  ⚠️  文本可能过短（{length_ratio:.2f}），但可以使用")
    
    # 保存到文件
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(suggested_text)
        print(f"\n✅ 建议的文本已保存到: {output_path}")
    
    print("\n重要提示:")
    print("1. prompt_text 必须与音频中的实际内容完全一致")
    print("2. 如果音频内容与文本不匹配，会导致声音质量差")
    print("3. 建议使用音频前5-7秒对应的文本内容")
    print("4. 可以使用 WhisperX 识别音频的实际内容")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

