#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 TTS 生成器（使用参考音频）
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tts_generator import TTSGenerator

def main():
    print("=" * 60)
    print("测试 TTS 生成器（声音克隆）")
    print("=" * 60)
    
    # 初始化生成器
    print("\n1. 初始化 TTS 生成器...")
    try:
        generator = TTSGenerator("config.yaml")
        print("✓ TTS 生成器初始化成功")
    except Exception as e:
        print(f"✗ TTS 生成器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查参考音频
    print("\n2. 检查参考音频...")
    ref_audio_path = generator.config['tts'].get('reference_audio')
    if ref_audio_path:
        abs_path = os.path.abspath(ref_audio_path)
        if os.path.exists(abs_path):
            file_size = os.path.getsize(abs_path) / (1024 * 1024)  # MB
            print(f"✓ 参考音频找到: {abs_path}")
            print(f"  文件大小: {file_size:.2f} MB")
        else:
            print(f"⚠ 参考音频不存在: {abs_path}")
            print("  将使用默认声音")
    else:
        print("⚠ 未配置参考音频，将使用默认声音")
    
    # 测试文本
    test_text = "当韩立再次睁开双眼，发现自己身处一片茫茫黄沙之中。"
    output_path = "outputs/test_tts_output.wav"
    
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    # 生成语音
    print(f"\n3. 生成语音...")
    print(f"   文本: {test_text}")
    print(f"   输出: {output_path}")
    
    try:
        generator.generate(test_text, output_path)
        print("✓ 语音生成成功")
        
        # 检查输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"✓ 输出文件存在: {output_path}")
            print(f"  文件大小: {file_size:.2f} KB")
        else:
            print(f"✗ 输出文件不存在: {output_path}")
    except Exception as e:
        print(f"✗ 语音生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"\n请播放音频文件检查效果: {output_path}")

if __name__ == "__main__":
    main()


