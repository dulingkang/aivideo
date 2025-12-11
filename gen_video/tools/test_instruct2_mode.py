#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CosyVoice2 instruct2 模式
instruct2 模式不需要 prompt_text 与音频内容精确匹配，可能避免生成时长异常问题
"""

import sys
from pathlib import Path

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_generator import TTSGenerator

def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    print("="*60)
    print("测试 CosyVoice2 instruct2 模式")
    print("="*60)
    
    # 初始化 TTS Generator
    print("\n[1/3] 初始化 TTS Generator...")
    tts_gen = TTSGenerator(str(config_path))
    print("✅ TTS Generator 初始化成功")
    
    # 测试文本
    test_text = "欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念。"
    output_path = Path(__file__).parent.parent / "outputs" / "test_instruct2_mode.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[2/3] 生成测试音频...")
    print(f"  生成文本: {test_text} (长度: {len(test_text)} 字符)")
    print(f"  使用模式: instruct2")
    print(f"  指令文本: 用专业、清晰、亲切的科普解说风格，语速适中，吐字清晰")
    
    # 使用 instruct2 模式
    tts_gen.synthesize_cosyvoice(
        test_text,
        str(output_path),
        mode="instruct2",
        instruction="用专业、清晰、亲切的科普解说风格，语速适中，吐字清晰",
        text_frontend=True
    )
    
    print(f"\n[3/3] 检查生成结果...")
    if output_path.exists():
        try:
            import librosa
            audio, sr = librosa.load(str(output_path), sr=None)
            duration = len(audio) / sr
            print(f"  ✅ 生成成功: {output_path}")
            print(f"  生成时长: {duration:.2f} 秒")
            
            # 估算正常时长（按3.5字/秒）
            expected_duration = len(test_text) / 3.5
            if duration > expected_duration * 2:
                print(f"  ⚠️  警告: 生成时长异常（正常应该约 {expected_duration:.2f} 秒）")
            elif duration < expected_duration * 0.5:
                print(f"  ⚠️  警告: 生成时长过短（正常应该约 {expected_duration:.2f} 秒）")
            else:
                print(f"  ✅ 生成时长正常（预期约 {expected_duration:.2f} 秒）")
        except Exception as e:
            print(f"  ⚠️  无法检查音频时长: {e}")
    else:
        print(f"  ❌ 生成失败: 文件不存在")
        return 1
    
    print(f"\n✅ 测试完成！")
    return 0

if __name__ == '__main__':
    sys.exit(main())

