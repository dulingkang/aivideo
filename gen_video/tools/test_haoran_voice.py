#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试"浩然"主持人声音
验证声音克隆配置是否正确
"""

import sys
from pathlib import Path

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_generator import TTSGenerator


def test_haoran_voice():
    """测试浩然声音"""
    print("="*60)
    print("测试浩然主持人声音")
    print("="*60)
    
    # 初始化TTS生成器
    print("\n1. 初始化TTS生成器...")
    try:
        tts = TTSGenerator("config.yaml")
        print("   ✅ TTS生成器初始化成功")
    except Exception as e:
        print(f"   ❌ TTS生成器初始化失败: {e}")
        return False
    
    # 检查配置
    print("\n2. 检查配置...")
    config = tts.config.get('tts', {}).get('cosyvoice', {})
    prompt_speech = config.get('prompt_speech', '')
    prompt_text = config.get('prompt_text', '')
    
    print(f"   参考音频: {prompt_speech}")
    print(f"   参考文本: {prompt_text[:50]}...")
    
    # 检查文件是否存在
    if Path(prompt_speech).exists():
        print(f"   ✅ 参考音频文件存在")
    else:
        print(f"   ❌ 参考音频文件不存在: {prompt_speech}")
        return False
    
    # 测试生成语音
    print("\n3. 测试生成语音...")
    test_text = "大家好，我是科普主持人。今天我们来聊聊科学的奥秘。"
    
    output_path = Path(__file__).parent.parent / "outputs" / "test_haoran_voice.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"   生成文本: {test_text}")
        tts.generate(
            text=test_text,
            output_path=str(output_path)
        )
        
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"   ✅ 语音生成成功: {output_path}")
            print(f"   文件大小: {file_size / 1024:.2f} KB")
            return True
        else:
            print(f"   ❌ 语音文件未生成")
            return False
            
    except Exception as e:
        print(f"   ❌ 语音生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_haoran_voice()
    if success:
        print("\n🎉 测试通过！浩然声音配置成功。")
        sys.exit(0)
    else:
        print("\n❌ 测试失败，请检查配置。")
        sys.exit(1)

