#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试清理后的参考音频效果
"""

import sys
from pathlib import Path

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_generator import TTSGenerator


def test_cleaned_voice():
    """测试清理后的声音"""
    print("="*60)
    print("测试清理后的参考音频效果")
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
    
    print(f"   参考音频: {prompt_speech}")
    if Path(prompt_speech).exists():
        file_size = Path(prompt_speech).stat().st_size
        print(f"   ✅ 参考音频文件存在 ({file_size / 1024:.2f} KB)")
    else:
        print(f"   ❌ 参考音频文件不存在")
        return False
    
    # 测试生成语音
    print("\n3. 测试生成语音（使用清理后的参考音频）...")
    test_text = "大家好，我是科普主持人。今天我们来聊聊科学的奥秘。"
    
    output_path = Path(__file__).parent.parent / "outputs" / "test_cleaned_voice.wav"
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
            print(f"\n请试听生成的文件，检查声音是否清晰。")
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
    success = test_cleaned_voice()
    if success:
        print("\n🎉 测试完成！请试听生成的声音。")
        sys.exit(0)
    else:
        print("\n❌ 测试失败，请检查配置。")
        sys.exit(1)

