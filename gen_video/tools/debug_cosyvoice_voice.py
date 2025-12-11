#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice 声音克隆问题排查工具
对比测试不同的 prompt_speech 和 prompt_text 配置
"""

import argparse
import sys
from pathlib import Path
import json

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_generator import TTSGenerator


def transcribe_audio(audio_path: Path, config_path: str = "config.yaml") -> str:
    """使用 WhisperX 识别音频内容"""
    try:
        from subtitle_generator import SubtitleGenerator
        subtitle_gen = SubtitleGenerator(config_path)
        print(f"  使用 WhisperX 识别音频...")
        result = subtitle_gen.transcribe(str(audio_path))
        
        if result and 'text' in result:
            return result['text']
        elif result and 'segments' in result:
            text = ' '.join([seg.get('text', '') for seg in result['segments']])
            return text
        return None
    except Exception as e:
        print(f"  ⚠️  WhisperX 识别失败: {e}")
        return None


def test_voice_config(tts_gen: TTSGenerator, prompt_speech: Path, prompt_text: str, 
                      test_text: str, output_name: str, mode: str = "zero_shot", 
                      text_frontend: bool = True, speed: float = 1.0):
    """测试特定的声音配置"""
    print(f"\n{'='*60}")
    print(f"测试配置: {output_name}")
    print(f"{'='*60}")
    print(f"  prompt_speech: {prompt_speech.name}")
    print(f"  prompt_text: {prompt_text[:50]}..." if len(prompt_text) > 50 else f"  prompt_text: {prompt_text}")
    print(f"  prompt_text 长度: {len(prompt_text)} 字符")
    print(f"  测试文本: {test_text[:50]}..." if len(test_text) > 50 else f"  测试文本: {test_text}")
    print(f"  测试文本长度: {len(test_text)} 字符")
    print(f"  mode: {mode}")
    print(f"  text_frontend: {text_frontend}")
    print(f"  speed: {speed}")
    
    # 检查 prompt_speech 是否存在
    if not prompt_speech.exists():
        print(f"  ❌ prompt_speech 文件不存在: {prompt_speech}")
        return False
    
    # 识别音频实际内容
    print(f"\n  识别 prompt_speech 实际内容...")
    actual_text = transcribe_audio(prompt_speech)
    if actual_text:
        print(f"  ✅ 识别结果: {actual_text}")
        print(f"  识别文本长度: {len(actual_text)} 字符")
        
        # 检查匹配度
        if prompt_text in actual_text or actual_text in prompt_text:
            print(f"  ✅ prompt_text 与音频内容匹配")
        else:
            print(f"  ⚠️  警告: prompt_text 与音频内容可能不匹配")
            print(f"     建议: 使用识别结果的前5-7秒对应文本作为 prompt_text")
    
    # 生成测试音频
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"debug_{output_name}.wav"
    
    print(f"\n  生成测试音频...")
    try:
        # 临时修改配置
        original_prompt_speech = tts_gen.cosyvoice_prompt_speech
        original_prompt_text = tts_gen.cosyvoice_prompt_text
        original_mode = tts_gen.cosyvoice_mode
        original_text_frontend = tts_gen.cosyvoice_text_frontend
        
        # 确保 prompt_speech 是字符串路径
        tts_gen.cosyvoice_prompt_speech = str(prompt_speech)
        tts_gen.cosyvoice_prompt_text = prompt_text
        tts_gen.cosyvoice_mode = mode
        tts_gen.cosyvoice_text_frontend = text_frontend
        
        # 重新加载 prompt_speech
        tts_gen._prompt_speech_16k = None
        
        # 生成音频
        tts_gen.synthesize_cosyvoice(
            test_text,
            str(output_path),
            mode=mode,
            text_frontend=text_frontend,
            speed=speed
        )
        
        # 恢复配置
        tts_gen.cosyvoice_prompt_speech = original_prompt_speech
        tts_gen.cosyvoice_prompt_text = original_prompt_text
        tts_gen.cosyvoice_mode = original_mode
        tts_gen.cosyvoice_text_frontend = original_text_frontend
        tts_gen._prompt_speech_16k = None
        
        print(f"  ✅ 生成成功: {output_path}")
        
        # 检查生成音频时长
        try:
            import librosa
            audio, sr = librosa.load(str(output_path), sr=None)
            duration = len(audio) / sr
            print(f"  生成音频时长: {duration:.2f} 秒")
            
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
        
        return True
    except Exception as e:
        print(f"  ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='CosyVoice 声音克隆问题排查工具')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--test-text', type=str, 
                       default='欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念。',
                       help='测试文本（默认：25字符）')
    parser.add_argument('--all', action='store_true',
                       help='运行所有测试配置')
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 1
    
    print("="*60)
    print("CosyVoice 声音克隆问题排查")
    print("="*60)
    
    # 初始化 TTS Generator
    print("\n[1/3] 初始化 TTS Generator...")
    try:
        tts_gen = TTSGenerator(str(config_path))
        print("✅ TTS Generator 初始化成功")
    except Exception as e:
        print(f"❌ TTS Generator 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 定义测试配置
    assets_dir = Path(__file__).parent.parent / "assets" / "prompts"
    voices_dir = Path(__file__).parent.parent / "assets" / "voices"
    
    test_configs = [
        {
            "name": "zero_shot_clean_original",
            "prompt_speech": assets_dir / "zero_shot_prompt_clean.wav",
            "prompt_text": "大家好, 我是云卷仙音，今天我们要继续讲述凡人修仙转的故事，在这个故事中,韩立经历了无数的挑",
            "description": "之前正常工作的配置（zero_shot_prompt_clean.wav）"
        },
        {
            "name": "haoran_clean",
            "prompt_speech": assets_dir / "haoran_prompt_clean.wav",
            "prompt_text": "欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念，讲解火星探索的科学价值和未来展望。",
            "description": "Haoran 清理后的音频（需要识别实际内容）"
        },
        {
            "name": "haoran_5s",
            "prompt_speech": assets_dir / "haoran_prompt_5s.wav",
            "prompt_text": "欢迎来到本期知识探索，我们将继续介绍黑洞的基本概念。",
            "description": "Haoran 前5秒音频（需要识别实际内容）"
        },
    ]
    
    # 如果指定了 --all，运行所有测试
    if args.all:
        print(f"\n[2/3] 运行所有测试配置...")
        results = []
        for config in test_configs:
            if config["prompt_speech"].exists():
                success = test_voice_config(
                    tts_gen,
                    config["prompt_speech"],
                    config["prompt_text"],
                    args.test_text,
                    config["name"],
                    mode="zero_shot",
                    text_frontend=True,
                    speed=1.0
                )
                results.append((config["name"], success))
            else:
                print(f"\n⚠️  跳过测试 {config['name']}: 文件不存在 {config['prompt_speech']}")
                results.append((config["name"], False))
        
        # 总结
        print(f"\n{'='*60}")
        print("测试总结")
        print(f"{'='*60}")
        for name, success in results:
            status = "✅ 成功" if success else "❌ 失败"
            print(f"  {name}: {status}")
        
        print(f"\n请手动检查 outputs/debug_*.wav 文件，对比生成效果")
        print(f"建议:")
        print(f"1. 如果 zero_shot_clean_original 正常，说明问题在于 haoran 音频或 prompt_text")
        print(f"2. 如果所有配置都异常，说明可能是代码问题")
        print(f"3. 使用识别出的实际音频内容更新 prompt_text")
    else:
        # 只运行第一个测试（之前正常工作的配置）
        print(f"\n[2/3] 运行对比测试（使用之前正常工作的配置）...")
        config = test_configs[0]
        if config["prompt_speech"].exists():
            test_voice_config(
                tts_gen,
                config["prompt_speech"],
                config["prompt_text"],
                args.test_text,
                config["name"],
                mode="zero_shot",
                text_frontend=True,
                speed=1.0
            )
        else:
            print(f"❌ 测试文件不存在: {config['prompt_speech']}")
            return 1
        
        print(f"\n✅ 测试完成，请检查 outputs/debug_{config['name']}.wav")
        print(f"\n如果这个配置正常，说明问题在于 haoran 音频或 prompt_text 不匹配")
        print(f"如果这个配置也不正常，说明可能是代码改动导致的问题")
        print(f"\n运行 --all 参数可以测试所有配置")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

