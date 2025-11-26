#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段1功能测试脚本
测试 InstantID、CosyVoice、WhisperX-large-v3 三个核心模块
"""

import os
import sys
from pathlib import Path

TEST_NARRATION = "当韩立再次睁开双眼，发现自己身处一片茫茫黄沙之中。远处传来阵阵风声，卷起漫天沙尘。"

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_instantid():
    """测试 InstantID 图像生成"""
    print("\n" + "="*60)
    print("测试 1/3: InstantID 图像生成")
    print("="*60)
    
    try:
        from image_generator import ImageGenerator
        
        print("\n[1.1] 初始化 ImageGenerator...")
        gen = ImageGenerator("config.yaml")
        print("✓ ImageGenerator 初始化成功")
        
        print("\n[1.2] 加载 InstantID Pipeline...")
        gen.load_pipeline()
        print("✓ InstantID Pipeline 加载成功")
        
        print("\n[1.3] 测试生成图像...")
        # 使用精简的中国修仙类 prompt（控制在 77 token 以内）
        # 注意：代码会自动检查并确保远景关键词在开头
        # 精简策略：保留最重要的远景、场景、角色关键词，移除冗余描述
        test_prompt = (
            "(extreme wide shot:2.0), (very long shot:1.8), (distant view:1.6), "
            "xianxia desert frontier, vast dunes and oasis glow, lone male cultivator "
            "in flowing robe, mid-ground silhouette, golden evening sun haze, drifting "
            "qi mist, distant stone temples, cinematic scale, natural color palette"
        )
        output_path = Path("outputs/test_instantid.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image = gen.generate_image(test_prompt, output_path)
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ 图像生成成功: {output_path}")
            print(f"  文件大小: {file_size:.2f} MB")
            return True
        else:
            print(f"✗ 图像文件未生成: {output_path}")
            return False
            
    except Exception as e:
        print(f"✗ InstantID 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cosyvoice():
    """测试 CosyVoice TTS"""
    print("\n" + "="*60)
    print("测试 2/3: CosyVoice TTS")
    print("="*60)
    
    try:
        from tts_generator import TTSGenerator
        
        print("\n[2.1] 初始化 TTSGenerator...")
        tts = TTSGenerator("config.yaml")
        print("✓ TTSGenerator 初始化成功")
        
        print("\n[2.2] 测试生成语音...")
        output_path = "outputs/test_cosyvoice.wav"
        os.makedirs("outputs", exist_ok=True)
        
        tts.generate(TEST_NARRATION, output_path)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"✓ 语音生成成功: {output_path}")
            print(f"  文件大小: {file_size:.2f} KB")
            return True, output_path
        else:
            print(f"✗ 音频文件未生成: {output_path}")
            return False, None
            
    except Exception as e:
        print(f"✗ CosyVoice 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_whisperx(audio_path=None):
    """测试 WhisperX-large-v3 字幕生成"""
    print("\n" + "="*60)
    print("测试 3/3: WhisperX-large-v3 字幕生成")
    print("="*60)
    
    try:
        from subtitle_generator import SubtitleGenerator
        
        print("\n[3.1] 初始化 SubtitleGenerator...")
        sub = SubtitleGenerator("config.yaml")
        print("✓ SubtitleGenerator 初始化成功")
        
        # 如果没有提供音频，使用 CosyVoice 生成的音频
        if audio_path is None:
            audio_path = "outputs/test_cosyvoice.wav"
        
        if not os.path.exists(audio_path):
            print(f"⚠ 音频文件不存在: {audio_path}")
            print("  跳过字幕测试（需要先运行 CosyVoice 测试）")
            return False
        
        print(f"\n[3.2] 使用音频文件: {audio_path}")
        output_path = "outputs/test_subtitle.srt"
        
        print("\n[3.3] 生成字幕...")
        sub.generate(audio_path, output_path, narration=TEST_NARRATION)
        
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.split('\n'))
            print(f"✓ 字幕生成成功: {output_path}")
            print(f"  字幕行数: {lines}")
            print(f"\n前3行内容预览:")
            for i, line in enumerate(content.split('\n')[:3]):
                print(f"  {line}")
            return True
        else:
            print(f"✗ 字幕文件未生成: {output_path}")
            return False
            
    except Exception as e:
        print(f"✗ WhisperX 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("阶段1功能测试")
    print("="*60)
    print("\n将测试以下模块：")
    print("  1. InstantID 图像生成")
    print("  2. CosyVoice TTS")
    print("  3. WhisperX-large-v3 字幕生成")
    
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    results = {}
    
    # 测试 InstantID
    results['instantid'] = test_instantid()
    
    # 测试 CosyVoice
    success, audio_path = test_cosyvoice()
    results['cosyvoice'] = success
    
    # 测试 WhisperX（使用 CosyVoice 生成的音频）
    if success:
        results['whisperx'] = test_whisperx(audio_path)
    else:
        print("\n⚠ 跳过 WhisperX 测试（CosyVoice 测试失败）")
        results['whisperx'] = False
    
    # 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    print(f"  InstantID:     {'✓ 通过' if results['instantid'] else '✗ 失败'}")
    print(f"  CosyVoice:     {'✓ 通过' if results['cosyvoice'] else '✗ 失败'}")
    print(f"  WhisperX:      {'✓ 通过' if results['whisperx'] else '✗ 失败'}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ 所有测试通过！阶段1功能正常。")
    else:
        print("\n⚠ 部分测试失败，请检查错误信息。")
        print("\n提示:")
        if not results['instantid']:
            print("  - InstantID: 检查模型是否下载完整，参考图像路径是否正确")
        if not results['cosyvoice']:
            print("  - CosyVoice: 检查模型是否下载，网络连接是否正常")
        if not results['whisperx']:
            print("  - WhisperX: 检查模型是否下载，对齐模型和VAD模型会在首次使用时自动下载")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

