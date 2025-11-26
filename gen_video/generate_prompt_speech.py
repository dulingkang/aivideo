#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 CosyVoice2 的 prompt_speech 参考音频
使用 ChatTTS 生成一个女性知性解说风格的参考音频
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_prompt_speech():
    """使用 ChatTTS 生成参考音频"""
    try:
        import ChatTTS
        
        print("="*60)
        print("生成 CosyVoice2 prompt_speech 参考音频")
        print("="*60)
        print()
        
        # 初始化 ChatTTS
        print("[1/3] 初始化 ChatTTS...")
        chat = ChatTTS.Chat()
        chat.load(compile=False)
        print("✓ ChatTTS 加载成功")
        
        # 生成参考文本（女性知性解说风格）
        prompt_text = "希望你以后能够做的比我还好呦。"
        print(f"\n[2/3] 生成参考音频...")
        print(f"参考文本: {prompt_text}")
        print("风格: 女性声音，温柔知性，情绪平稳，语调亲切自然，解说语气，吐字清晰，语速适中")
        
        # 生成音频
        texts = [prompt_text]
        wavs = chat.infer(
            texts,
            use_decoder=True,
            skip_refine_text=False,
        )
        
        # 保存音频
        output_dir = Path("assets/prompts")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "zero_shot_prompt.wav"
        
        print(f"\n[3/3] 保存音频文件...")
        import soundfile as sf
        sf.write(str(output_path), wavs[0], 24000)
        
        # 转换为 16kHz（CosyVoice2 需要）
        print("转换为 16kHz（CosyVoice2 要求）...")
        import librosa
        audio, sr = librosa.load(str(output_path), sr=24000)
        audio_16k = librosa.resample(audio, orig_sr=24000, target_sr=16000)
        sf.write(str(output_path), audio_16k, 16000)
        
        file_size = output_path.stat().st_size / 1024  # KB
        duration = len(audio_16k) / 16000  # 秒
        
        print(f"\n✓ 参考音频生成成功！")
        print(f"  文件路径: {output_path}")
        print(f"  文件大小: {file_size:.2f} KB")
        print(f"  时长: {duration:.2f} 秒")
        print(f"  采样率: 16000 Hz")
        print(f"\n提示: 请将以下配置添加到 config.yaml:")
        print(f"  cosyvoice:")
        print(f"    prompt_speech: {output_path}")
        print(f"    prompt_text: \"{prompt_text}\"")
        
        return str(output_path)
        
    except ImportError:
        print("✗ 错误: 未安装 ChatTTS")
        print("  请运行: pip install ChatTTS")
        return None
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    output_path = generate_prompt_speech()
    if output_path:
        print(f"\n✓ 完成！参考音频已保存到: {output_path}")
    else:
        print("\n✗ 生成失败，请检查错误信息")
        sys.exit(1)
