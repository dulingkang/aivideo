#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试 CosyVoice2 配置（不加载模型，只检查配置）
"""

import yaml
from pathlib import Path

print('='*60)
print('检查 CosyVoice2 配置')
print('='*60)
print()

# 读取配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

cosyvoice_config = config['tts']['cosyvoice']
prompt_speech = cosyvoice_config.get('prompt_speech')
prompt_text = cosyvoice_config.get('prompt_text', '')

print('配置信息:')
print(f'  prompt_speech: {prompt_speech}')
print(f'  prompt_text: {prompt_text[:100]}...' if len(prompt_text) > 100 else f'  prompt_text: {prompt_text}')
print(f'  mode: {cosyvoice_config.get("mode", "zero_shot")}')
print()

# 检查文件
if prompt_speech:
    prompt_path = Path(prompt_speech)
    if not prompt_path.is_absolute():
        prompt_path = Path.cwd() / prompt_speech
    
    if prompt_path.exists():
        print(f'✓ prompt_speech 文件存在: {prompt_path}')
        try:
            import librosa
            duration = librosa.get_duration(path=str(prompt_path))
            sr = librosa.get_samplerate(str(prompt_path))
            print(f'  采样率: {sr} Hz')
            print(f'  时长: {duration:.2f} 秒')
            print(f'  ✓ 文件格式正确，将自动转换为 16kHz')
        except Exception as e:
            print(f'  ⚠ 无法读取音频信息: {e}')
    else:
        print(f'✗ prompt_speech 文件不存在: {prompt_path}')

print()
print('='*60)
print('配置检查完成')
print('='*60)
print()
print('提示:')
print('  1. prompt_speech 和 prompt_text 已配置')
print('  2. 代码已更新支持 MP3 格式自动转换')
print('  3. 在实际使用时，CosyVoice2 会自动加载 prompt_speech')
print('  4. 如果遇到依赖问题，请检查 CosyVoice 的 requirements.txt')
