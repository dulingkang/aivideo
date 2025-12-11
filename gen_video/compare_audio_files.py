#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比音频文件，排查问题
"""

import sys
from pathlib import Path

def compare_audio_files(audio1_path: str, audio2_path: str):
    """对比两个音频文件的格式和属性"""
    print("=" * 60)
    print("对比音频文件")
    print("=" * 60)
    
    try:
        from cosyvoice.utils.file_utils import load_wav
        import torch
        
        # 加载两个音频文件
        print(f"\n1. 加载音频文件 1: {audio1_path}")
        try:
            audio1 = load_wav(audio1_path, 16000)
            print(f"   ✓ 加载成功")
            print(f"   形状: {audio1.shape}")
            print(f"   数据类型: {audio1.dtype}")
            print(f"   时长: {audio1.shape[-1] / 16000:.2f} 秒")
            print(f"   最小值: {audio1.min().item():.4f}")
            print(f"   最大值: {audio1.max().item():.4f}")
            print(f"   平均值: {audio1.mean().item():.4f}")
            print(f"   标准差: {audio1.std().item():.4f}")
        except Exception as e:
            print(f"   ✗ 加载失败: {e}")
            audio1 = None
        
        print(f"\n2. 加载音频文件 2: {audio2_path}")
        try:
            audio2 = load_wav(audio2_path, 16000)
            print(f"   ✓ 加载成功")
            print(f"   形状: {audio2.shape}")
            print(f"   数据类型: {audio2.dtype}")
            print(f"   时长: {audio2.shape[-1] / 16000:.2f} 秒")
            print(f"   最小值: {audio2.min().item():.4f}")
            print(f"   最大值: {audio2.max().item():.4f}")
            print(f"   平均值: {audio2.mean().item():.4f}")
            print(f"   标准差: {audio2.std().item():.4f}")
        except Exception as e:
            print(f"   ✗ 加载失败: {e}")
            audio2 = None
        
        # 对比分析
        if audio1 is not None and audio2 is not None:
            print(f"\n3. 对比分析:")
            print(f"   形状是否相同: {audio1.shape == audio2.shape}")
            print(f"   数据类型是否相同: {audio1.dtype == audio2.dtype}")
            
            # 检查音频范围
            audio1_range = audio1.max() - audio1.min()
            audio2_range = audio2.max() - audio2.min()
            print(f"   音频1范围: {audio1_range.item():.4f}")
            print(f"   音频2范围: {audio2_range.item():.4f}")
            
            # 检查是否有异常值
            if audio1.max() > 1.0 or audio1.min() < -1.0:
                print(f"   ⚠️  音频1超出正常范围 [-1, 1]")
            if audio2.max() > 1.0 or audio2.min() < -1.0:
                print(f"   ⚠️  音频2超出正常范围 [-1, 1]")
            
            # 检查音量
            audio1_volume = audio1.abs().mean()
            audio2_volume = audio2.abs().mean()
            print(f"   音频1平均音量: {audio1_volume.item():.4f}")
            print(f"   音频2平均音量: {audio2_volume.item():.4f}")
            
            if audio2_volume < 0.01:
                print(f"   ⚠️  音频2音量太小，可能导致问题")
            if audio2_volume > 0.5:
                print(f"   ⚠️  音频2音量太大，可能导致问题")
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("   请确保 CosyVoice 已正确安装")


if __name__ == "__main__":
    # 对比之前正常工作的音频和现在的音频
    audio1 = "assets/prompts/zero_shot_prompt_clean.wav"  # 之前正常工作的
    audio2 = "assets/prompts/haoran_prompt_5s.wav"  # 现在有问题的
    
    if len(sys.argv) > 1:
        audio2 = sys.argv[1]
    
    compare_audio_files(audio1, audio2)

