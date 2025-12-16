#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 prompt_speech 文件是否可用
"""

import os
from pathlib import Path
import soundfile as sf

def check_prompt_speech():
    prompt_path = Path("assets/prompts/zero_shot_prompt.wav")
    
    print("="*60)
    print("检查 prompt_speech 文件")
    print("="*60)
    print()
    
    if not prompt_path.exists():
        print(f"✗ 文件不存在: {prompt_path}")
        return False
    
    try:
        data, sr = sf.read(str(prompt_path))
        duration = len(data) / sr
        
        print(f"✓ 文件存在: {prompt_path}")
        print(f"  采样率: {sr} Hz")
        print(f"  时长: {duration:.2f} 秒")
        print(f"  声道数: {data.ndim}")
        print(f"  文件大小: {prompt_path.stat().st_size / 1024:.2f} KB")
        
        # 检查是否符合要求
        checks = []
        if sr == 16000:
            checks.append(("✓", "采样率为 16kHz（符合要求）"))
        else:
            checks.append(("✗", f"采样率为 {sr}Hz（需要 16kHz）"))
        
        if duration >= 3 and duration <= 60:
            checks.append(("✓", f"时长 {duration:.1f}秒（合适）"))
        else:
            checks.append(("⚠", f"时长 {duration:.1f}秒（建议 3-60 秒）"))
        
        print()
        print("检查结果:")
        for status, msg in checks:
            print(f"  {status} {msg}")
        
        if all(c[0] == "✓" for c in checks):
            print()
            print("✓ 文件符合要求，可以在 config.yaml 中使用")
            return True
        else:
            print()
            print("⚠ 文件可能不符合要求，请检查")
            return False
            
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        return False

if __name__ == "__main__":
    success = check_prompt_speech()
    if success:
        print()
        print("提示: 配置已在 config.yaml 中设置")
        print("  cosyvoice:")
        print("    prompt_speech: assets/prompts/zero_shot_prompt.wav")
    sys.exit(0 if success else 1)
