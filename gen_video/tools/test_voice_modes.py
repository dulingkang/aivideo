#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同的声音克隆模式
对比 zero_shot 和 instruct2 模式的效果
"""

import sys
from pathlib import Path

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts_generator import TTSGenerator


def test_zero_shot():
    """测试 zero_shot 模式"""
    print("="*60)
    print("测试 zero_shot 模式")
    print("="*60)
    
    tts = TTSGenerator("config.yaml")
    
    test_text = "大家好，我是科普主持人。今天我们来聊聊科学的奥秘。"
    output_path = Path(__file__).parent.parent / "outputs" / "test_zero_shot.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"生成文本: {test_text}")
        tts.generate(
            text=test_text,
            output_path=str(output_path),
            mode="zero_shot"
        )
        
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"✅ zero_shot 模式生成成功: {output_path}")
            print(f"   文件大小: {file_size / 1024:.2f} KB")
            return True
        else:
            print(f"❌ 语音文件未生成")
            return False
    except Exception as e:
        print(f"❌ zero_shot 模式失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_instruct2():
    """测试 instruct2 模式"""
    print("\n" + "="*60)
    print("测试 instruct2 模式")
    print("="*60)
    
    tts = TTSGenerator("config.yaml")
    
    test_text = "大家好，我是科普主持人。今天我们来聊聊科学的奥秘。"
    instruction = "用专业、清晰、亲切的科普解说风格"
    output_path = Path(__file__).parent.parent / "outputs" / "test_instruct2.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"生成文本: {test_text}")
        print(f"指令文本: {instruction}")
        tts.generate(
            text=test_text,
            output_path=str(output_path),
            mode="instruct2",
            instruction=instruction
        )
        
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"✅ instruct2 模式生成成功: {output_path}")
            print(f"   文件大小: {file_size / 1024:.2f} KB")
            return True
        else:
            print(f"❌ 语音文件未生成")
            return False
    except Exception as e:
        print(f"❌ instruct2 模式失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("声音克隆模式对比测试")
    print("="*60)
    
    results = []
    
    # 测试 zero_shot
    results.append(("zero_shot", test_zero_shot()))
    
    # 测试 instruct2
    results.append(("instruct2", test_instruct2()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    for mode, result in results:
        status = "✅ 成功" if result else "❌ 失败"
        print(f"{mode}: {status}")
    
    print("\n请试听生成的两个音频文件，选择效果更好的模式。")
    print("文件位置:")
    print("  - zero_shot: outputs/test_zero_shot.wav")
    print("  - instruct2: outputs/test_instruct2.wav")


if __name__ == '__main__':
    main()

