#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试双环境配置
验证主环境和 CosyVoice 子环境是否正常工作
"""

import sys
from pathlib import Path

# 添加 gen_video 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_main_environment():
    """测试主环境"""
    print("="*60)
    print("测试主环境")
    print("="*60)
    
    try:
        import transformers
        print(f"✅ transformers 版本: {transformers.__version__}")
        
        # 检查是否支持 Flux.1
        try:
            from transformers import modeling_layers
            print("✅ transformers.modeling_layers 可用（支持 Flux.1）")
        except (ImportError, AttributeError):
            print("⚠️  transformers.modeling_layers 不可用（可能不支持 Flux.1）")
        
        import torch
        print(f"✅ torch 版本: {torch.__version__}")
        
        return True
    except Exception as e:
        print(f"❌ 主环境测试失败: {e}")
        return False

def test_cosyvoice_subprocess():
    """测试 CosyVoice 子进程调用"""
    print("\n" + "="*60)
    print("测试 CosyVoice 子进程调用")
    print("="*60)
    
    try:
        from tts_generator import TTSGenerator
        
        print("\n[1/3] 初始化 TTSGenerator...")
        tts = TTSGenerator("config.yaml")
        
        if not hasattr(tts, 'cosyvoice_use_subprocess') or not tts.cosyvoice_use_subprocess:
            print("❌ 子进程模式未启用")
            print("   请在 config.yaml 中设置 use_subprocess: true")
            return False
        
        print("✅ 子进程模式已启用")
        print(f"   Python: {tts.cosyvoice_subprocess_python}")
        print(f"   脚本: {tts.cosyvoice_subprocess_script}")
        
        print("\n[2/3] 测试子进程调用...")
        test_text = "大家好，我是科普哥哥。这是一个测试。"
        output_path = "outputs/test_dual_env.wav"
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        tts.generate(test_text, output_path)
        
        print("\n[3/3] 检查输出文件...")
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / 1024  # KB
            print(f"✅ 音频文件已生成: {output_path}")
            print(f"   文件大小: {file_size:.2f} KB")
            return True
        else:
            print(f"❌ 音频文件未生成: {output_path}")
            return False
            
    except Exception as e:
        print(f"❌ CosyVoice 子进程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("双环境配置测试")
    print("="*60)
    print()
    
    # 测试主环境
    main_ok = test_main_environment()
    
    # 测试 CosyVoice 子进程
    cosyvoice_ok = test_cosyvoice_subprocess()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"主环境: {'✅ 正常' if main_ok else '❌ 异常'}")
    print(f"CosyVoice 子进程: {'✅ 正常' if cosyvoice_ok else '❌ 异常'}")
    
    if main_ok and cosyvoice_ok:
        print("\n✅ 双环境配置正常，可以开始使用")
        return 0
    else:
        print("\n❌ 配置有问题，请检查")
        return 1

if __name__ == "__main__":
    sys.exit(main())

