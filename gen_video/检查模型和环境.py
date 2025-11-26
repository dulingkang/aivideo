#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 CosyVoice2 模型和环境配置
"""

import sys
from pathlib import Path

# 添加路径
cosyvoice_repo_path = Path(__file__).parent.parent / "CosyVoice"
if str(cosyvoice_repo_path) not in sys.path:
    sys.path.insert(0, str(cosyvoice_repo_path))

matcha_path = cosyvoice_repo_path / "third_party" / "Matcha-TTS"
if matcha_path.exists() and str(matcha_path) not in sys.path:
    sys.path.append(str(matcha_path))

print("="*70)
print("检查 CosyVoice2 模型和环境")
print("="*70)
print()

# 1. 检查依赖库版本
print("[1] 检查依赖库版本")
print("-" * 70)
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"  ✗ 无法检查 PyTorch: {e}")

try:
    import torchaudio
    print(f"  TorchAudio: {torchaudio.__version__}")
except Exception as e:
    print(f"  ✗ 无法检查 TorchAudio: {e}")

try:
    import numpy as np
    print(f"  NumPy: {np.__version__}")
except Exception as e:
    print(f"  ✗ 无法检查 NumPy: {e}")
print()

# 2. 检查模型文件
print("[2] 检查模型文件")
print("-" * 70)
model_path = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/cosyvoice/CosyVoice2-0.5B")
if model_path.exists():
    print(f"  ✓ 模型路径存在: {model_path}")
    
    # 检查关键文件
    required_files = [
        "cosyvoice2.yaml",
        "llm.pt",
        "flow.pt",
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v2.onnx",
        "spk2info.pt",
    ]
    
    for file_name in required_files:
        file_path = model_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"    ✓ {file_name} ({size:.2f} MB)")
        else:
            print(f"    ✗ {file_name} 不存在")
else:
    print(f"  ✗ 模型路径不存在: {model_path}")
    print("  将使用 ModelScope ID: iic/CosyVoice2-0.5B")
print()

# 3. 检查模型加载
print("[3] 检查模型加载")
print("-" * 70)
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    
    model_path_str = str(model_path) if model_path.exists() else "iic/CosyVoice2-0.5B"
    print(f"  加载模型: {model_path_str}")
    
    cosyvoice = CosyVoice2(
        model_path_str,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False
    )
    
    print("  ✓ 模型加载成功")
    print(f"    采样率: {cosyvoice.sample_rate} Hz")
    
    # 检查模型属性
    if hasattr(cosyvoice, 'model'):
        print(f"    模型类型: {type(cosyvoice.model).__name__}")
    
    if hasattr(cosyvoice, 'frontend'):
        print(f"    前端类型: {type(cosyvoice.frontend).__name__}")
    
except Exception as e:
    print(f"  ✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
print()

# 4. 测试简单的音频生成
print("[4] 测试简单音频生成")
print("-" * 70)
try:
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    
    # 使用官方示例
    official_prompt = cosyvoice_repo_path / "asset" / "zero_shot_prompt.wav"
    
    if official_prompt.exists():
        print(f"  使用官方示例音频: {official_prompt}")
        
        prompt_speech_16k = load_wav(str(official_prompt), 16000)
        print(f"  ✓ 参考音频加载成功")
        print(f"    形状: {prompt_speech_16k.shape}")
        print(f"    数值范围: [{prompt_speech_16k.min():.4f}, {prompt_speech_16k.max():.4f}]")
        
        # 简单测试
        test_text = "你好"
        prompt_text = "希望你以后能够做的比我还好呦。"
        
        print(f"  测试文本: {test_text}")
        print(f"  生成中...")
        
        audio_chunks = []
        for i, result in enumerate(cosyvoice.inference_zero_shot(
            test_text,
            prompt_text,
            prompt_speech_16k,
            stream=False,
            text_frontend=False
        )):
            if isinstance(result, dict) and 'tts_speech' in result:
                audio_chunks.append(result['tts_speech'])
            else:
                audio_chunks.append(result)
        
        if audio_chunks:
            first_audio = audio_chunks[0]
            if isinstance(first_audio, dict) and 'tts_speech' in first_audio:
                audio_tensor = first_audio['tts_speech']
            else:
                audio_tensor = first_audio
            
            print(f"  ✓ 生成成功")
            print(f"    形状: {audio_tensor.shape}")
            print(f"    数值范围: [{audio_tensor.min():.4f}, {audio_tensor.max():.4f}]")
            
            # 计算能量
            if isinstance(audio_tensor, torch.Tensor):
                rms = torch.sqrt(torch.mean(audio_tensor ** 2)).item()
            else:
                import numpy as np
                audio_np = np.array(audio_tensor)
                rms = np.sqrt(np.mean(audio_np ** 2))
            
            print(f"    RMS 能量: {rms:.6f}")
            
            if rms < 0.001:
                print("    ⚠ 警告: 音频能量太低！")
                print("    可能的原因:")
                print("      1. 模型输出异常")
                print("      2. 文本太短导致模型无法生成有效音频")
                print("      3. prompt_text 与参考音频不匹配")
            else:
                print("    ✓ 音频能量正常")
            
            # 保存测试文件
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > audio_tensor.shape[1]:
                audio_tensor = audio_tensor.T
            
            output_path = "simple_test.wav"
            torchaudio.save(output_path, audio_tensor, cosyvoice.sample_rate)
            print(f"    已保存: {output_path}")
    else:
        print(f"  ✗ 官方示例音频不存在: {official_prompt}")
        
except Exception as e:
    print(f"  ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
print()

# 5. 检查可能的配置问题
print("[5] 检查可能的配置问题")
print("-" * 70)
print("  如果官方示例也不正常，可能的原因：")
print("    1. 模型文件损坏或不完整")
print("    2. 依赖库版本不匹配")
print("    3. CUDA/GPU 问题（如果使用 GPU）")
print("    4. 模型量化问题（如果使用了量化）")
print("    5. 内存不足导致模型加载不完整")
print()
print("  建议：")
print("    1. 重新下载模型文件")
print("    2. 检查依赖库版本是否匹配官方要求")
print("    3. 尝试使用 CPU 模式测试（如果当前使用 GPU）")
print("    4. 检查系统日志是否有错误信息")
print()

