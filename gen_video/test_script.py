#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本
检查系统是否配置正确
"""

import os
import sys
from pathlib import Path


def test_imports():
    """测试导入"""
    print("测试导入...")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("✗ PyTorch 未安装")
        return False
    
    try:
        import yaml
        print("✓ PyYAML 已安装")
    except ImportError:
        print("✗ PyYAML 未安装")
        return False
    
    try:
        import ffmpeg
        print("✓ ffmpeg-python 已安装")
    except ImportError:
        print("✗ ffmpeg-python 未安装")
        return False
    
    # 检查 FFmpeg 命令
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg: {version_line}")
        else:
            print("✗ FFmpeg 命令执行失败")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg 未安装或不在 PATH 中")
        return False
    
    return True


def test_generative_models():
    """测试 generative-models"""
    print("\n测试 generative-models...")
    
    generative_models_path = os.environ.get("GENERATIVE_MODELS_PATH", "../generative-models")
    if not os.path.exists(generative_models_path):
        print(f"⚠ generative-models 路径不存在: {generative_models_path}")
        print("  请设置 GENERATIVE_MODELS_PATH 环境变量")
        return False
    
    print(f"✓ generative-models 路径: {generative_models_path}")
    
    # 尝试导入
    sys.path.insert(0, generative_models_path)
    try:
        import sgm
        print("✓ sgm 模块可导入")
    except ImportError as e:
        print(f"⚠ sgm 模块导入失败: {e}")
        print("  可能需要安装: pip install -e . (在 generative-models 目录中)")
        return False
    
    return True


def test_tts():
    """测试 TTS"""
    print("\n测试 TTS...")
    
    try:
        import ChatTTS
        print("✓ ChatTTS 已安装")
    except ImportError:
        print("⚠ ChatTTS 未安装（可选）")
        print("  安装: pip install chattts")
    
    try:
        from TTS.api import TTS
        print("✓ Coqui TTS 已安装")
    except ImportError:
        print("⚠ Coqui TTS 未安装（可选）")
        print("  安装: pip install TTS")
    
    return True


def test_whisperx():
    """测试 WhisperX"""
    print("\n测试 WhisperX...")
    
    try:
        import whisperx
        print("✓ WhisperX 已安装")
    except ImportError:
        print("⚠ WhisperX 未安装（可选）")
        print("  安装: pip install whisperx")
        return False
    
    return True


def test_config():
    """测试配置文件"""
    print("\n测试配置文件...")
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return False
    
    print(f"✓ 配置文件存在: {config_path}")
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✓ 配置文件格式正确")
        print(f"  输入目录: {config['paths']['input_dir']}")
        print(f"  输出目录: {config['paths']['output_dir']}")
        print(f"  模型类型: {config['video']['model_type']}")
        
    except Exception as e:
        print(f"✗ 配置文件读取失败: {e}")
        return False
    
    return True


def test_paths():
    """测试路径"""
    print("\n测试路径...")
    
    # 检查脚本文件
    script_path = Path("../灵界/2.md")
    if script_path.exists():
        print(f"✓ 脚本文件存在: {script_path}")
    else:
        print(f"⚠ 脚本文件不存在: {script_path}")
    
    # 检查图像目录
    image_dir = Path("../灵界/img2/jpgsrc")
    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        print(f"✓ 图像目录存在: {image_dir}")
        print(f"  找到 {len(image_files)} 张图像")
    else:
        print(f"⚠ 图像目录不存在: {image_dir}")
    
    return True


def main():
    print("=" * 60)
    print("AI视频生成系统 - 测试脚本")
    print("=" * 60)
    
    results = []
    
    results.append(("导入测试", test_imports()))
    results.append(("generative-models", test_generative_models()))
    results.append(("TTS", test_tts()))
    results.append(("WhisperX", test_whisperx()))
    results.append(("配置文件", test_config()))
    results.append(("路径", test_paths()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ 所有测试通过！系统已准备好运行。")
    else:
        print("\n⚠ 部分测试未通过，请检查上述错误信息。")
        print("  某些功能可能无法使用，但不影响基本流程。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

