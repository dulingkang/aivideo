#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice 子进程包装器
用于在独立环境中运行 CosyVoice，避免 transformers 版本冲突

使用方法：
    python cosyvoice_subprocess_wrapper.py --text "要生成的文本" --output output.wav
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any


def main():
    """CosyVoice 子进程入口"""
    parser = argparse.ArgumentParser(description="CosyVoice 子进程包装器")
    parser.add_argument("--text", type=str, required=True, help="要生成的文本")
    parser.add_argument("--output", type=str, required=True, help="输出音频文件路径")
    parser.add_argument("--config", type=str, help="配置文件路径（JSON格式）")
    parser.add_argument("--prompt-speech", type=str, help="参考音频路径")
    parser.add_argument("--prompt-text", type=str, help="参考文本")
    parser.add_argument("--mode", type=str, default="zero_shot", help="模式：zero_shot, instruct2, cross_lingual")
    parser.add_argument("--instruction", type=str, help="指令文本（instruct2模式）")
    parser.add_argument("--text-frontend", type=str, default="True", help="是否启用文本前端处理（True/False）")
    parser.add_argument("--seed", type=int, help="随机种子")
    
    args = parser.parse_args()
    
    # 加载配置（如果提供）
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 合并命令行参数和配置文件
    prompt_speech = args.prompt_speech or config.get('prompt_speech')
    prompt_text = args.prompt_text or config.get('prompt_text')
    mode = args.mode or config.get('mode', 'zero_shot')
    instruction = args.instruction or config.get('instruction')
    text_frontend = args.text_frontend if args.text_frontend is not None else config.get('text_frontend', True)
    seed = args.seed or config.get('seed')
    
    try:
        # 导入 CosyVoice（在独立环境中）
        import sys
        cosyvoice_repo_path = Path(__file__).parent.parent.parent / "CosyVoice"
        if cosyvoice_repo_path.exists():
            if str(cosyvoice_repo_path) not in sys.path:
                sys.path.insert(0, str(cosyvoice_repo_path))
            matcha_path = cosyvoice_repo_path / "third_party" / "Matcha-TTS"
            if matcha_path.exists() and str(matcha_path) not in sys.path:
                sys.path.append(str(matcha_path))
        else:
            raise ImportError(f"CosyVoice 仓库不存在: {cosyvoice_repo_path}")
        
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        import torchaudio
        import torch
        
        # 加载模型配置（从配置文件或命令行参数）
        # 注意：子进程模式下，配置应该通过命令行参数传递，而不是从文件读取
        # 但为了兼容性，也支持从配置文件读取
        model_path = config.get('model_path')
        model_name = config.get('model_name', 'CosyVoice2-0.5B')
        use_cosyvoice2 = config.get('use_cosyvoice2', True)
        
        # 确定模型路径
        if not model_path:
            # 尝试从 CosyVoice 仓库的 pretrained_models 目录加载
            model_dir = cosyvoice_repo_path / "pretrained_models" / model_name
            if model_dir.exists():
                model_path = str(model_dir)
            else:
                # 使用 ModelScope ID
                modelscope_id_map = {
                    'CosyVoice2-0.5B': 'iic/CosyVoice2-0.5B',
                    'CosyVoice-300M': 'iic/CosyVoice-300M',
                    'CosyVoice-300M-SFT': 'iic/CosyVoice-300M-SFT',
                    'CosyVoice-300M-Instruct': 'iic/CosyVoice-300M-Instruct',
                }
                model_path = modelscope_id_map.get(model_name, f'iic/{model_name}')
        
        print(f"加载 CosyVoice 模型: {model_path}")
        if use_cosyvoice2:
            cosyvoice = CosyVoice2(
                model_path,
                load_jit=False,
                load_trt=False,
                load_vllm=False,
                fp16=False
            )
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice
            cosyvoice = CosyVoice(
                model_path,
                load_jit=False,
                load_trt=False,
                fp16=False
            )
        
        # 设置种子
        if seed is not None:
            from cosyvoice.utils.common import set_all_random_seed
            set_all_random_seed(seed)
        
        # 加载参考音频
        prompt_speech_16k = None
        if prompt_speech:
            prompt_speech_path = Path(prompt_speech)
            if prompt_speech_path.exists():
                prompt_speech_16k = load_wav(str(prompt_speech_path), 16000)
            else:
                print(f"⚠ 警告: 参考音频文件不存在: {prompt_speech}")
        
        # 生成语音
        print(f"生成语音: {args.text[:50]}...")
        
        if mode == "zero_shot":
            # 检查参数（Tensor 不能直接用 if not 判断）
            import torch
            if prompt_speech_16k is not None:
                # 检查 Tensor 是否为空
                if isinstance(prompt_speech_16k, torch.Tensor):
                    if prompt_speech_16k.numel() == 0:
                        prompt_speech_16k = None
            if prompt_speech_16k is None or not prompt_text:
                raise ValueError("zero_shot 模式需要 prompt_speech 和 prompt_text")
            
            for i, result in enumerate(cosyvoice.inference_zero_shot(
                args.text,
                prompt_text,
                prompt_speech_16k,
                stream=False,
                text_frontend=text_frontend
            )):
                # 确保输出目录存在
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # 使用绝对路径保存
                abs_output_path = output_path.resolve()
                torchaudio.save(str(abs_output_path), result['tts_speech'], cosyvoice.sample_rate)
                print(f"✓ 语音已保存: {abs_output_path}")
                break
        
        elif mode == "instruct2":
            # 检查参数（Tensor 不能直接用 if not 判断）
            import torch
            if prompt_speech_16k is not None:
                if isinstance(prompt_speech_16k, torch.Tensor):
                    if prompt_speech_16k.numel() == 0:
                        prompt_speech_16k = None
            if prompt_speech_16k is None or not instruction:
                raise ValueError("instruct2 模式需要 prompt_speech 和 instruction")
            
            for i, result in enumerate(cosyvoice.inference_instruct2(
                args.text,
                instruction,
                prompt_speech_16k,
                stream=False,
                text_frontend=text_frontend
            )):
                # 确保输出目录存在
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # 使用绝对路径保存
                abs_output_path = output_path.resolve()
                torchaudio.save(str(abs_output_path), result['tts_speech'], cosyvoice.sample_rate)
                print(f"✓ 语音已保存: {abs_output_path}")
                break
        
        elif mode == "cross_lingual":
            # 检查参数（Tensor 不能直接用 if not 判断）
            import torch
            if prompt_speech_16k is not None:
                if isinstance(prompt_speech_16k, torch.Tensor):
                    if prompt_speech_16k.numel() == 0:
                        prompt_speech_16k = None
            if prompt_speech_16k is None:
                raise ValueError("cross_lingual 模式需要 prompt_speech")
            
            for i, result in enumerate(cosyvoice.inference_cross_lingual(
                args.text,
                prompt_speech_16k,
                stream=False,
                text_frontend=text_frontend
            )):
                # 确保输出目录存在
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # 使用绝对路径保存
                abs_output_path = output_path.resolve()
                torchaudio.save(str(abs_output_path), result['tts_speech'], cosyvoice.sample_rate)
                print(f"✓ 语音已保存: {abs_output_path}")
                break
        
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        # 返回成功
        sys.exit(0)
        
    except Exception as e:
        error_msg = f"❌ CosyVoice 生成失败: {e}"
        print(error_msg, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # 确保错误信息被输出到 stderr，以便主进程捕获
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()

