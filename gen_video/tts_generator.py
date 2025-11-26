#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS 配音生成脚本
支持 ChatTTS, OpenVoice, Coqui TTS
"""

import json
import os
import random
import re
import tempfile
import yaml
import argparse
import subprocess
import shutil
from pathlib import Path
import torch
import numpy as np
from typing import Optional, List


class TTSGenerator:
    """TTS 语音生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化 TTS 生成器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path.cwd() / self.config_path).resolve()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.tts_config = self.config['tts']
        self.engine = self.tts_config['engine']
        # 优先使用全局 TTS seed，如果没有则使用 CosyVoice 的 seed（如果存在）
        self.seed: Optional[int] = self.tts_config.get("seed")
        if self.seed is None and self.tts_config.get("cosyvoice", {}).get("seed") is not None:
            self.seed = self.tts_config["cosyvoice"]["seed"]
        self.piper_sample_rate = 22050
        self.chattts_style_prompt = self.tts_config.get("style_prompt", "")
        self.chattts_negative_prompt = self.tts_config.get("negative_prompt", "")
        
        # 加载模型
        self.model = None
        self.load_model()
        
        # 创建输出目录
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)
    
    def load_model(self):
        """加载 TTS 模型"""
        print(f"加载 TTS 引擎: {self.engine}")
        
        if self.engine == "cosyvoice":
            self.load_cosyvoice()
        elif self.engine == "chattts":
            self.load_chattts()
        elif self.engine == "openvoice":
            self.load_openvoice()
        elif self.engine == "coqui":
            self.load_coqui()
        elif self.engine == "gtts":
            self.load_gtts()
        elif self.engine == "piper":
            self.load_piper()
        else:
            raise ValueError(f"不支持的 TTS 引擎: {self.engine}")

    def load_gtts(self):
        """加载 gTTS 引擎"""
        try:
            from gtts import gTTS  # noqa: F401
            self.synthesize_func = self.synthesize_gtts
            print("✓ gTTS 可用")
        except ImportError:
            print("错误: 未安装 gTTS")
            print("安装命令: pip install gTTS")
            raise
    
    def load_cosyvoice(self):
        """加载 CosyVoice 模型（从 GitHub 仓库，根据官方 README）"""
        try:
            # 根据 GitHub README，需要添加路径
            import sys
            cosyvoice_repo_path = Path(__file__).parent.parent / "CosyVoice"
            if cosyvoice_repo_path.exists():
                if str(cosyvoice_repo_path) not in sys.path:
                    sys.path.insert(0, str(cosyvoice_repo_path))
                # 添加 third_party/Matcha-TTS 路径（根据 README）
                matcha_path = cosyvoice_repo_path / "third_party" / "Matcha-TTS"
                if matcha_path.exists() and str(matcha_path) not in sys.path:
                    sys.path.append(str(matcha_path))
                print(f"✓ 从克隆的仓库导入 CosyVoice: {cosyvoice_repo_path}")
            else:
                raise ImportError(f"CosyVoice 仓库不存在: {cosyvoice_repo_path}")
            
            # 根据 README，导入方式：from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
            
            cosyvoice_config = self.tts_config.get("cosyvoice", {})
            model_name = cosyvoice_config.get("model_name", "CosyVoice2-0.5B")
            model_path = cosyvoice_config.get("model_path")
            use_cosyvoice2 = cosyvoice_config.get("use_cosyvoice2", True)  # 默认使用 CosyVoice2
            
            # 根据 GitHub README，CosyVoice2 使用方式：
            # cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
            # CosyVoice 使用方式：
            # cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
            
            print(f"加载 CosyVoice 模型: {model_name}")
            print(f"  使用版本: {'CosyVoice2' if use_cosyvoice2 else 'CosyVoice'}")
            
            # 确定模型路径（优先级：CosyVoice仓库的pretrained_models > config指定的路径 > ModelScope ID > 相对路径）
            # 期望的 yaml 文件名
            expected_yaml = "cosyvoice2.yaml" if use_cosyvoice2 else "cosyvoice.yaml"
            
            # 1. 优先检查 CosyVoice 仓库的 pretrained_models 目录
            model_dir = cosyvoice_repo_path / "pretrained_models" / model_name
            if model_dir.exists() and (model_dir / expected_yaml).exists():
                print(f"  找到模型: {model_dir}")
                model_dir = str(model_dir)
            # 2. 检查 config 中指定的路径（如果存在且包含模型文件）
            elif model_path and os.path.exists(model_path):
                model_path_obj = Path(model_path)
                # 检查是否是模型目录本身，还是模型的父目录
                if (model_path_obj / expected_yaml).exists():
                    print(f"  使用 config 指定的模型路径: {model_path}")
                    model_dir = model_path
                elif (model_path_obj / model_name / expected_yaml).exists():
                    print(f"  使用 config 指定的模型路径: {model_path_obj / model_name}")
                    model_dir = str(model_path_obj / model_name)
                else:
                    print(f"  ⚠ config 指定的路径不存在模型文件，尝试 ModelScope ID...")
                    # 使用 ModelScope ID（官方推荐方式，会自动下载）
                    modelscope_id_map = {
                        'CosyVoice2-0.5B': 'iic/CosyVoice2-0.5B',
                        'CosyVoice-300M': 'iic/CosyVoice-300M',
                        'CosyVoice-300M-SFT': 'iic/CosyVoice-300M-SFT',
                        'CosyVoice-300M-Instruct': 'iic/CosyVoice-300M-Instruct',
                    }
                    model_dir = modelscope_id_map.get(model_name, f'iic/{model_name}')
                    print(f"  使用 ModelScope ID（首次运行会自动下载）: {model_dir}")
            # 3. 使用 ModelScope ID（官方推荐方式，会自动下载）
            else:
                modelscope_id_map = {
                    'CosyVoice2-0.5B': 'iic/CosyVoice2-0.5B',
                    'CosyVoice-300M': 'iic/CosyVoice-300M',
                    'CosyVoice-300M-SFT': 'iic/CosyVoice-300M-SFT',
                    'CosyVoice-300M-Instruct': 'iic/CosyVoice-300M-Instruct',
                }
                model_dir = modelscope_id_map.get(model_name, f'iic/{model_name}')
                print(f"  使用 ModelScope ID（首次运行会自动下载）: {model_dir}")
            
            # 加载模型
            fp16 = (self.tts_config.get("quantization", "fp16") == "fp16")
            if use_cosyvoice2:
                self.cosyvoice = CosyVoice2(
                    str(model_dir),
                    load_jit=False,
                    load_trt=False,
                    load_vllm=False,
                    fp16=fp16
                )
            else:
                self.cosyvoice = CosyVoice(
                    str(model_dir),
                    load_jit=False,
                    load_trt=False,
                    fp16=fp16
                )
            
            # 保存配置
            self.cosyvoice_auto_emotion = self.tts_config.get("auto_emotion", True)
            self.cosyvoice_default_emotion = cosyvoice_config.get("default_emotion", "中文女")
            self.cosyvoice_sample_rate = self.cosyvoice.sample_rate if hasattr(self.cosyvoice, 'sample_rate') else 24000
            self.cosyvoice_bit_depth = self.tts_config.get("bit_depth", 24)
            self.cosyvoice_stream = self.tts_config.get("stream", False)
            self.use_cosyvoice2 = use_cosyvoice2
            # CosyVoice2 配置（按照官方用法）
            self.cosyvoice_prompt_speech = cosyvoice_config.get("prompt_speech")  # 默认 prompt_speech 路径
            self.cosyvoice_prompt_text = cosyvoice_config.get("prompt_text", "希望你以后能够做的比我还好呦。")
            self.cosyvoice_mode = cosyvoice_config.get("mode", "zero_shot")
            
            self.synthesize_func = self.synthesize_cosyvoice
            print("✓ CosyVoice 加载成功")
            
        except ImportError as e:
            print("错误: 无法导入 CosyVoice")
            print("  请确保已克隆 CosyVoice 仓库:")
            print("    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git")
            print("  并安装依赖:")
            print("    pip install -r CosyVoice/requirements.txt")
            raise
        except Exception as e:
            print(f"加载 CosyVoice 失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_chattts(self):
        """加载 ChatTTS 模型"""
        try:
            import ChatTTS

            self.chattts = ChatTTS.Chat()
            compile_opt = bool(self.tts_config.get("compile", False))
            self.chattts.load(compile=compile_opt)

            self.chattts_style_prompt = self.tts_config.get("style_prompt", self.chattts_style_prompt)
            self.chattts_negative_prompt = self.tts_config.get("negative_prompt", self.chattts_negative_prompt)

            self.synthesize_func = self.synthesize_chattts
            self.apply_seed()
            print("✓ ChatTTS 加载成功")

        except ImportError:
            print("错误: 未安装 ChatTTS")
            print("安装命令: pip install ChatTTS")
            raise
        except Exception as e:
            print(f"加载 ChatTTS 失败: {e}")
            raise
    
    def load_openvoice(self):
        """加载 OpenVoice 模型"""
        try:
            # OpenVoice 需要额外的模型文件
            raise NotImplementedError("OpenVoice 暂未实现，请使用 ChatTTS")
        except Exception as e:
            print(f"加载 OpenVoice 失败: {e}")
            raise
    
    def load_coqui(self):
        """加载 Coqui TTS 模型"""
        try:
            from TTS.api import TTS
            
            # 使用中文 TTS 模型
            model_name = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
            self.tts = TTS(model_name)
            
            self.synthesize_func = self.synthesize_coqui
            print("✓ Coqui TTS 加载成功")
            
        except ImportError:
            print("错误: 未安装 Coqui TTS")
            print("安装命令: pip install TTS")
            raise
        except Exception as e:
            print(f"加载 Coqui TTS 失败: {e}")
            raise

    def load_piper(self):
        """加载 Piper 模型"""
        if shutil.which("piper") is None:
            raise RuntimeError("未找到 piper 命令，请确认 piper-tts 已正确安装")

        model_path = self.tts_config.get('model_path')
        config_path = self.tts_config.get('config_path')

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper 模型不存在: {model_path}")
        if config_path and not os.path.exists(config_path):
            raise FileNotFoundError(f"Piper 配置文件不存在: {config_path}")

        self.piper_model_path = model_path
        self.piper_config_path = config_path
        self.synthesize_func = self.synthesize_piper
        if self.piper_config_path:
            try:
                with open(self.piper_config_path, "r", encoding="utf-8") as cf:
                    cfg = json.load(cf)
                    self.piper_sample_rate = int(cfg.get("sample_rate", self.piper_sample_rate))
            except Exception as exc:
                print(f"⚠ 无法读取 Piper 配置文件采样率，使用默认 {self.piper_sample_rate} Hz: {exc}")
        print("✓ Piper 模型就绪")

    def apply_seed(self):
        if self.seed is None:
            return
        try:
            import ChatTTS
            if hasattr(ChatTTS, "utils") and hasattr(ChatTTS.utils, "seed_everything"):
                ChatTTS.utils.seed_everything(self.seed)
            elif hasattr(ChatTTS, "seed_everything"):
                ChatTTS.seed_everything(self.seed)
        except Exception as exc:
            print(f"⚠ ChatTTS 设置种子失败: {exc}")
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32 - 1))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _load_prompt_speech(self, prompt_path: Path):
        """
        加载参考音频（支持多种格式）
        
        Args:
            prompt_path: 音频文件路径
            
        Returns:
            prompt_speech_16k: 16kHz 单声道音频张量
        """
        from cosyvoice.utils.file_utils import load_wav
        
        if prompt_path.suffix.lower() == '.wav':
            # WAV 格式，直接使用 load_wav（官方推荐方式）
            prompt_speech_16k = load_wav(str(prompt_path), 16000)
            print(f"  加载 prompt_speech: {prompt_path} (使用 load_wav, 16kHz)")
        else:
            # 非 WAV 格式（MP3, FLAC 等），先转换为 WAV，再使用 load_wav（确保格式完全一致）
            # 这样可以确保与官方用法完全一致，避免格式问题
            try:
                import librosa
                import soundfile as sf
                import tempfile
                import os
                
                # 加载音频并转换为 16kHz 单声道
                # librosa.load 默认返回归一化的音频（[-1, 1] 范围）
                audio, sr = librosa.load(str(prompt_path), sr=16000, mono=True)
                
                # 确保数值范围正确（librosa 默认返回 [-1, 1]，但需要检查）
                max_val = abs(audio).max()
                if max_val > 1.0:
                    audio = audio / max_val
                    print(f"  归一化音频数值范围到 [-1, 1]")
                
                # 临时保存为 WAV，使用 PCM_16 格式（与官方 load_wav 期望的格式一致）
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_wav_path = tmp_file.name
                
                # 使用 PCM_16 格式保存，soundfile 会自动将 [-1, 1] 范围的浮点数转换为 16-bit 整数
                sf.write(tmp_wav_path, audio, 16000, subtype='PCM_16')
                
                # 使用官方的 load_wav 方法加载（确保格式完全一致）
                prompt_speech_16k = load_wav(tmp_wav_path, 16000)
                
                # 清理临时文件
                os.unlink(tmp_wav_path)
                
                print(f"  加载 prompt_speech: {prompt_path} (转换为 WAV 后使用 load_wav, 16kHz)")
            except ImportError:
                raise ValueError(f"不支持 {prompt_path.suffix} 格式，请安装 librosa 和 soundfile，或转换为 WAV 格式")
            except Exception as e:
                raise ValueError(f"无法加载音频文件 {prompt_path}: {e}")
        
        return prompt_speech_16k

    def synthesize_cosyvoice(self, text: str, output_path: str, **kwargs):
        """使用 CosyVoice 合成语音（根据 GitHub README API）"""
        import torchaudio
        import numpy as np
        from cosyvoice.utils.file_utils import load_wav
        
        # 根据 GitHub README，CosyVoice2 和 CosyVoice 的 API 不同：
        # CosyVoice2: inference_zero_shot, inference_instruct2, inference_cross_lingual
        # CosyVoice: inference_sft, inference_zero_shot, inference_cross_lingual, inference_instruct
        
        speaker = kwargs.get("speaker", self.cosyvoice_default_emotion or "中文女")
        # 获取 prompt_speech（优先使用 kwargs，其次使用配置中的默认值）
        prompt_speech = kwargs.get("prompt_speech")
        if prompt_speech is None and self.use_cosyvoice2:
            prompt_speech = self.cosyvoice_prompt_speech
        # 获取 prompt_text（优先使用 kwargs，其次使用配置中的默认值）
        prompt_text = kwargs.get("prompt_text")
        if not prompt_text and self.use_cosyvoice2:
            prompt_text = self.cosyvoice_prompt_text
        # 获取 mode（优先使用 kwargs，其次使用配置中的默认值）
        mode = kwargs.get("mode")
        if not mode and self.use_cosyvoice2:
            mode = self.cosyvoice_mode
        
        try:
            audio_chunks = []
            
            if self.use_cosyvoice2:
                # CosyVoice2 使用方式（按照官方示例）
                # 支持：inference_zero_shot, inference_instruct2, inference_cross_lingual
                
                # 获取模式参数
                mode = kwargs.get("mode", "zero_shot")  # zero_shot, instruct2, cross_lingual
                instruction = kwargs.get("instruction", "")  # 用于 instruct2
                
                if mode == "zero_shot":
                    # zero_shot 模式（需要 prompt_speech）
                    if prompt_speech is None:
                        raise ValueError("CosyVoice2 zero_shot 模式需要 prompt_speech 参数")
                    
                    if isinstance(prompt_speech, str):
                        # 支持多种音频格式（WAV, MP3, FLAC 等）
                        prompt_path = Path(prompt_speech)
                        if not prompt_path.is_absolute():
                            # 相对路径，尝试从配置目录或当前目录查找
                            config_dir = Path(self.config_path).parent if hasattr(self, 'config_path') else Path.cwd()
                            prompt_path = config_dir / prompt_speech
                            if not prompt_path.exists():
                                prompt_path = Path.cwd() / prompt_speech
                        
                        if not prompt_path.exists():
                            raise FileNotFoundError(f"prompt_speech 文件不存在: {prompt_speech}")
                        
                        # 使用统一的音频加载方法
                        prompt_speech_16k = self._load_prompt_speech(prompt_path)
                    else:
                        prompt_speech_16k = prompt_speech
                    
                    # 按照官方用法：inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=False, text_frontend=False)
                    # 注意：对于中文文本，建议使用 text_frontend=True 进行文本规范化处理
                    # text_frontend=False 可能导致中文文本无法正确识别，输出乱码或英文
                    text_frontend = kwargs.get("text_frontend", True)  # 默认 True，启用中文文本规范化
                    
                    # 设置随机种子（如果配置了 seed，确保生成结果可复现）
                    if self.seed is not None:
                        try:
                            from cosyvoice.utils.common import set_all_random_seed
                            set_all_random_seed(self.seed)
                        except ImportError:
                            # 如果无法导入 CosyVoice 的 set_all_random_seed，使用通用的种子设置
                            self.apply_seed()
                    
                    for i, result in enumerate(self.cosyvoice.inference_zero_shot(
                        text,
                        prompt_text if prompt_text else "希望你以后能够做的比我还好呦。",  # 默认 prompt_text
                        prompt_speech_16k,
                        stream=self.cosyvoice_stream,
                        text_frontend=text_frontend
                    )):
                        if isinstance(result, dict) and 'tts_speech' in result:
                            audio_chunks.append(result['tts_speech'])
                        else:
                            audio_chunks.append(result)
                
                elif mode == "instruct2":
                    # instruct2 模式（需要 prompt_speech 和 instruction）
                    if prompt_speech is None:
                        raise ValueError("CosyVoice2 instruct2 模式需要 prompt_speech 参数")
                    if not instruction:
                        raise ValueError("CosyVoice2 instruct2 模式需要 instruction 参数")
                    
                    if isinstance(prompt_speech, str):
                        # 使用相同的音频加载逻辑
                        prompt_path = Path(prompt_speech)
                        if not prompt_path.is_absolute():
                            config_dir = Path(self.config_path).parent if hasattr(self, 'config_path') else Path.cwd()
                            prompt_path = config_dir / prompt_speech
                            if not prompt_path.exists():
                                prompt_path = Path.cwd() / prompt_speech
                        
                        if not prompt_path.exists():
                            raise FileNotFoundError(f"prompt_speech 文件不存在: {prompt_speech}")
                        
                        # 使用统一的音频加载方法
                        prompt_speech_16k = self._load_prompt_speech(prompt_path)
                    else:
                        prompt_speech_16k = prompt_speech
                    
                    # 按照官方用法：inference_instruct2(text, instruction, prompt_speech_16k, stream=False, text_frontend=False)
                    # 对于中文文本，建议使用 text_frontend=True 进行文本规范化处理
                    text_frontend = kwargs.get("text_frontend", True)  # 默认 True，启用中文文本规范化
                    
                    # 设置随机种子（如果配置了 seed，确保生成结果可复现）
                    if self.seed is not None:
                        try:
                            from cosyvoice.utils.common import set_all_random_seed
                            set_all_random_seed(self.seed)
                        except ImportError:
                            # 如果无法导入 CosyVoice 的 set_all_random_seed，使用通用的种子设置
                            self.apply_seed()
                    
                    for i, result in enumerate(self.cosyvoice.inference_instruct2(
                        text,
                        instruction,
                        prompt_speech_16k,
                        stream=self.cosyvoice_stream,
                        text_frontend=text_frontend
                    )):
                        if isinstance(result, dict) and 'tts_speech' in result:
                            audio_chunks.append(result['tts_speech'])
                        else:
                            audio_chunks.append(result)
                
                elif mode == "cross_lingual":
                    # cross_lingual 模式（需要 prompt_speech）
                    if prompt_speech is None:
                        raise ValueError("CosyVoice2 cross_lingual 模式需要 prompt_speech 参数")
                    
                    if isinstance(prompt_speech, str):
                        # 使用相同的音频加载逻辑
                        prompt_path = Path(prompt_speech)
                        if not prompt_path.is_absolute():
                            config_dir = Path(self.config_path).parent if hasattr(self, 'config_path') else Path.cwd()
                            prompt_path = config_dir / prompt_speech
                            if not prompt_path.exists():
                                prompt_path = Path.cwd() / prompt_speech
                        
                        if not prompt_path.exists():
                            raise FileNotFoundError(f"prompt_speech 文件不存在: {prompt_speech}")
                        
                        # 使用统一的音频加载方法
                        prompt_speech_16k = self._load_prompt_speech(prompt_path)
                    else:
                        prompt_speech_16k = prompt_speech
                    
                    # 按照官方用法：inference_cross_lingual(text, prompt_speech_16k, stream=False, text_frontend=False)
                    # 对于中文文本，建议使用 text_frontend=True 进行文本规范化处理
                    text_frontend = kwargs.get("text_frontend", True)  # 默认 True，启用中文文本规范化
                    
                    # 设置随机种子（如果配置了 seed，确保生成结果可复现）
                    if self.seed is not None:
                        try:
                            from cosyvoice.utils.common import set_all_random_seed
                            set_all_random_seed(self.seed)
                        except ImportError:
                            # 如果无法导入 CosyVoice 的 set_all_random_seed，使用通用的种子设置
                            self.apply_seed()
                    
                    for i, result in enumerate(self.cosyvoice.inference_cross_lingual(
                        text,
                        prompt_speech_16k,
                        stream=self.cosyvoice_stream,
                        text_frontend=text_frontend
                    )):
                        if isinstance(result, dict) and 'tts_speech' in result:
                            audio_chunks.append(result['tts_speech'])
                        else:
                            audio_chunks.append(result)
                else:
                    raise ValueError(f"CosyVoice2 不支持模式: {mode}，支持的模式：zero_shot, instruct2, cross_lingual")
            else:
                # CosyVoice 使用方式（SFT 模式）
                if hasattr(self.cosyvoice, "inference_sft"):
                    for i, result in enumerate(self.cosyvoice.inference_sft(
                        text,
                        speaker,
                        stream=self.cosyvoice_stream
                    )):
                        if isinstance(result, dict) and 'tts_speech' in result:
                            audio_chunks.append(result['tts_speech'])
                        else:
                            audio_chunks.append(result)
                else:
                    raise AttributeError("CosyVoice 不支持 inference_sft，请检查版本")
            
            # 合并音频块
            if len(audio_chunks) == 0:
                raise RuntimeError("CosyVoice 未生成音频")
            
            # 处理音频块
            processed_chunks = []
            for chunk in audio_chunks:
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu()
                    if chunk.dim() > 1:
                        chunk = chunk.squeeze()
                    processed_chunks.append(chunk.numpy())
                elif isinstance(chunk, np.ndarray):
                    if chunk.ndim > 1:
                        chunk = chunk.squeeze()
                    processed_chunks.append(chunk)
                else:
                    processed_chunks.append(np.array(chunk))
            
            # 合并所有音频块
            if len(processed_chunks) > 1:
                audio = np.concatenate(processed_chunks)
            else:
                audio = processed_chunks[0]
            
            # 确保是 1D numpy array
            if audio.ndim > 1:
                audio = audio.squeeze()
            
        except Exception as e:
            print(f"⚠ CosyVoice 生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"CosyVoice 生成失败: {e}") from e
        
        # 获取采样率
        sample_rate = self.cosyvoice.sample_rate if hasattr(self.cosyvoice, 'sample_rate') else self.cosyvoice_sample_rate
        
        # 保存音频（使用 torchaudio，CosyVoice 推荐的方式）
        try:
            # 转换为 torch.Tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio)
            else:
                audio_tensor = audio
            
            # 确保是 2D (channels, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            torchaudio.save(output_path, audio_tensor, sample_rate)
            print(f"✓ 语音已保存: {output_path}")
            print(f"  时长: {len(audio) / sample_rate:.2f} 秒")
            print(f"  采样率: {sample_rate} Hz")
        except Exception as e:
            # 备用方法：使用 soundfile
            import soundfile as sf
            sf.write(output_path, audio, sample_rate)
            print(f"✓ 语音已保存: {output_path} (使用 soundfile)")
            print(f"  时长: {len(audio) / sample_rate:.2f} 秒")
            print(f"  采样率: {sample_rate} Hz")
    
    def synthesize_chattts(self, text: str, output_path: str, **kwargs):
        """使用 ChatTTS 合成语音"""
        import soundfile as sf
        import numpy as np
        
        import ChatTTS

        temperature = kwargs.get("temperature", self.tts_config.get("temperature", 0.5))
        style_prompt = kwargs.get("style_prompt", self.chattts_style_prompt)
        negative_prompt = kwargs.get("negative_prompt", self.chattts_negative_prompt)

        params_infer_code = ChatTTS.Chat.InferCodeParams(temperature=temperature)

        params_refine_text = None
        if style_prompt or negative_prompt:
            params_kwargs = {}
            if style_prompt:
                params_kwargs["prompt"] = style_prompt
            if negative_prompt:
                params_kwargs["negative_prompt"] = negative_prompt
            try:
                params_refine_text = ChatTTS.Chat.RefineTextParams(**params_kwargs)
            except TypeError:
                # 某些版本不支持 negative_prompt
                params_kwargs.pop("negative_prompt", None)
                if params_kwargs:
                    params_refine_text = ChatTTS.Chat.RefineTextParams(**params_kwargs)

        try:
            wavs = self.chattts.infer(
                [text],
                params_infer_code=params_infer_code,
                params_refine_text=params_refine_text,
                skip_refine_text=False,
                do_text_normalization=False,
                do_homophone_replacement=False,
                split_text=False,
            )
        except Exception as e:
            print(f"⚠ ChatTTS 生成失败: {e}")
            print("  尝试使用简化参数输出...")
            wavs = self.chattts.infer([text], skip_refine_text=True, split_text=False)
        
        # 保存音频
        if isinstance(wavs, list) and len(wavs) > 0:
            audio = wavs[0]
            # 确保是 numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            # ChatTTS 默认采样率为 24000
            sample_rate = 24000
            
            sf.write(output_path, audio, sample_rate)
            print(f"✓ 语音已保存: {output_path}")
            print(f"  时长: {len(audio) / sample_rate:.2f} 秒")
        else:
            raise ValueError("ChatTTS 返回的音频数据格式不正确")
    
    def synthesize_coqui(self, text: str, output_path: str, **kwargs):
        """使用 Coqui TTS 合成语音"""
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
        )
        print(f"✓ 语音已保存: {output_path}")

    def synthesize_gtts(self, text: str, output_path: str, **kwargs):
        """使用 gTTS 合成语音"""
        from gtts import gTTS

        language = kwargs.get('language', self.tts_config.get('language', 'zh-cn'))
        slow = kwargs.get('slow', self.tts_config.get('slow', False))

        tts = gTTS(text=text, lang=language, slow=slow)
        tts.save(output_path)
        print(f"✓ gTTS 语音已保存: {output_path}")

    def synthesize_piper(self, text: str, output_path: str, **kwargs):
        """使用 Piper 合成语音"""
        cmd = [
            "piper",
            "--model",
            self.piper_model_path,
            "--output_file",
            output_path,
        ]

        if self.piper_config_path:
            cmd.extend(["--config", self.piper_config_path])

        cmd.extend(["--text", text])

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✓ Piper 语音已保存: {output_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Piper 生成失败: {e.stderr.decode('utf-8', 'ignore')}")
    
    def generate(
        self,
        text: str,
        output_path: str,
        temperature: Optional[float] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        **kwargs  # 支持 CosyVoice2 的额外参数（prompt_speech, prompt_text, mode, text_frontend 等）
    ) -> str:
        """
        生成语音
        
        Args:
            text: 输入文本
            output_path: 输出音频路径
            temperature: 温度参数
            speed: 语速
            pitch: 音调
            
        Returns:
            输出音频路径
        """
        print(f"\n生成语音: {len(text)} 字符")
        
        # 使用配置中的默认值
        temperature = temperature or self.tts_config.get('temperature', 0.7)
        speed = speed or self.tts_config.get('speed', 1.0)
        pitch = pitch or self.tts_config.get('pitch', 1.0)
        
        if self.engine == "piper" and self.tts_config.get("split_sentences", False):
            try:
                return self.generate_piper_with_pauses(
                    text=text,
                    output_path=output_path,
                    pause_duration=float(self.tts_config.get("sentence_pause", 0.35)),
                )
            except Exception as exc:
                print(f"⚠ Piper 分句生成失败，回退为单段输出: {exc}")

        # 生成语音（传递所有额外参数，包括 CosyVoice2 的 prompt_speech, prompt_text, mode, text_frontend 等）
        self.synthesize_func(
            text,
            output_path,
            temperature=temperature,
            speed=speed,
            pitch=pitch,
            **kwargs
        )
        
        return output_path

    def generate_piper_with_pauses(
        self,
        text: str,
        output_path: str,
        pause_duration: float = 0.35,
    ) -> str:
        """将文本按句分段生成 Piper 语音并插入停顿"""
        sentences = self.split_text_into_sentences(text)
        if len(sentences) <= 1:
            self.synthesize_piper(text, output_path)
            return output_path

        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("未安装 soundfile，无法执行分句合成。请执行 `pip install soundfile`.") from exc

        sample_rate = self.piper_sample_rate
        audio_segments: List[np.ndarray] = []

        for idx, sentence in enumerate(sentences):
            if not sentence:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = Path(tmp.name)
            try:
                self.synthesize_piper(sentence, str(temp_path))
                audio, sr = sf.read(str(temp_path), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio_segments.append(audio)
                sample_rate = sr
                if idx < len(sentences) - 1 and pause_duration > 0:
                    pause_samples = int(sr * pause_duration)
                    if pause_samples > 0:
                        audio_segments.append(np.zeros(pause_samples, dtype=audio.dtype))
            finally:
                temp_path.unlink(missing_ok=True)

        if not audio_segments:
            self.synthesize_piper(text, output_path)
            return output_path

        combined_audio = np.concatenate(audio_segments)
        sf.write(output_path, combined_audio, sample_rate)
        print(f"✓ Piper 分句语音已保存: {output_path}")
        return output_path

    def split_text_into_sentences(self, text: str) -> List[str]:
        """按中文标点切分文本"""
        text = text.strip()
        if not text:
            return []
        pattern = r"[^。！？!?；;]+[。！？!?；;]?"
        sentences = re.findall(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def batch_generate(self, texts: List[str], output_dir: str, prefix: str = "audio") -> List[str]:
        """批量生成语音"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"{prefix}_{i:03d}.wav"
            
            try:
                self.generate(text, str(output_path))
                output_files.append(str(output_path))
            except Exception as e:
                print(f"✗ 生成失败 {i}: {e}")
                continue
        
        return output_files


def main():
    parser = argparse.ArgumentParser(description="TTS 语音生成")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--text", type=str, help="输入文本")
    parser.add_argument("--input", type=str, help="输入文本文件（每行一段）")
    parser.add_argument("--output", type=str, help="输出音频路径或目录")
    parser.add_argument("--temperature", type=float, help="温度参数")
    parser.add_argument("--speed", type=float, help="语速")
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = TTSGenerator(args.config)
    
    # 获取文本
    if args.text:
        texts = [args.text]
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("错误: 需要提供 --text 或 --input 参数")
        return
    
    # 生成语音
    if len(texts) == 1:
        # 单个文本
        output_path = args.output or generator.config['paths']['output_dir'] / "audio.wav"
        generator.generate(texts[0], str(output_path), temperature=args.temperature, speed=args.speed)
    else:
        # 批量生成
        output_dir = args.output or generator.config['paths']['output_dir']
        generator.batch_generate(texts, str(output_dir))


if __name__ == "__main__":
    main()



