#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字幕生成脚本
使用 WhisperX 进行语音识别和字幕对齐
"""

import os
import yaml
import argparse
from pathlib import Path
import torch
import soundfile as sf
from typing import Optional, List, Any
import json
import math
import warnings

# 抑制 WhisperX 模型加载时的滑动窗口注意力警告
# 该警告不影响功能，只是提示 sdpa 实现不支持滑动窗口注意力优化
warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*sdpa.*")


class SubtitleGenerator:
    """字幕生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化字幕生成器"""
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.subtitle_config = self.config['subtitle']
        self.config_dir = self.config_path.parent
        self.script_config = self.subtitle_config.get("script", {})
        # 如果遇到 bad_alloc 错误，可能是 CUDA 问题，优先使用 CPU
        # 可以通过环境变量强制使用 CPU: export WHISPERX_FORCE_CPU=1
        force_cpu = os.environ.get("WHISPERX_FORCE_CPU", "0") == "1"
        if force_cpu:
            self.device = "cpu"
            print("  ⚠ 检测到 WHISPERX_FORCE_CPU=1，强制使用 CPU 模式")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        self.model = None
        self.align_model = None
        self.metadata = None
        self.fallback = False
        self.load_model()
        
        # 创建输出目录
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)
    
    def load_model(self):
        """加载 WhisperX 模型"""
        model_size = self.subtitle_config['model_size']
        language = self.subtitle_config['language']
        
        try:
            import whisperx

            print(f"加载 WhisperX 模型: {model_size}, 语言: {language}")
            download_root = self.subtitle_config.get("model_dir")
            model_source = model_size
            if download_root and os.path.isdir(download_root):
                model_source = download_root
            local_files_only = self.subtitle_config.get("local_files_only")
            if local_files_only is None:
                local_files_only = bool(download_root)
            
            # 计算类型（H20 96GB 显存充足，使用 float16）
            compute_type = self.subtitle_config.get("compute_type", "float16" if self.device == "cuda" else "int8")
            
            # 内存优化：彻底清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # 强制垃圾回收
                import gc
                gc.collect()
            
            print(f"  计算类型: {compute_type}")
            print(f"  设备: {self.device}")
            
            # 优先尝试 int8 量化（更节省内存）
            if compute_type == "float16":
                print(f"  尝试使用 float16...")
                try:
                    self.model = whisperx.load_model(
                        model_source,
                        self.device,
                        compute_type=compute_type,
                        language=language,
                        download_root=download_root,
                        asr_options={"best_of": 1},
                        local_files_only=local_files_only,
                    )
                    print("✓ WhisperX 模型加载成功（float16）")
                except (RuntimeError, Exception) as e:
                    if "out of memory" in str(e).lower() or "bad_alloc" in str(e).lower():
                        print(f"⚠ float16 显存不足，自动切换到 int8 量化...")
                        # 清理缓存后重试
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                        compute_type = "int8"
                    else:
                        raise
            
            # 如果 float16 失败或直接使用 int8
            # 注意: 如果之前遇到过 bad_alloc，直接使用 CPU 模式
            if compute_type == "int8" or self.model is None:
                # 检查是否应该使用 CPU（如果之前失败过或配置了强制 CPU）
                use_cpu_fallback = False
                if self.device == "cpu":
                    use_cpu_fallback = True
                    print(f"  使用 CPU 模式（避免 CUDA 问题）...")
                else:
                    print(f"  使用 int8 量化（更节省内存）...")
                
                try:
                    if torch.cuda.is_available() and not use_cpu_fallback:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
                    
                    # 如果之前失败过或配置了强制 CPU，直接使用 CPU
                    device_to_use = "cpu" if use_cpu_fallback else self.device
                    
                    self.model = whisperx.load_model(
                        model_source,
                        device_to_use,
                        compute_type="int8",
                        language=language,
                        download_root=download_root,
                        asr_options={"best_of": 1},
                        local_files_only=local_files_only,
                    )
                    if use_cpu_fallback:
                        print("✓ WhisperX 模型加载成功（CPU 模式，int8 量化）")
                        print("  注意: CPU 模式较慢，但可以避免 CUDA 相关问题")
                    else:
                        print("✓ WhisperX 模型加载成功（int8 量化）")
                except (RuntimeError, Exception) as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "bad_alloc" in error_msg:
                        # 尝试使用 CPU 模式
                        print(f"  ⚠ CUDA 模式失败，尝试使用 CPU 模式...")
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                            self.device = "cpu"
                            self.model = whisperx.load_model(
                                model_source,
                                "cpu",
                                compute_type="int8",
                                language=language,
                                download_root=download_root,
                                asr_options={"best_of": 1},
                                local_files_only=local_files_only,
                            )
                            print("✓ WhisperX 模型加载成功（CPU 模式，int8 量化）")
                            print("  注意: CPU 模式较慢，但可以避免 CUDA 相关问题")
                        except Exception as e2:
                            raise RuntimeError(
                                f"无法加载 WhisperX 模型。\n"
                                f"CUDA 错误: {e}\n"
                                f"CPU 错误: {e2}\n"
                                f"可能的原因:\n"
                                f"  1) 模型文件损坏或不完整（尝试重新下载）\n"
                                f"  2) WhisperX 或 faster-whisper 版本问题（尝试更新）\n"
                                f"  3) CUDA 驱动版本不兼容（检查 CUDA 版本）\n"
                                f"  4) C++ 库问题（可能需要重新编译 faster-whisper）\n"
                                f"建议:\n"
                                f"  - 检查模型文件: {model_source}\n"
                                f"  - 尝试删除模型目录重新下载\n"
                                f"  - 更新 whisperx: pip install --upgrade whisperx\n"
                                f"  - 更新 faster-whisper: pip install --upgrade faster-whisper"
                            ) from e2
                    else:
                        raise
            
            # 启用对齐功能（H20 96GB 显存充足）
            align_enabled = self.subtitle_config.get('align', False)
            if align_enabled:
                try:
                    # 使用配置的对齐模型（中文专用）
                    align_model_name = self.subtitle_config.get("align_model", None)
                    if align_model_name:
                        print(f"加载对齐模型: {align_model_name}")
                        self.align_model, self.metadata = whisperx.load_align_model(
                            language_code=language,
                            device=self.device,
                            model_name=align_model_name,
                        )
                    else:
                        # 使用默认对齐模型
                        self.align_model, self.metadata = whisperx.load_align_model(
                            language_code=language,
                            device=self.device,
                        )
                    print("✓ 对齐模型加载成功")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "bad_alloc" in str(e).lower():
                        print(f"⚠ 内存不足，禁用对齐功能: {e}")
                        print("  提示: 可以降低 batch_size 或关闭其他程序以释放内存")
                    else:
                        print(f"⚠ 对齐模型加载失败，将禁用对齐功能: {e}")
                        print("  提示: 请确保已安装 pyannote.audio 并更新到最新版本")
                    self.align_model = None
                    self.metadata = None
                except Exception as e:
                    print(f"⚠ 对齐模型加载失败，将禁用对齐功能: {e}")
                    print("  提示: 请确保已安装 pyannote.audio 并更新到最新版本")
                    self.align_model = None
                    self.metadata = None
            else:
                self.align_model = None
                self.metadata = None
            
            # 加载 VAD 模型（如果配置了）
            self.vad_model = None
            vad_model_name = self.subtitle_config.get("vad_model")
            if vad_model_name:
                try:
                    print(f"加载 VAD 模型: {vad_model_name}")
                    self.vad_model = whisperx.load_vad_model(
                        vad_model_name,
                        device=self.device,
                    )
                    print("✓ VAD 模型加载成功")
                except Exception as e:
                    print(f"⚠ VAD 模型加载失败: {e}")
                    self.vad_model = None
            
            self.whisperx = whisperx
            print("✓ WhisperX 模型加载成功")
        except Exception as e:
            print(f"⚠ WhisperX 加载失败: {e}")
            print("  将使用简易字幕生成模式（根据配音文本估算时间）")
            self.fallback = True
    
    def generate(
        self,
        audio_path: str,
        output_path: str,
        format: Optional[str] = None,
        narration: Optional[str] = None,
        segments: Optional[List[str]] = None,
        video_durations: Optional[List[float]] = None,
        total_duration: Optional[float] = None,
    ) -> str:
        """
        生成字幕
        
        Args:
            audio_path: 输入音频路径
            output_path: 输出字幕路径
            format: 字幕格式 (srt, vtt, ass)
            
        Returns:
            输出字幕路径
        """
        print(f"\n生成字幕: {audio_path} -> {output_path}")
        
        format = format or self.subtitle_config['format']
        
        use_script_segments = bool(
            segments and self.subtitle_config.get("use_script_segments", False)
        )
        if not self.fallback and self.model is not None and not use_script_segments:
            import whisperx
            import os

            # 禁用 VAD 以避免段错误（通过环境变量）
            os.environ["WHISPERX_VAD"] = "false"
            
            try:
                audio = whisperx.load_audio(audio_path)
                
                # 批量大小（large-v3 可支持更大 batch）
                batch_size = self.subtitle_config.get("batch_size", 32)
                
                # 使用 VAD（如果已加载）
                use_vad = self.vad_model is not None
                if use_vad:
                    # 新版本 WhisperX 的 transcribe 已不再支持 vad_filter 参数
                    print("⚠ 当前 WhisperX 版本不支持在 transcribe 中直接使用 VAD，暂时禁用 VAD")
                    use_vad = False
                
                if use_vad:
                    print("进行语音识别（使用 VAD）...")
                else:
                    print("进行语音识别（未使用 VAD）...")
                
                # 直接使用模型 transcribe
                result = self.model.transcribe(
                    audio, 
                    batch_size=batch_size,
                    language=self.subtitle_config.get("language", "zh"),
                )
                
                # 转换结果格式
                if "segments" not in result:
                    # 如果返回格式不同，尝试转换
                    if "text" in result:
                        # 简单格式，需要手动分段
                        text = result["text"]
                        duration = self.get_audio_duration(audio_path)
                        result = {
                            "segments": [{
                                "start": 0.0,
                                "end": duration,
                                "text": text
                            }]
                        }
                
                if self.align_model and self.metadata:
                    print("进行时间对齐（字级对齐）...")
                    try:
                        result = whisperx.align(
                            result["segments"],
                            self.align_model,
                            self.metadata,
                            audio,
                            self.device,
                            return_char_alignments=True,  # 启用字级对齐（误差 < 50ms）
                        )
                        print("✓ 时间对齐完成（字级精度）")
                    except Exception as e:
                        print(f"⚠ 时间对齐失败，使用原始识别结果: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 如果提供了精确的音频时长列表，优先使用它来分配时间轴（最准确）
                # 这样可以确保字幕时间轴与音频片段完全对齐
                if video_durations and segments and len(video_durations) == len(segments):
                    print("  使用精确音频时长列表分配字幕时间轴（确保与音频完全对齐）...")
                    # 使用精确的音频时长列表重新分配时间轴
                    total_audio_duration = sum(video_durations)
                    segment_entries = self.allocate_segments_by_video_durations(
                        total_audio_duration, segments, video_durations
                    )
                    # 构建新的 segments
                    result["segments"] = [
                        {"start": start, "end": end, "text": text}
                        for start, end, text in segment_entries
                    ]
                    print(f"  ✓ 已使用精确音频时长列表分配字幕时间轴，确保完全对齐")
                elif segments and len(segments) > 0:
                    # 如果没有精确时长列表，尝试将识别结果与提供的 segments 对齐
                    audio_duration = self.get_audio_duration(audio_path)
                    result = self.align_segments_to_provided_texts(result, segments, audio_duration)
                
                result = self.apply_script_text_override(result, narration, segments)
                
                if format == "srt":
                    self.save_srt(result, output_path)
                elif format == "vtt":
                    self.save_vtt(result, output_path)
                elif format == "ass":
                    self.save_ass(result, output_path)
                else:
                    raise ValueError(f"不支持的字幕格式: {format}")
                print(f"✓ 字幕已保存: {output_path}")
                return output_path
            except Exception as e:
                print(f"⚠ WhisperX 识别失败: {e}")
                import traceback
                traceback.print_exc()
                print("  回退到简易字幕生成模式...")
                self.fallback = True

        # fallback 模式：根据实际音频时长和旁白文本分配字幕时间
        # 优先使用传入的总时长（如果是分段音频，应该是所有片段的总时长）
        # 如果没有传入总时长，则使用单个音频文件的时长
        if total_duration is not None and total_duration > 0:
            duration = total_duration
            print(f"  使用传入的总音频时长: {duration:.2f}s")
        else:
            duration = self.get_audio_duration(audio_path)
            print(f"  使用音频文件时长: {duration:.2f}s")
        
        scripted_segments = segments or self.build_segments_from_text(narration or "")
        
        # 如果提供了视频时长列表，使用视频时长分配字幕时间（确保字幕与视频对应）
        if video_durations and len(video_durations) == len(scripted_segments):
            segment_entries = self.allocate_segments_by_video_durations(
                duration, scripted_segments, video_durations
            )
            print(f"  使用实际视频时长分配字幕时间，确保字幕与视频片段对应")
        else:
            # 如果没有视频时长信息，根据旁白字数按比例分配实际音频时长（更准确）
            # 因为实际生成的音频时长可能和估算的不完全一致
            # 使用实际音频总时长，按文本长度比例分配
            if video_durations:
                print(f"  ⚠ 视频时长数量 ({len(video_durations)}) 与文本数量 ({len(scripted_segments)}) 不一致，回退到文本长度比例分配")
            segment_entries = self.allocate_segments_by_duration(
                duration, scripted_segments
            )
            print(f"  使用实际音频时长 ({duration:.2f}s) 按文本长度比例分配字幕时间")
        result = {
            "segments": [
                {"start": start, "end": end, "text": text} for start, end, text in segment_entries
            ]
        }
        if format == "srt":
            self.save_srt(result, output_path)
        elif format == "vtt":
            self.save_vtt(result, output_path)
        elif format == "ass":
            self.save_ass(result, output_path)
        else:
            raise ValueError(f"不支持的字幕格式: {format}")
        print(f"✓ 已生成占位字幕: {output_path}")
        return output_path
    
    def apply_script_text_override(
        self,
        result: dict,
        narration: Optional[str],
        provided_segments: Optional[List[str]],
    ) -> dict:
        """使用原稿文本替换 WhisperX 识别结果"""
        script_conf = self.script_config or {}
        if not script_conf.get("use_original_text", False):
            return result
        if not result or "segments" not in result:
            return result
        
        asr_segments = result.get("segments") or []
        if not asr_segments:
            return result
        
        script_segments = self.collect_script_segments(narration, provided_segments, script_conf)
        if not script_segments:
            print("⚠ 未找到原稿文本，保留 WhisperX 识别内容")
            return result
        
        aligned_texts = self.align_script_to_segments(asr_segments, script_segments)
        if not aligned_texts:
            return result
        
        for seg, new_text in zip(asr_segments, aligned_texts):
            seg["text"] = new_text.strip()
        print("✓ 已使用原稿文本替换字幕内容")
        return result
    
    def collect_script_segments(
        self,
        narration: Optional[str],
        provided_segments: Optional[List[str]],
        script_conf: dict,
    ) -> List[str]:
        """收集用于替换字幕的原稿文本片段"""
        segments = []
        if provided_segments:
            segments = [s.strip() for s in provided_segments if s and str(s).strip()]
        
        if not segments and narration:
            segments = self.build_segments_from_text(
                narration,
                delimiters=script_conf.get("delimiters"),
            )
        
        if not segments:
            segments = self.load_script_segments_from_file(script_conf)
        
        return [s for s in segments if s.strip()]
    
    def load_script_segments_from_file(self, script_conf: dict) -> List[str]:
        """从配置的脚本文件中读取文本"""
        script_path = script_conf.get("path")
        if not script_path:
            return []
        
        resolved_path = Path(script_path)
        if not resolved_path.is_absolute():
            resolved_path = (self.config_dir / resolved_path).resolve()
        if not resolved_path.exists():
            print(f"⚠ 原稿文件不存在: {resolved_path}")
            return []
        
        fmt = script_conf.get("format", "auto").lower()
        if fmt == "auto":
            suffix = resolved_path.suffix.lower()
            fmt = "json" if suffix == ".json" else "text"
        
        texts: List[str] = []
        try:
            if fmt == "json":
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                fields = script_conf.get("fields") or []
                primary_field = script_conf.get("field")
                ordered_fields = []
                if primary_field:
                    ordered_fields.append(primary_field)
                ordered_fields.extend([fld for fld in fields if fld and fld != primary_field])
                if not ordered_fields:
                    ordered_fields = ["full_narration"]
                
                for field in ordered_fields:
                    texts.extend(self.extract_text_from_json_field(data, field))
                
                if script_conf.get("include_scene_narration", False):
                    scenes = data.get("scenes", [])
                    if isinstance(scenes, list):
                        for scene in scenes:
                            if isinstance(scene, dict):
                                narration = scene.get("narration")
                                if isinstance(narration, str):
                                    texts.append(narration)
            else:
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        except Exception as e:
            print(f"⚠ 读取原稿文件失败: {e}")
            return []
        
        merged = "\n".join([t for t in texts if isinstance(t, str)])
        if not merged.strip():
            return []
        
        return self.build_segments_from_text(
            merged,
            delimiters=script_conf.get("delimiters"),
        )
    
    def extract_text_from_json_field(self, data: Any, field_path: str) -> List[str]:
        """按照 field_path （支持 scenes[].narration）提取文本"""
        if not field_path:
            return []
        tokens = [token for token in field_path.split(".") if token]
        return self._extract_json_values(data, tokens)
    
    def _extract_json_values(self, obj: Any, tokens: List[str]) -> List[str]:
        if not tokens:
            if isinstance(obj, str):
                return [obj]
            if isinstance(obj, list):
                return [item for item in obj if isinstance(item, str)]
            return []
        
        token = tokens[0]
        is_list = token.endswith("[]")
        key = token[:-2] if is_list else token
        
        if isinstance(obj, dict):
            value = obj.get(key)
        else:
            value = None
        
        if value is None:
            return []
        
        remainder = tokens[1:]
        results: List[str] = []
        
        if is_list:
            if not isinstance(value, list):
                return []
            for item in value:
                results.extend(self._extract_json_values(item, remainder))
        else:
            results.extend(self._extract_json_values(value, remainder))
        return results
    
    def align_script_to_segments(
        self,
        asr_segments: List[dict],
        script_segments: List[str],
    ) -> List[str]:
        """根据 ASR 片段数量重新分配脚本文本"""
        target = len(asr_segments)
        source = len(script_segments)
        if target == 0 or source == 0:
            return []
        if target == source:
            return script_segments
        
        aligned: List[str] = []
        for idx in range(target):
            start = math.floor(idx * source / target)
            end = math.floor((idx + 1) * source / target)
            if idx == target - 1:
                end = source
            if end <= start:
                end = min(start + 1, source)
            start = min(start, source - 1)
            end = max(end, start + 1)
            end = min(end, source)
            chunk = "".join(script_segments[start:end]).strip()
            if not chunk:
                chunk = script_segments[min(start, source - 1)]
            aligned.append(chunk)
        return aligned
    
    def align_segments_to_provided_texts(
        self,
        result: dict,
        provided_segments: List[str],
        audio_duration: float,
    ) -> dict:
        """将 WhisperX 识别结果与提供的文本 segments 对齐
        
        Args:
            result: WhisperX 识别结果，包含 segments 列表
            provided_segments: 提供的文本片段列表
            audio_duration: 音频总时长（秒）
        
        Returns:
            对齐后的结果字典
        """
        if not result or "segments" not in result:
            return result
        
        asr_segments = result.get("segments", [])
        if not asr_segments:
            return result
        
        # 如果提供的 segments 数量与识别结果数量相同，直接使用识别结果的时间
        if len(provided_segments) == len(asr_segments):
            # 保留识别结果的时间，但可以选择使用提供的文本
            # 这里我们保留识别结果，因为后续会通过 apply_script_text_override 处理文本
            return result
        
        # 如果数量不同，需要重新分配时间
        # 使用 allocate_segments_by_duration 方法按文本长度分配时间
        segment_entries = self.allocate_segments_by_duration(
            audio_duration, provided_segments
        )
        
        # 构建新的 segments
        new_segments = []
        for start, end, text in segment_entries:
            new_segments.append({
                "start": start,
                "end": end,
                "text": text
            })
        
        result["segments"] = new_segments
        return result
    
    def _convert_to_simplified(self, text: str) -> str:
        """将繁体中文转换为简体中文"""
        try:
            import zhconv
            return zhconv.convert(text, 'zh-cn')
        except ImportError:
            try:
                from opencc import OpenCC
                cc = OpenCC('t2s')  # Traditional to Simplified
                return cc.convert(text)
            except ImportError:
                # 如果都没有安装，返回原文本
                return text
    
    def save_srt(self, result: dict, output_path: str):
        """保存为 SRT 格式"""
        # 检查是否需要转换为简体
        convert_to_simplified = self.subtitle_config.get("convert_to_simplified", True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"], 1):
                start = self.format_timestamp(segment["start"])
                end = self.format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                # 转换为简体中文（如果启用）
                if convert_to_simplified:
                    text = self._convert_to_simplified(text)
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def save_vtt(self, result: dict, output_path: str):
        """保存为 VTT 格式"""
        # 检查是否需要转换为简体
        convert_to_simplified = self.subtitle_config.get("convert_to_simplified", True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in result["segments"]:
                start = self.format_timestamp(segment["start"], vtt=True)
                end = self.format_timestamp(segment["end"], vtt=True)
                text = segment["text"].strip()
                
                # 转换为简体中文（如果启用）
                if convert_to_simplified:
                    text = self._convert_to_simplified(text)
                
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def save_ass(self, result: dict, output_path: str):
        """保存为 ASS 格式"""
        font_config = self.subtitle_config.get('font', {})
        # 检查是否需要转换为简体
        convert_to_simplified = self.subtitle_config.get("convert_to_simplified", True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # ASS 文件头
            f.write("[Script Info]\n")
            f.write("Title: Generated Subtitles\n")
            f.write("ScriptType: v4.00+\n\n")
            
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write(f"Style: Default,{font_config.get('family', 'SimHei')},{font_config.get('size', 24)},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{font_config.get('outline_width', 2)},0,2,10,10,10,1\n\n")
            
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            for segment in result["segments"]:
                start = self.format_timestamp_ass(segment["start"])
                end = self.format_timestamp_ass(segment["end"])
                text = segment["text"].strip()
                
                # 转换为简体中文（如果启用）
                if convert_to_simplified:
                    text = self._convert_to_simplified(text)
                
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
    
    def format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """格式化时间戳"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def format_timestamp_ass(self, seconds: float) -> str:
        """格式化 ASS 时间戳"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def parse_srt_time(self, time_str: str) -> float:
        """解析SRT时间戳为秒数（格式: 00:00:00,000）"""
        try:
            time_str = time_str.strip()
            time_parts = time_str.replace(',', '.').split(':')
            if len(time_parts) == 3:
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                secs_parts = time_parts[2].split('.')
                seconds = int(secs_parts[0])
                millis = int(secs_parts[1]) if len(secs_parts) > 1 else 0
                total_seconds = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
                return total_seconds
        except Exception:
            pass
        return 0.0
    
    def merge_srt_files(self, srt_paths: List[str], output_path: str, audio_durations: Optional[List[float]] = None) -> str:
        """合并多个SRT字幕文件，累加前面的时长偏移
        
        Args:
            srt_paths: SRT文件路径列表
            output_path: 输出SRT文件路径
            audio_durations: 每个音频段的时长列表（用于验证和调整，可选）
        
        Returns:
            合并后的SRT文件路径
        """
        if not srt_paths:
            raise ValueError("SRT文件列表不能为空")
        
        all_segments = []
        current_offset = 0.0  # 当前累积的时间偏移
        subtitle_index = 1  # 字幕序号（从1开始）
        
        # 检查是否需要转换为简体
        convert_to_simplified = self.subtitle_config.get("convert_to_simplified", True)
        
        for idx, srt_path in enumerate(srt_paths):
            if not os.path.exists(srt_path):
                print(f"  ⚠ 字幕文件不存在，跳过: {srt_path}")
                # 如果提供了音频时长，累加偏移
                if audio_durations and idx < len(audio_durations):
                    current_offset += audio_durations[idx]
                continue
            
            # 读取SRT文件
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析SRT内容
            blocks = content.strip().split('\n\n')
            segment_count = 0
            segment_duration = 0.0  # 当前字幕文件的时长（最后一段的结束时间）
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # 解析时间轴（第二行：00:00:00,000 --> 00:00:00,000）
                time_line = lines[1].strip()
                if '-->' not in time_line:
                    continue
                
                start_str, end_str = time_line.split('-->')
                start_time = self.parse_srt_time(start_str.strip())
                end_time = self.parse_srt_time(end_str.strip())
                
                # 更新当前字幕文件的时长（使用最后一段的结束时间）
                if end_time > segment_duration:
                    segment_duration = end_time
                
                # 累加时间偏移
                new_start = start_time + current_offset
                new_end = end_time + current_offset
                
                # 获取文本（第三行及之后）
                text = '\n'.join(lines[2:]).strip()
                
                # 转换为简体中文（如果启用）
                if convert_to_simplified:
                    text = self._convert_to_simplified(text)
                
                all_segments.append({
                    'index': subtitle_index,
                    'start': new_start,
                    'end': new_end,
                    'text': text
                })
                
                subtitle_index += 1
                segment_count += 1
            
            # 累加当前段的时长偏移
            # 优先使用提供的音频时长（取整后的时长，确保与视频对应）
            # 如果没有提供，使用字幕文件中最后一段的结束时间
            if audio_durations and idx < len(audio_durations):
                # 使用提供的取整后的时长，确保与视频对应
                current_offset += audio_durations[idx]
                segment_duration = audio_durations[idx]
            elif segment_count > 0 and segment_duration > 0:
                # 如果没有提供音频时长，使用该字幕文件的时长（最后一段的结束时间）
                current_offset += segment_duration
            
            print(f"  ✓ 合并字幕段 {idx+1}/{len(srt_paths)}: {segment_count} 条字幕，段时长 {segment_duration:.2f}s，累积偏移 {current_offset:.2f}s")
        
        # 保存合并后的SRT文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in all_segments:
                start_time = self.format_timestamp(segment['start'])
                end_time = self.format_timestamp(segment['end'])
                f.write(f"{segment['index']}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
        
        print(f"  ✓ 字幕合并完成: 共 {len(all_segments)} 条字幕，总时长 {current_offset:.2f}s")
        return output_path

    def get_audio_duration(self, audio_path: str) -> float:
        try:
            audio, sr = sf.read(audio_path)
            return len(audio) / sr
        except Exception:
            return 1.0

    def build_segments_from_text(self, narration: str, delimiters: Optional[str] = None) -> List[str]:
        narration = narration.strip()
        if not narration:
            return []
        delimiter_chars = delimiters or self.script_config.get("delimiters") or "。！？!?；;>\n"
        delimiter_set = set(delimiter_chars)
        segments = []
        current = []
        for char in narration:
            current.append(char)
            if char in delimiter_set:
                segments.append("".join(current).strip())
                current = []
        if current:
            segments.append("".join(current).strip())
        return [s for s in segments if s]

    def allocate_segments_by_duration(
        self,
        total_duration: float,
        texts: List[str],
        min_duration: float = 0.8,
        gap: float = 0.1,
    ) -> List[tuple]:
        """根据音频时长按文本长度比例分配时间"""
        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            cleaned = ["(配音)"]
        total_chars = sum(max(len(t), 1) for t in cleaned)
        start = 0.0
        segments = []
        for idx, text in enumerate(cleaned):
            weight = max(len(text), 1) / total_chars if total_chars else 1.0 / len(cleaned)
            duration = max(total_duration * weight, min_duration)
            end = start + duration
            if idx == len(cleaned) - 1:
                end = max(total_duration, end)
            segments.append((max(start, 0.0), min(end, total_duration), text))
            start = end + gap
            if start >= total_duration:
                start = total_duration
        # 修正最后结束时间
        if segments:
            last_start, _, last_text = segments[-1]
            segments[-1] = (last_start, total_duration, last_text)
        return segments
    
    def allocate_segments_by_video_durations(
        self,
        total_audio_duration: float,
        texts: List[str],
        video_durations: List[float],
        gap: float = 0.0,  # 默认不添加间隔，确保时长严格对应
    ) -> List[tuple]:
        """根据视频片段时长分配字幕时间（更准确的对齐方式）
        
        Args:
            total_audio_duration: 音频总时长（秒）- 应该是所有分段音频的总和
            texts: 字幕文本列表（对应每个视频片段）
            video_durations: 每个视频片段的时长列表（秒）- 实际是音频时长列表
            gap: 字幕之间的间隔（秒），默认0，确保时长严格对应
        
        Returns:
            List[tuple]: [(start, end, text), ...]
        """
        if len(texts) != len(video_durations):
            print(f"  ⚠ 警告: 文本数量 ({len(texts)}) 与视频时长数量 ({len(video_durations)}) 不一致，回退到文本长度比例分配")
            return self.allocate_segments_by_duration(total_audio_duration, texts)
        
        # 计算视频总时长（实际是音频总时长）
        total_video_duration = sum(video_durations)
        
        # 验证：如果传入的 total_audio_duration 与 video_durations 总和差异较大，使用 video_durations 总和
        # 因为 video_durations 是准确的音频时长列表
        if abs(total_audio_duration - total_video_duration) > 0.5:
            print(f"  ⚠ 传入的总时长 ({total_audio_duration:.2f}s) 与分段时长总和 ({total_video_duration:.2f}s) 不一致，使用分段时长总和")
            total_audio_duration = total_video_duration
        
        # 直接使用 video_durations（即 audio_durations）作为字幕时长，不缩放，不添加间隔
        # 确保每个字幕片段的时间严格对应每个分段音频的时长
        start = 0.0
        segments = []
        for idx, (text, audio_dur) in enumerate(zip(texts, video_durations)):
            if not text or not text.strip():
                continue
            
            # 直接使用取整后的音频时长作为字幕时长，确保与视频片段严格对应
            # audio_dur 已经是取整后的整数秒（通过 round() 四舍五入）
            # 视频生成时也使用相同的 audio_durations，所以字幕时间 = 视频时间 = 音频时间
            segment_duration = max(audio_dur, 0.1)  # 最少0.1秒（但通常 audio_dur 已经是整数 >= 1）
            end = start + segment_duration
            
            # 确保不超过总时长（但理论上不应该超过，因为总时长就是所有片段的总和）
            if end > total_audio_duration:
                end = total_audio_duration
                print(f"  ⚠ 字幕片段 {idx+1} 结束时间 ({end:.2f}s) 超过总时长 ({total_audio_duration:.2f}s)，已截断")
            
            segments.append((max(start, 0.0), min(end, total_audio_duration), text.strip()))
            
            # 下一个字幕的开始时间（不添加间隔，直接接续，确保时长严格对应）
            # 这样每个字幕片段的时间 = 对应视频片段的时间 = 对应音频片段的时间（都是取整后的整数秒）
            start = end
            if start >= total_audio_duration:
                break
        
        # 修正最后结束时间，确保覆盖到音频结束（使用实际总时长）
        if segments:
            last_start, _, last_text = segments[-1]
            segments[-1] = (last_start, total_audio_duration, last_text)
            print(f"  字幕时间分配: 共 {len(segments)} 个片段，总时长 {total_audio_duration:.2f}s")
            print(f"  ✓ 每个字幕片段的时间严格对应视频片段和音频片段（都使用取整后的整数秒）")
        
        return segments


def main():
    parser = argparse.ArgumentParser(description="字幕生成")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--audio", type=str, required=True, help="输入音频路径")
    parser.add_argument("--output", type=str, help="输出字幕路径")
    parser.add_argument("--format", type=str, choices=["srt", "vtt", "ass"], help="字幕格式")
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = SubtitleGenerator(args.config)
    
    # 生成字幕
    output_path = args.output or generator.config['paths']['output_dir'] / "subtitle.srt"
    generator.generate(args.audio, str(output_path), format=args.format)


if __name__ == "__main__":
    main()



