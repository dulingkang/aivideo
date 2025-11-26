#!/usr/bin/env python3
"""
字幕转写脚本：使用WhisperX转写整集视频，生成带时间戳的字幕
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from typing import List, Dict

try:
    import whisperx
    import torch
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("警告: whisperx 未安装")

def transcribe_video(video_path, output_json, language='zh', device='cuda', 
                    model_name='medium', compute_type='int8', align=False):
    """
    使用WhisperX转写视频，生成带时间戳的字幕
    
    Args:
        video_path: 视频路径
        output_json: 输出JSON路径
        language: 语言代码，'zh'为中文
        device: 设备 'cuda' 或 'cpu'
        model_name: Whisper模型名称
        compute_type: 计算类型 'int8', 'float16', 'float32'
        align: 是否进行时间戳对齐
    """
    if not WHISPERX_AVAILABLE:
        raise ImportError("需要安装: pip install whisperx")
    
    video_path = Path(video_path)
    output_json = Path(output_json)
    
    # 禁用VAD以避免段错误（必须在模型加载之前设置）
    os.environ["WHISPERX_VAD"] = "false"
    os.environ.pop("WHISPERX_VAD_MODEL", None)
    # 确保 VAD 被禁用
    if "WHISPERX_VAD" in os.environ:
        print(f"✓ VAD 已禁用: {os.environ.get('WHISPERX_VAD')}")
    
    print(f"加载WhisperX模型: {model_name} (语言: {language}, 设备: {device})")
    
    # 内存优化：清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    
    try:
        # 尝试使用本地模型目录（如果存在）
        model_dir = Path("gen_video/models/faster-whisper-{}".format(model_name))
        model_source = str(model_dir) if model_dir.exists() else model_name
        
        model = whisperx.load_model(
            model_source,
            device,
            compute_type=compute_type,
            language=language,
            asr_options={"best_of": 1},
            local_files_only=model_dir.exists()
        )
        print(f"✓ 模型加载成功")
    except Exception as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "bad_alloc" in error_msg:
            print(f"⚠ 显存不足，尝试使用 int8 量化或 CPU 模式...")
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            # 尝试 int8 量化
            if compute_type != "int8" and device != "cpu":
                try:
                    print(f"  尝试 int8 量化...")
                    model = whisperx.load_model(
                        model_source,
                        device,
                        compute_type="int8",
                        language=language,
                        asr_options={"best_of": 1},
                        local_files_only=model_dir.exists()
                    )
                    print(f"✓ 模型加载成功（int8 量化）")
                    compute_type = "int8"  # 更新 compute_type
                except Exception as e2:
                    print(f"  int8 量化也失败，尝试 CPU 模式...")
                    # 最后尝试 CPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                    model = whisperx.load_model(
                        model_source,
                        "cpu",
                        compute_type="int8",
                        language=language,
                        asr_options={"best_of": 1},
                        local_files_only=model_dir.exists()
                    )
                    print(f"✓ 模型加载成功（CPU 模式）")
                    device = "cpu"  # 更新 device
        else:
            print(f"错误: 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print(f"转写视频: {video_path}")
    
    # 加载音频
    try:
        audio = whisperx.load_audio(str(video_path))
        print(f"✓ 音频加载成功，时长: {len(audio) / 16000:.2f} 秒")
    except Exception as e:
        print(f"错误: 音频加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 执行转写
    print("执行转写...")
    print(f"  音频长度: {len(audio)} 采样点 ({len(audio) / 16000:.2f} 秒)")
    print(f"  设备: {device}, 计算类型: {compute_type}")
    
    try:
        # 根据设备调整 batch_size
        batch_size = 16 if device == "cuda" else 4
        if compute_type == "int8":
            batch_size = min(batch_size, 8)  # int8 模式使用更小的 batch
        
        print(f"  使用 batch_size: {batch_size}")
        print("  开始转写（这可能需要一些时间）...")
        
        # 执行转写（不传递 VAD 参数，依赖环境变量禁用 VAD）
        # 注意：某些版本的 WhisperX 可能不支持 vad_onset/vad_offset 参数
        result = model.transcribe(
            audio, 
            batch_size=batch_size
        )
        
        # 检查结果
        if not result:
            raise ValueError("转写返回空结果")
        
        segments = result.get('segments', [])
        if not segments:
            print("⚠ 警告: 转写完成但没有识别到任何片段")
            # 仍然创建文件，但包含空片段列表
        else:
            print(f"✓ 转写完成，共 {len(segments)} 个片段")
            
    except KeyboardInterrupt:
        print("\n⚠ 转写被用户中断")
        raise
    except Exception as e:
        print(f"\n错误: 转写失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果是内存错误，提供建议
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "bad_alloc" in error_msg:
            print("\n建议:")
            print("  1. 使用更小的模型: --model small 或 --model base")
            print("  2. 使用 CPU 模式: --device cpu")
            print("  3. 使用 int8 量化: --compute-type int8")
            print("  4. 清理显存后重试")
        elif "cuda" in error_msg or "gpu" in error_msg:
            print("\n建议:")
            print("  1. 尝试使用 CPU 模式: --device cpu")
            print("  2. 检查 CUDA 驱动和 PyTorch 版本兼容性")
        
        raise
    
    # 对齐时间戳（可选，可能失败）
    if align:
        try:
            print("进行时间戳对齐...")
            model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        except Exception as e:
            print(f"警告: 时间戳对齐失败，使用原始结果: {e}")
    
    # 提取字幕段
    subtitle_segments = []
    for segment in result.get("segments", []):
        subtitle_segments.append({
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "text": segment.get("text", "").strip(),
            "words": segment.get("words", [])  # 字词级时间戳（如果有）
        })
    
    # 保存结果
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "video_path": str(video_path),
        "language": language,
        "model": model_name,
        "total_segments": len(subtitle_segments),
        "segments": subtitle_segments
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 转写完成: {output_json}")
    print(f"  共 {len(subtitle_segments)} 个字幕段")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description='使用WhisperX转写视频字幕')
    parser.add_argument('--input', '-i', required=True, help='输入视频路径')
    parser.add_argument('--output', '-o', required=True, help='输出JSON路径')
    parser.add_argument('--language', '-l', default='zh', help='语言代码 (zh/en/ja等)')
    parser.add_argument('--device', '-d', default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--model', '-m', default='medium', 
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper模型大小')
    parser.add_argument('--compute-type', default='int8',
                       choices=['int8', 'float16', 'float32'],
                       help='计算类型（int8更省内存）')
    parser.add_argument('--align', action='store_true',
                       help='启用时间戳对齐（可能失败，建议不启用）')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 视频文件不存在: {input_path}")
        return 1
    
    try:
        transcribe_video(
            args.input,
            args.output,
            language=args.language,
            device=args.device,
            model_name=args.model,
            compute_type=args.compute_type,
            align=args.align
        )
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

