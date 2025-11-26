#!/usr/bin/env python3
"""
使用OCR提取视频字幕：从视频帧中识别硬编码字幕
支持 PaddleOCR 和 EasyOCR
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

def detect_subtitle_region(frame, bottom_ratio=0.15):
    """
    检测字幕区域（通常在视频底部）
    
    Args:
        frame: 视频帧（numpy array）
        bottom_ratio: 底部区域比例（默认15%）
    
    Returns:
        (x, y, w, h) 字幕区域坐标
    """
    h, w = frame.shape[:2]
    # 字幕通常在底部15%区域
    subtitle_y = int(h * (1 - bottom_ratio))
    subtitle_h = int(h * bottom_ratio)
    
    return (0, subtitle_y, w, subtitle_h)

def preprocess_frame(frame, region=None):
    """
    预处理帧以提高OCR识别率
    
    Args:
        frame: 原始帧
        region: 字幕区域 (x, y, w, h)，如果为None则使用整个帧
    
    Returns:
        预处理后的图像
    """
    if region:
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]
    else:
        roi = frame
    
    # 转换为灰度图
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 二值化（可选，某些OCR模型不需要）
    # _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return enhanced

def extract_frames(video_path, output_dir, fps=1.0):
    """
    从视频中提取帧（用于OCR识别）
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        fps: 提取帧率（每秒提取多少帧，默认1帧/秒）
    
    Returns:
        提取的帧文件列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用ffmpeg提取帧
    frame_pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-q:v', '2',  # 高质量
        '-y',
        frame_pattern
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = result.stderr
        print(f"警告: 帧提取失败: {error_msg}")
        
        # 检查是否是文件损坏问题
        if "moov atom not found" in error_msg or "Invalid data" in error_msg:
            print("错误: 视频文件可能损坏或不完整")
            print(f"  文件路径: {video_path}")
            print(f"  建议: 检查视频文件是否完整，或使用原始视频文件")
            raise ValueError(f"视频文件损坏或不完整: {video_path}")
        
        # 其他错误也抛出异常
        raise RuntimeError(f"FFmpeg提取帧失败: {error_msg}")
    
    # 返回所有提取的帧文件
    frames = sorted(output_dir.glob('frame_*.jpg'))
    if len(frames) == 0:
        raise RuntimeError(f"未能提取任何帧，请检查视频文件: {video_path}")
    
    return frames

def extract_subtitles_paddleocr(video_path, output_json, fps=1.0, 
                                 ocr_model=None, subtitle_region=None):
    """
    使用PaddleOCR提取字幕
    
    Args:
        video_path: 视频路径
        output_json: 输出JSON路径
        fps: 提取帧率
        ocr_model: PaddleOCR模型实例（如果已加载）
        subtitle_region: 字幕区域 (x, y, w, h)
    """
    if not PADDLEOCR_AVAILABLE:
        raise ImportError("需要安装: pip install paddleocr")
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 检查文件大小（如果太小可能损坏）
    file_size = video_path.stat().st_size
    if file_size < 1024 * 100:  # 小于100KB可能有问题
        print(f"警告: 视频文件很小 ({file_size} bytes)，可能损坏或不完整")
    
    # 初始化OCR模型（中文+英文）
    if ocr_model is None:
        print("加载PaddleOCR模型（中文）...")
        # 新版本PaddleOCR参数已更新：移除use_gpu，use_angle_cls改为use_textline_orientation
        try:
            # 尝试新版本API
            ocr_model = PaddleOCR(
                use_textline_orientation=True,  # 替代 use_angle_cls
                lang='ch'  # 中文，GPU会自动检测
            )
        except TypeError:
            # 如果新参数不支持，尝试旧版本API
            try:
                ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')
            except Exception as e:
                print(f"警告: 使用最简参数初始化: {e}")
                # 使用最简参数
                ocr_model = PaddleOCR(lang='ch')
        print("✓ 模型加载完成")
    
    video_path = Path(video_path)
    output_json = Path(output_json)
    
    # 创建临时目录存储帧
    temp_dir = output_json.parent / "ocr_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"提取视频帧（{fps} fps）...")
    frames = extract_frames(video_path, temp_dir, fps)
    print(f"✓ 提取了 {len(frames)} 帧")
    
    # 获取视频时长
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
    ]
    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    total_duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
    
    print("开始OCR识别...")
    segments = []
    last_text = ""
    segment_start = 0.0
    
    for i, frame_path in enumerate(frames):
        # 计算当前帧的时间戳
        timestamp = (i / fps) if fps > 0 else 0.0
        
        # 读取帧
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # 检测字幕区域
        if subtitle_region is None:
            subtitle_region = detect_subtitle_region(frame)
        
        # 预处理
        processed = preprocess_frame(frame, subtitle_region)
        
        # OCR识别
        try:
            results = ocr_model.ocr(processed, cls=True)
            
            # 提取文本
            texts = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                        confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                        if confidence > 0.5:  # 置信度阈值
                            texts.append(text)
            
            current_text = " ".join(texts).strip()
            
            # 如果文本变化，创建新的字幕段
            if current_text and current_text != last_text:
                # 结束上一个段
                if last_text and segment_start < timestamp:
                    segments.append({
                        "start": segment_start,
                        "end": timestamp,
                        "text": last_text
                    })
                
                # 开始新段
                segment_start = timestamp
                last_text = current_text
                
        except Exception as e:
            print(f"  警告: 帧 {i} OCR失败: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  处理进度: {i+1}/{len(frames)}")
    
    # 添加最后一个段
    if last_text:
        segments.append({
            "start": segment_start,
            "end": total_duration,
            "text": last_text
        })
    
    # 合并相同文本的连续段
    merged_segments = []
    for seg in segments:
        if merged_segments and merged_segments[-1]["text"] == seg["text"]:
            # 合并到上一个段
            merged_segments[-1]["end"] = seg["end"]
        else:
            merged_segments.append(seg)
    
    # 保存结果
    output_data = {
        "video_path": str(video_path),
        "method": "paddleocr",
        "total_segments": len(merged_segments),
        "segments": merged_segments
    }
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ OCR提取完成: {output_json}")
    print(f"  共 {len(merged_segments)} 个字幕段")
    
    # 清理临时文件
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"✓ 清理临时文件: {temp_dir}")
    
    return output_data

def extract_subtitles_easyocr(video_path, output_json, fps=1.0, 
                              ocr_reader=None, subtitle_region=None):
    """
    使用EasyOCR提取字幕（备选方案，不依赖PaddlePaddle）
    """
    if not EASYOCR_AVAILABLE:
        raise ImportError("需要安装: pip install easyocr")
    
    video_path = Path(video_path)
    output_json = Path(output_json)
    
    # 初始化OCR模型（中文+英文）
    if ocr_reader is None:
        print("加载EasyOCR模型（中文+英文）...")
        try:
            # 尝试使用GPU
            ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())
        except Exception as e:
            print(f"警告: GPU初始化失败，使用CPU: {e}")
            ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        print("✓ 模型加载完成")
    
    # 创建临时目录存储帧
    temp_dir = output_json.parent / "ocr_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"提取视频帧（{fps} fps）...")
    frames = extract_frames(video_path, temp_dir, fps)
    print(f"✓ 提取了 {len(frames)} 帧")
    
    # 获取视频时长
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
    ]
    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    total_duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
    
    print("开始OCR识别...")
    segments = []
    last_text = ""
    segment_start = 0.0
    
    for i, frame_path in enumerate(frames):
        # 计算当前帧的时间戳
        timestamp = (i / fps) if fps > 0 else 0.0
        
        # 读取帧
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # 检测字幕区域
        if subtitle_region is None:
            subtitle_region = detect_subtitle_region(frame)
        
        # 预处理
        processed = preprocess_frame(frame, subtitle_region)
        
        # OCR识别
        try:
            results = ocr_reader.readtext(processed)
            
            # 提取文本
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 置信度阈值
                    texts.append(text)
            
            current_text = " ".join(texts).strip()
            
            # 如果文本变化，创建新的字幕段
            if current_text and current_text != last_text:
                # 结束上一个段
                if last_text and segment_start < timestamp:
                    segments.append({
                        "start": segment_start,
                        "end": timestamp,
                        "text": last_text
                    })
                
                # 开始新段
                segment_start = timestamp
                last_text = current_text
                
        except Exception as e:
            print(f"  警告: 帧 {i} OCR失败: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  处理进度: {i+1}/{len(frames)}")
    
    # 添加最后一个段
    if last_text:
        segments.append({
            "start": segment_start,
            "end": total_duration,
            "text": last_text
        })
    
    # 合并相同文本的连续段
    merged_segments = []
    for seg in segments:
        if merged_segments and merged_segments[-1]["text"] == seg["text"]:
            # 合并到上一个段
            merged_segments[-1]["end"] = seg["end"]
        else:
            merged_segments.append(seg)
    
    # 保存结果
    output_data = {
        "video_path": str(video_path),
        "method": "easyocr",
        "total_segments": len(merged_segments),
        "segments": merged_segments
    }
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ OCR提取完成: {output_json}")
    print(f"  共 {len(merged_segments)} 个字幕段")
    
    # 清理临时文件
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"✓ 清理临时文件: {temp_dir}")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description='使用OCR提取视频字幕')
    parser.add_argument('--input', '-i', required=True, help='输入视频路径')
    parser.add_argument('--output', '-o', required=True, help='输出JSON路径')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='提取帧率（每秒提取多少帧，默认1.0）')
    parser.add_argument('--method', choices=['paddleocr', 'easyocr'], 
                       default='easyocr', help='OCR方法（默认easyocr，不依赖PaddlePaddle）')
    parser.add_argument('--subtitle-region', help='字幕区域 x:y:w:h（可选，自动检测）')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 视频文件不存在: {input_path}")
        return 1
    
    subtitle_region = None
    if args.subtitle_region:
        parts = args.subtitle_region.split(':')
        if len(parts) == 4:
            subtitle_region = tuple(map(int, parts))
    
    try:
        if args.method == 'paddleocr':
            if not PADDLEOCR_AVAILABLE:
                print("错误: PaddleOCR未安装")
                print("安装方法: pip install paddleocr")
                return 1
            extract_subtitles_paddleocr(
                args.input, args.output, args.fps, 
                subtitle_region=subtitle_region
            )
        elif args.method == 'easyocr':
            if not EASYOCR_AVAILABLE:
                print("错误: EasyOCR未安装")
                print("安装方法: pip install easyocr")
                return 1
            extract_subtitles_easyocr(
                args.input, args.output, args.fps,
                subtitle_region=subtitle_region
            )
        
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

