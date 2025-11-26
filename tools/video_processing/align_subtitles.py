#!/usr/bin/env python3
"""
字幕对齐脚本：将字幕段对齐到镜头片段
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

def load_scene_list(scene_list_path):
    """
    加载场景列表（PySceneDetect生成的CSV或自定义JSON）
    
    支持格式：
    1. CSV格式（PySceneDetect默认）: Timecode,Scene Number
    2. JSON格式: [{"start": 0.0, "end": 5.2, "scene_id": "scene_000"}, ...]
    3. JSON格式（split_scenes.py生成）: {"total_scenes": N, "scenes": [...]}
    """
    scene_list_path = Path(scene_list_path)
    
    if scene_list_path.suffix == '.json':
        with open(scene_list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否是 split_scenes.py 生成的格式（包含 "scenes" 键）
        if isinstance(data, dict) and "scenes" in data:
            return data["scenes"]
        # 如果是数组格式，直接返回
        elif isinstance(data, list):
            return data
        else:
            # 其他格式，尝试提取场景列表
            raise ValueError(f"无法识别的场景列表格式: {scene_list_path}")
    else:
        # 尝试解析CSV
        scenes = []
        with open(scene_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 跳过标题行
            for i, line in enumerate(lines[1:], 1):
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        timecode = float(parts[0])
                        scene_id = f"scene_{i-1:03d}"
                        # 下一个场景的开始时间作为当前场景的结束时间
                        next_timecode = float(lines[i+1].split(',')[0]) if i < len(lines)-1 else None
                        scenes.append({
                            "start": timecode,
                            "end": next_timecode,
                            "scene_id": scene_id
                        })
                    except ValueError:
                        continue
        return scenes

def load_subtitles(subtitle_json):
    """加载字幕JSON（支持WhisperX和OCR两种格式）"""
    with open(subtitle_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查格式：WhisperX格式有"segments"，OCR格式也有"segments"
    segments = data.get("segments", [])
    
    # 确保每个segment都有必要的字段
    normalized_segments = []
    for seg in segments:
        normalized_seg = {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
            "words": seg.get("words", [])  # OCR可能没有words字段
        }
        if normalized_seg["text"]:  # 只保留有文本的段
            normalized_segments.append(normalized_seg)
    
    return normalized_segments

def align_subtitles_to_scenes(subtitle_segments, scene_list, output_json, time_offset=0.0):
    """
    将字幕段对齐到镜头
    
    Args:
        subtitle_segments: 字幕段列表（时间戳基于原始视频）
        scene_list: 场景列表（时间戳基于清洗后的视频）
        output_json: 输出JSON路径
        time_offset: 时间偏移（秒），字幕时间戳需要减去这个值才能对齐到清洗后的场景
                    例如：如果trim_start=147，则time_offset=147
    """
    # 为每个场景创建字幕列表
    scene_subtitles = {}
    for scene in scene_list:
        scene_id = scene.get("scene_id", scene.get("id", "unknown"))
        scene_start = scene.get("start", 0.0)
        scene_end = scene.get("end", float('inf'))
        
        scene_subtitles[scene_id] = {
            "scene_id": scene_id,
            "start": scene_start,
            "end": scene_end,
            "subtitles": []
        }
    
    # 将字幕段分配到对应的场景
    for subtitle in subtitle_segments:
        # 将字幕时间戳从原始视频时间转换为清洗后视频时间
        sub_start = subtitle.get("start", 0.0) - time_offset
        sub_end = subtitle.get("end", 0.0) - time_offset
        sub_text = subtitle.get("text", "").strip()
        
        if not sub_text:
            continue
        
        # 过滤掉时间偏移后完全不在有效范围内的字幕（trim_start之前的字幕）
        if sub_end < 0:
            continue  # 字幕完全在被裁剪的开头部分，跳过
        
        # 如果字幕跨越了trim_start点，截断开始时间到0
        if sub_start < 0:
            sub_start = 0.0  # 截断到清洗后视频的开始时间
        
        # 确保结束时间也是非负数（理论上不应该出现，但作为保险）
        if sub_end < 0:
            continue
        
        # 找到字幕所属的场景（字幕时间与场景时间有重叠）
        for scene in scene_list:
            scene_id = scene.get("scene_id", scene.get("id", "unknown"))
            scene_start = scene.get("start", 0.0)
            scene_end = scene.get("end", float('inf'))
            
            # 检查时间重叠（使用偏移后的字幕时间）
            if sub_start < scene_end and sub_end > scene_start:
                scene_subtitles[scene_id]["subtitles"].append({
                    "start": sub_start,  # 此时sub_start已经确保>=0
                    "end": sub_end,      # 此时sub_end已经确保>=0
                    "text": sub_text,
                    "words": subtitle.get("words", [])
                })
                break  # 一个字幕段只分配给一个场景（第一个匹配的）
    
    # 合并字幕文本（用于描述）
    for scene_id, scene_data in scene_subtitles.items():
        subtitle_texts = [s["text"] for s in scene_data["subtitles"]]
        scene_data["combined_text"] = " ".join(subtitle_texts)
        scene_data["subtitle_count"] = len(scene_data["subtitles"])
    
    # 保存结果
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "total_scenes": len(scene_subtitles),
            "scenes": scene_subtitles
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 对齐完成: {output_json}")
    print(f"  共 {len(scene_subtitles)} 个场景")
    
    # 统计信息
    scenes_with_subtitles = sum(1 for s in scene_subtitles.values() if s["subtitles"])
    print(f"  其中 {scenes_with_subtitles} 个场景有字幕")
    
    return scene_subtitles

def main():
    parser = argparse.ArgumentParser(description='将字幕对齐到镜头片段')
    parser.add_argument('--subtitles', '-s', required=True, help='字幕JSON路径')
    parser.add_argument('--scenes', required=True, help='场景列表路径（CSV或JSON）')
    parser.add_argument('--output', '-o', required=True, help='输出JSON路径')
    parser.add_argument('--time-offset', type=float, default=0.0,
                       help='时间偏移（秒），字幕时间戳需要减去这个值才能对齐到清洗后的场景。例如：如果trim_start=147，则--time-offset=147')
    
    args = parser.parse_args()
    
    subtitle_path = Path(args.subtitles)
    scene_path = Path(args.scenes)
    
    if not subtitle_path.exists():
        print(f"错误: 字幕文件不存在: {subtitle_path}")
        return 1
    
    if not scene_path.exists():
        print(f"错误: 场景列表文件不存在: {scene_path}")
        return 1
    
    try:
        subtitle_segments = load_subtitles(subtitle_path)
        scene_list = load_scene_list(scene_path)
        
        if args.time_offset > 0:
            print(f"应用时间偏移: {args.time_offset}秒（字幕时间戳将减去此值以对齐到清洗后的场景）")
        
        align_subtitles_to_scenes(subtitle_segments, scene_list, args.output, args.time_offset)
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

