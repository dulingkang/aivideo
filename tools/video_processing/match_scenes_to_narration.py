#!/usr/bin/env python3
"""
根据narration文本和时长，匹配检索到的场景视频
策略：
1. 如果单个视频太长：裁剪到narration时长
2. 如果单个视频太短：从检索结果中按顺序拼接多个场景，直到满足时长
3. 如果top-k都不够：可以使用多个检索结果的组合
"""

import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil

def get_video_duration(video_path: Path) -> float:
    """获取视频时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

def find_scene_video(episode_id: str, scene_id: str, base_dir: Path) -> Optional[Path]:
    """查找场景对应的视频文件"""
    # 提取纯场景ID
    if '_' in scene_id and scene_id.split('_')[0].isdigit():
        _, pure_scene_id = scene_id.split('_', 1)
    else:
        pure_scene_id = scene_id
    
    episode_dir = base_dir / f"episode_{episode_id}" / "scenes"
    
    # 查找视频文件
    patterns = [
        f"*{pure_scene_id}*.mp4",
        f"*Scene-{pure_scene_id.split('_')[-1]}*.mp4",
        f"*{pure_scene_id.split('_')[-1].zfill(3)}*.mp4",
    ]
    
    for pattern in patterns:
        matches = list(episode_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None

def trim_video(input_path: Path, output_path: Path, duration: float, mute: bool = True):
    """裁剪视频到指定时长（可选静音）"""
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-t', str(duration),
        '-c:v', 'libx264',  # 重新编码视频以确保兼容性
        '-preset', 'fast',
        '-crf', '23',
        '-y'
    ]
    
    if mute:
        cmd.append('-an')  # 去掉音频（静音）
    else:
        cmd.extend(['-c:a', 'aac'])  # 重新编码音频
    
    cmd.append(str(output_path))
    subprocess.run(cmd, capture_output=True, text=True, check=True)

def concatenate_videos(video_paths: List[Path], output_path: Path, mute: bool = True):
    """拼接多个视频文件（可选静音，统一添加BGM和旁白）"""
    if len(video_paths) == 1:
        # 只有一个视频，也需要重新编码以确保格式一致
        cmd = [
            'ffmpeg', '-i', str(video_paths[0]),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-y'
        ]
        if mute:
            cmd.append('-an')  # 去掉音频（静音）
        else:
            cmd.extend(['-c:a', 'aac'])
        cmd.append(str(output_path))
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return
    
    # 创建临时文件列表
    concat_file = output_path.parent / f"concat_{output_path.stem}.txt"
    with open(concat_file, 'w') as f:
        for video_path in video_paths:
            f.write(f"file '{video_path.absolute()}'\n")
    
    try:
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c:v', 'libx264',  # 重新编码视频以确保所有片段参数一致
            '-preset', 'fast',
            '-crf', '23',
            '-y'
        ]
        if mute:
            cmd.append('-an')  # 去掉音频（静音）
        else:
            cmd.extend(['-c:a', 'aac'])  # 重新编码音频
        
        cmd.append(str(output_path))
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        if concat_file.exists():
            concat_file.unlink()

def match_scene_to_duration(
    search_results: List[Dict],
    target_duration: float,
    base_dir: Path,
    tolerance: float = 0.5
) -> Tuple[List[Path], float, str]:
    """
    匹配场景视频到目标时长
    
    Args:
        search_results: 检索结果列表（按分数降序）
        target_duration: 目标时长（秒）
        base_dir: 视频基础目录
        tolerance: 时长容差（秒）
    
    Returns:
        (视频路径列表, 实际总时长, 匹配策略说明)
    """
    matched_videos = []
    total_duration = 0.0
    strategy = ""
    
    # 策略1: 尝试单个视频匹配
    for result in search_results:
        scene_data = result.get('scene_data', {})
        scene_id = result.get('scene_id', '')
        episode_id = scene_data.get('episode_id', '')
        
        video_path = find_scene_video(episode_id, scene_id, base_dir)
        if not video_path or not video_path.exists():
            continue
        
        video_duration = get_video_duration(video_path)
        
        # 如果视频时长接近目标时长（在容差范围内）
        if abs(video_duration - target_duration) <= tolerance:
            matched_videos = [video_path]
            total_duration = video_duration
            strategy = f"单视频完美匹配: {scene_id} ({video_duration:.2f}s)"
            break
        
        # 如果视频比目标时长短，但可以用
        if video_duration < target_duration and video_duration > target_duration * 0.5:
            matched_videos.append(video_path)
            total_duration += video_duration
            strategy = f"单视频部分匹配: {scene_id} ({video_duration:.2f}s)"
            
            # 继续查找更多视频拼接
            remaining = target_duration - total_duration
            
            for next_result in search_results:
                if next_result == result:
                    continue
                
                next_scene_data = next_result.get('scene_data', {})
                next_scene_id = next_result.get('scene_id', '')
                next_episode_id = next_scene_data.get('episode_id', '')
                
                next_video_path = find_scene_video(next_episode_id, next_scene_id, base_dir)
                if not next_video_path or not next_video_path.exists():
                    continue
                
                next_video_duration = get_video_duration(next_video_path)
                
                if total_duration + next_video_duration <= target_duration + tolerance:
                    matched_videos.append(next_video_path)
                    total_duration += next_video_duration
                    strategy += f" + {next_scene_id} ({next_video_duration:.2f}s)"
                
                if total_duration >= target_duration:
                    break
            
            break
    
    # 如果还没匹配到，使用策略2: 多个视频拼接
    if not matched_videos:
        for result in search_results:
            scene_data = result.get('scene_data', {})
            scene_id = result.get('scene_id', '')
            episode_id = scene_data.get('episode_id', '')
            
            video_path = find_scene_video(episode_id, scene_id, base_dir)
            if not video_path or not video_path.exists():
                continue
            
            video_duration = get_video_duration(video_path)
            
            if total_duration + video_duration <= target_duration + tolerance * 2:
                matched_videos.append(video_path)
                total_duration += video_duration
                if not strategy:
                    strategy = f"多视频拼接: {scene_id}"
                else:
                    strategy += f" + {scene_id}"
            
            if total_duration >= target_duration:
                break
    
    # 如果还没匹配到，使用策略3: 单个最长视频裁剪
    if not matched_videos and search_results:
        # 找到最长的视频
        best_video = None
        best_duration = 0.0
        best_scene_id = ""
        
        for result in search_results:
            scene_data = result.get('scene_data', {})
            scene_id = result.get('scene_id', '')
            episode_id = scene_data.get('episode_id', '')
            
            video_path = find_scene_video(episode_id, scene_id, base_dir)
            if not video_path or not video_path.exists():
                continue
            
            video_duration = get_video_duration(video_path)
            if video_duration > best_duration:
                best_video = video_path
                best_duration = video_duration
                best_scene_id = scene_id
        
        if best_video:
            matched_videos = [best_video]
            total_duration = min(best_duration, target_duration)
            strategy = f"单视频裁剪: {best_scene_id} ({best_duration:.2f}s -> {total_duration:.2f}s)"
    
    return matched_videos, total_duration, strategy

def process_narration_scenes(
    narration_file: Path,
    search_base_dir: Path,
    video_base_dir: Path,
    output_dir: Path,
    top_k: int = 5,
    tolerance: float = 0.5
):
    """
    处理narration文件，匹配场景视频
    
    narration_file格式:
    {
        "narration_parts": [
            {
                "text": "旁白文本",
                "duration": 5.0  # 时长（秒）
            },
            ...
        ]
    }
    """
    with open(narration_file, 'r', encoding='utf-8') as f:
        narration_data = json.load(f)
    
    narration_parts = narration_data.get('narration_parts', [])
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载检索结果（对每个narration段落）
    matched_scenes = []
    
    print(f"处理 {len(narration_parts)} 个narration段落...\n")
    
    for i, part in enumerate(narration_parts, 1):
        text = part.get('text', '')
        duration = part.get('duration', 0.0)
        
        print(f"[{i}/{len(narration_parts)}] 匹配: {text[:50]}... (时长: {duration:.2f}s)")
        
        # 检索相关场景
        # TODO: 这里需要调用search_scenes.py或直接使用检索功能
        # 暂时假设已经有检索结果文件
        search_result_file = search_base_dir / f"narration_{i}_search.json"
        
        if not search_result_file.exists():
            print(f"  ✗ 检索结果文件不存在: {search_result_file}")
            print(f"  提示: 需要先为每个narration段落运行检索")
            continue
        
        with open(search_result_file, 'r', encoding='utf-8') as f:
            search_data = json.load(f)
        
        search_results = search_data.get('results', [])[:top_k]
        
        if not search_results:
            print(f"  ✗ 没有检索结果")
            continue
        
        # 匹配视频
        matched_videos, actual_duration, strategy = match_scene_to_duration(
            search_results,
            duration,
            video_base_dir,
            tolerance
        )
        
        if not matched_videos:
            print(f"  ✗ 无法匹配视频")
            continue
        
        # 处理视频
        if len(matched_videos) == 1 and actual_duration > duration + tolerance:
            # 需要裁剪
            output_video = output_dir / f"scene_{i:03d}_trimmed.mp4"
            trim_video(matched_videos[0], output_video, duration)
            print(f"  ✓ 裁剪: {strategy}")
        elif len(matched_videos) > 1:
            # 需要拼接
            output_video = output_dir / f"scene_{i:03d}_concat.mp4"
            concatenate_videos(matched_videos, output_video)
            print(f"  ✓ 拼接: {strategy}")
        else:
            # 直接使用
            output_video = output_dir / f"scene_{i:03d}.mp4"
            shutil.copy2(matched_videos[0], output_video)
            print(f"  ✓ 使用: {strategy}")
        
        matched_scenes.append({
            "index": i,
            "text": text,
            "target_duration": duration,
            "actual_duration": actual_duration,
            "video_path": str(output_video),
            "strategy": strategy
        })
    
    # 保存匹配结果
    result_file = output_dir / "matched_scenes.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "matched_scenes": matched_scenes,
            "total_scenes": len(matched_scenes)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 匹配完成: {len(matched_scenes)} 个场景")
    print(f"  结果保存在: {result_file}")

def main():
    parser = argparse.ArgumentParser(description='根据narration时长匹配场景视频')
    parser.add_argument('--narration', '-n', required=True,
                       help='narration JSON文件（包含text和duration）')
    parser.add_argument('--search-results-dir', '-s', required=True,
                       help='检索结果目录（每个narration段落对应一个检索结果JSON）')
    parser.add_argument('--video-base-dir', '-v', default='processed',
                       help='视频基础目录（默认: processed/）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录')
    parser.add_argument('--top-k', type=int, default=5,
                       help='每个narration使用的top-k检索结果（默认: 5）')
    parser.add_argument('--tolerance', type=float, default=0.5,
                       help='时长容差（秒，默认: 0.5）')
    
    args = parser.parse_args()
    
    narration_file = Path(args.narration)
    if not narration_file.exists():
        print(f"错误: narration文件不存在: {narration_file}")
        return 1
    
    search_base_dir = Path(args.search_results_dir)
    if not search_base_dir.exists():
        print(f"错误: 检索结果目录不存在: {search_base_dir}")
        return 1
    
    video_base_dir = Path(args.video_base_dir)
    
    process_narration_scenes(
        narration_file,
        search_base_dir,
        video_base_dir,
        args.output,
        args.top_k,
        args.tolerance
    )
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

