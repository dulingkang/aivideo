#!/usr/bin/env python3
"""
根据检索结果提取视频场景片段
可以从检索到的场景中提取对应的视频文件
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import shutil

def load_search_results(search_results_json: Path) -> List[Dict]:
    """加载检索结果"""
    with open(search_results_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('results', [])

def find_scene_video(episode_id: str, scene_id: str, base_dir: Path) -> Path:
    """
    查找场景对应的视频文件
    
    场景ID格式可能是：
    - scene_001
    - 142_scene_001
    """
    # 提取纯场景ID（去除集号前缀）
    if '_' in scene_id and scene_id.split('_')[0].isdigit():
        # 格式：142_scene_001
        _, pure_scene_id = scene_id.split('_', 1)
    else:
        # 格式：scene_001
        pure_scene_id = scene_id
    
    # 查找视频文件
    episode_dir = base_dir / f"episode_{episode_id}" / "scenes"
    
    # 可能的文件名格式
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

def extract_scenes_from_search(
    search_results_json: Path,
    base_dir: Path,
    output_dir: Path,
    copy_files: bool = True,
    max_scenes: int = None
) -> List[Path]:
    """
    从检索结果中提取视频场景
    
    Args:
        search_results_json: 检索结果JSON文件
        base_dir: 视频处理基础目录（processed/）
        output_dir: 输出目录
        copy_files: 是否复制文件（False则创建软链接）
        max_scenes: 最大提取场景数（None表示全部）
    """
    results = load_search_results(search_results_json)
    
    if max_scenes:
        results = results[:max_scenes]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_files = []
    
    print(f"从检索结果提取 {len(results)} 个场景...")
    print(f"输出目录: {output_dir}")
    
    for i, result in enumerate(results, 1):
        scene_data = result.get('scene_data', {})
        scene_id = result.get('scene_id', 'unknown')
        score = result.get('score', 0)
        episode_id = scene_data.get('episode_id', 'unknown')
        
        # 查找视频文件
        video_path = find_scene_video(episode_id, scene_id, base_dir)
        
        if not video_path or not video_path.exists():
            print(f"  [{i}/{len(results)}] ✗ 未找到: {scene_id} (episode {episode_id})")
            continue
        
        # 生成输出文件名
        output_filename = f"{scene_id}_score_{score:.4f}.mp4"
        output_path = output_dir / output_filename
        
        # 复制或链接文件
        if copy_files:
            shutil.copy2(video_path, output_path)
            print(f"  [{i}/{len(results)}] ✓ 复制: {scene_id} -> {output_filename}")
        else:
            try:
                output_path.symlink_to(video_path.absolute())
                print(f"  [{i}/{len(results)}] ✓ 链接: {scene_id} -> {output_filename}")
            except Exception as e:
                print(f"  [{i}/{len(results)}] ✗ 链接失败: {e}")
                continue
        
        extracted_files.append(output_path)
    
    print(f"\n✓ 提取完成: {len(extracted_files)} 个场景")
    return extracted_files

def create_playlist(extracted_files: List[Path], output_playlist: Path):
    """创建播放列表文件（M3U格式）"""
    with open(output_playlist, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")
        for video_file in extracted_files:
            f.write(f"#EXTINF:-1,{video_file.stem}\n")
            f.write(f"{video_file.name}\n")
    print(f"✓ 播放列表已创建: {output_playlist}")

def main():
    parser = argparse.ArgumentParser(description='从检索结果中提取视频场景')
    parser.add_argument('--search-results', '-s', required=True,
                       help='检索结果JSON文件')
    parser.add_argument('--base-dir', '-b', default='processed',
                       help='视频处理基础目录（默认: processed/）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录')
    parser.add_argument('--link', action='store_true',
                       help='创建软链接而不是复制文件（节省空间）')
    parser.add_argument('--max-scenes', type=int,
                       help='最大提取场景数（默认：全部）')
    parser.add_argument('--create-playlist', action='store_true',
                       help='创建M3U播放列表')
    
    args = parser.parse_args()
    
    search_results = Path(args.search_results)
    if not search_results.exists():
        print(f"错误: 检索结果文件不存在: {search_results}")
        return 1
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output)
    
    extracted_files = extract_scenes_from_search(
        search_results,
        base_dir,
        output_dir,
        copy_files=not args.link,
        max_scenes=args.max_scenes
    )
    
    if args.create_playlist and extracted_files:
        playlist_path = output_dir / "playlist.m3u"
        create_playlist(extracted_files, playlist_path)
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

