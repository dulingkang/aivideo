#!/usr/bin/env python3
"""
根据 renjie/episode_*.json 文件生成完整视频
工作流程：
1. 读取JSON文件，解析所有场景
2. 对于每个场景，检索匹配的视频
3. 根据duration匹配视频时长
4. 添加开头和结尾视频
5. 拼接所有视频片段
6. 后续可以添加BGM和旁白
"""

import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))

from search_scenes import load_index, load_scene_metadata, hybrid_search, build_keyword_index
from sentence_transformers import SentenceTransformer
from smart_scene_matcher import decision_make
import faiss

def get_video_duration(video_path: Path) -> float:
    """获取视频时长（秒）"""
    if not video_path.exists():
        return 0.0
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

def trim_video(input_path: Path, output_path: Path, duration: float):
    """裁剪视频到指定时长（静音版本，使用-c copy快速处理）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-t', str(duration),
        '-c', 'copy',  # 直接复制流（快速）
        '-y',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

def concatenate_videos(video_paths: List[Path], output_path: Path):
    """拼接多个视频文件（使用-c copy快速拼接）"""
    if len(video_paths) == 1:
        # 直接复制
        import shutil
        shutil.copy2(video_paths[0], output_path)
        return
    
    # 创建临时文件列表
    concat_file = output_path.parent / f"concat_{output_path.stem}.txt"
    try:
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                if video_path.exists():
                    f.write(f"file '{video_path.absolute()}'\n")
                else:
                    print(f"⚠️  警告: 视频文件不存在: {video_path}")
        
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',  # 直接复制流（快速）
            '-y',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        if concat_file.exists():
            concat_file.unlink()

def find_scene_video(episode_id: str, scene_id: str, base_dir: Path) -> Optional[Path]:
    """查找场景视频文件"""
    # 尝试不同的命名格式
    possible_names = [
        f"episode_{episode_id}_clean-Scene-{scene_id:03d}.mp4",
        f"episode_{episode_id}_clean-Scene-{scene_id}.mp4",
        f"{episode_id}_scene_{scene_id:03d}.mp4",
        f"{episode_id}_scene_{scene_id}.mp4",
    ]
    
    episode_dir = base_dir / f"episode_{episode_id}" / "scenes"
    
    for name in possible_names:
        video_path = episode_dir / name
        if video_path.exists():
            return video_path
    
    # 尝试在scenes目录中查找
    scene_files = list(episode_dir.glob(f"*Scene-{scene_id:03d}*.mp4"))
    if not scene_files:
        scene_files = list(episode_dir.glob(f"*Scene-{scene_id}*.mp4"))
    if scene_files:
        return scene_files[0]
    
    return None

def search_and_match_scene(
    scene_data: Dict,
    index,
    index_metadata,
    scenes,
    keyword_index,
    clip_model,
    video_base_dir: Path,
    scene_id: int
) -> Optional[Path]:
    """检索并匹配场景视频"""
    # 优先使用description，其次使用narration
    query = scene_data.get('description', '') or scene_data.get('narration', '')
    if not query:
        return None
    
    print(f"  检索: {query[:50]}...")
    
    # 执行混合检索
    search_results = hybrid_search(
        query,
        index,
        index_metadata,
        scenes,
        keyword_index,
        clip_model,
        vector_weight=0.7,
        keyword_weight=0.3,
        top_k=10
    )
    
    if not search_results:
        print(f"  ✗ 未找到匹配的场景")
        return None
    
    # 使用智能决策函数
    target_duration = scene_data.get('duration', 0)
    if target_duration <= 0:
        # 如果没有指定duration，使用第一个检索结果的视频
        scene_id_full = search_results[0][0]
        parts = scene_id_full.split('_')
        if len(parts) >= 2:
            episode_id = parts[0]
            scene_num = parts[1].replace('scene_', '')
            video_path = find_scene_video(episode_id, scene_num, video_base_dir)
            if video_path:
                return video_path
    
    # 使用decision_make进行智能匹配
    decision = decision_make(
        search_results=[{
            'scene_id': r[0],
            'score': r[1],
            'scene_data': r[2]
        } for r in search_results],
        target_duration=target_duration,
        base_dir=video_base_dir,
        narration_text=query,
        prefer_retrieved=True
    )
    
    if decision.get('decision') == 'use_retrieved':
        video_path = Path(decision['video_path'])
        if video_path.exists():
            # 如果需要裁剪
            if decision.get('needs_trim') and target_duration > 0:
                output_path = video_base_dir.parent / 'temp' / f"scene_{scene_id}_trimmed.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                trim_video(video_path, output_path, target_duration)
                return output_path
            return video_path
    
    return None

def generate_video_from_json(
    json_file: Path,
    index_path: Path,
    metadata_path: Path,
    scene_metadata_files: List[Path],
    video_base_dir: Path,
    opening_video: Optional[Path],
    ending_video: Optional[Path],
    output_path: Path,
    skip_opening: bool = False,
    skip_ending: bool = False
):
    """根据JSON文件生成完整视频"""
    
    # 加载JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    episode_id = data.get('episode')
    scenes = data.get('scenes', [])
    
    print("=" * 60)
    print(f"生成视频: 第{episode_id}集 - {data.get('title', '')}")
    print("=" * 60)
    print(f"总场景数: {len(scenes)}")
    print()
    
    # 加载索引
    print("加载索引...")
    index = load_index(index_path)
    index_metadata = json.load(open(metadata_path, 'r', encoding='utf-8'))
    all_scenes = {}
    for scene_file in scene_metadata_files:
        scenes_data = json.load(open(scene_file, 'r', encoding='utf-8'))
        all_scenes.update(scenes_data)
    
    keyword_index = build_keyword_index(all_scenes, use_subtitle=False)
    clip_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("✓ 索引加载完成\n")
    
    # 处理每个场景
    video_segments = []
    
    for i, scene in enumerate(scenes):
        scene_id = scene.get('id')
        duration = scene.get('duration', 0)
        narration = scene.get('narration', '')
        
        # 开头场景（id=0）
        if scene_id == 0:
            if skip_opening:
                print(f"[{i+1}/{len(scenes)}] 跳过开头场景")
                continue
            if opening_video and opening_video.exists():
                # 裁剪到指定时长
                if duration > 0:
                    temp_path = video_base_dir.parent / 'temp' / 'opening_trimmed.mp4'
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    trim_video(opening_video, temp_path, duration)
                    video_segments.append(temp_path)
                    print(f"[{i+1}/{len(scenes)}] ✅ 开头视频: {duration}秒")
                else:
                    video_segments.append(opening_video)
                    print(f"[{i+1}/{len(scenes)}] ✅ 开头视频")
            else:
                print(f"[{i+1}/{len(scenes)}] ⚠️  开头视频不存在，跳过")
            continue
        
        # 结尾场景（id=999）
        if scene_id == 999:
            if skip_ending:
                print(f"[{i+1}/{len(scenes)}] 跳过结尾场景")
                continue
            if ending_video and ending_video.exists():
                # 裁剪到指定时长
                if duration > 0:
                    temp_path = video_base_dir.parent / 'temp' / 'ending_trimmed.mp4'
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    trim_video(ending_video, temp_path, duration)
                    video_segments.append(temp_path)
                    print(f"[{i+1}/{len(scenes)}] ✅ 结尾视频: {duration}秒")
                else:
                    video_segments.append(ending_video)
                    print(f"[{i+1}/{len(scenes)}] ✅ 结尾视频")
            else:
                print(f"[{i+1}/{len(scenes)}] ⚠️  结尾视频不存在，跳过")
            continue
        
        # 普通场景：检索并匹配
        print(f"[{i+1}/{len(scenes)}] 场景 {scene_id} (时长: {duration}秒)")
        print(f"  旁白: {narration[:50]}...")
        
        matched_video = search_and_match_scene(
            scene,
            index,
            index_metadata,
            all_scenes,
            keyword_index,
            clip_model,
            video_base_dir,
            scene_id
        )
        
        if matched_video:
            video_segments.append(matched_video)
            actual_duration = get_video_duration(matched_video)
            print(f"  ✅ 匹配成功: {matched_video.name} ({actual_duration:.2f}秒)")
        else:
            print(f"  ✗ 未找到匹配的视频")
        
        print()
    
    # 拼接所有视频片段
    if not video_segments:
        print("❌ 错误: 没有找到任何视频片段")
        return False
    
    print("=" * 60)
    print(f"拼接 {len(video_segments)} 个视频片段...")
    print("=" * 60)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concatenate_videos(video_segments, output_path)
    
    total_duration = get_video_duration(output_path)
    print(f"\n✅ 视频生成完成！")
    print(f"  输出文件: {output_path}")
    print(f"  总时长: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='根据JSON文件生成完整视频')
    parser.add_argument('--json', '-j', required=True,
                       help='JSON文件路径 (renjie/episode_*.json)')
    parser.add_argument('--index', required=True,
                       help='FAISS索引路径')
    parser.add_argument('--metadata', required=True,
                       help='索引metadata路径')
    parser.add_argument('--scenes', '-s', required=True, nargs='+',
                       help='场景metadata JSON文件（可多个）')
    parser.add_argument('--video-dir', required=True,
                       help='视频文件基础目录（processed/）')
    parser.add_argument('--opening', 
                       help='开头视频路径（可选，如果不指定会使用JSON中id=0的场景）')
    parser.add_argument('--ending',
                       help='结尾视频路径（可选，如果不指定会使用JSON中id=999的场景）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出视频路径')
    parser.add_argument('--skip-opening', action='store_true',
                       help='跳过开头视频')
    parser.add_argument('--skip-ending', action='store_true',
                       help='跳过结尾视频')
    
    args = parser.parse_args()
    
    opening_video = Path(args.opening) if args.opening else None
    ending_video = Path(args.ending) if args.ending else None
    
    success = generate_video_from_json(
        Path(args.json),
        Path(args.index),
        Path(args.metadata),
        [Path(f) for f in args.scenes],
        Path(args.video_dir),
        opening_video,
        ending_video,
        Path(args.output),
        skip_opening=args.skip_opening,
        skip_ending=args.skip_ending
    )
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

