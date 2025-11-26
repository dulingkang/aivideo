#!/usr/bin/env python3
"""
根据 renjie/episode_*.json 文件生成完整视频（完整工作流）
工作流程：
1. 读取JSON文件，提取narration文本
2. 使用TTS生成配音（先读配音，才能确定每个场景对应的时长）
3. 获取实际音频时长
4. 根据音频时长和场景描述检索原视频，或AI生成视频
5. 添加开头和结尾视频
6. 拼接所有视频片段
7. 后续添加BGM
"""

import json
import argparse
import subprocess
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加gen_video路径以使用TTS和AI生成功能
sys.path.insert(0, str(Path(__file__).parent / "gen_video"))
# 添加tools/video_processing路径以使用视频检索功能
sys.path.insert(0, str(Path(__file__).parent / "tools" / "video_processing"))

from search_scenes import load_index, load_scene_metadata, hybrid_search, build_keyword_index
from sentence_transformers import SentenceTransformer
from smart_scene_matcher import decision_make
import faiss

# 导入gen_video的模块
try:
    from tts_generator import TTSGenerator
    from image_generator import ImageGenerator
    from video_generator import VideoGenerator
    from video_composer import VideoComposer
    from subtitle_generator import SubtitleGenerator
    GEN_VIDEO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  警告: 无法导入gen_video模块: {e}")
    print("  将只支持原视频检索，不支持AI生成和BGM")
    GEN_VIDEO_AVAILABLE = False

def get_media_duration(media_path: Path) -> float:
    """获取视频或音频时长（秒）"""
    if not media_path.exists():
        return 0.0
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(media_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

def trim_video(input_path: Path, output_path: Path, duration: float):
    """裁剪视频到指定时长（重新编码保证时间戳精准）"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-y',
        '-i', str(input_path),
        '-t', f"{duration:.3f}",
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '20',
        '-c:a', 'aac',
        '-ar', '48000',
        '-ac', '2',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

def concatenate_videos(video_paths: List[Path], output_path: Path):
    """拼接多个视频文件（使用-c copy快速拼接）"""
    if len(video_paths) == 1:
        import shutil
        shutil.copy2(video_paths[0], output_path)
        return
    
    concat_file = output_path.parent / f"concat_{output_path.stem}.txt"
    try:
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                if video_path.exists():
                    f.write(f"file '{video_path.absolute()}'\n")
        
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

def assemble_scene_videos(
    matched_videos: List[Path],
    target_duration: float,
    temp_root: Path,
    label: str,
    video_base_dir: Path,
    primary_scene_key: Optional[str] = None,
    max_adjacent: int = 3
) -> Tuple[Optional[Path], float]:
    """
    将多个原视频片段裁剪/拼接，使其总时长精确匹配 target_duration。
    返回 (assembled_path, remaining_shortfall)
    """
    tolerance = 0.05  # 50ms 容差
    attempt_videos = [Path(v) for v in matched_videos]
    attempt = 0
    
    while True:
        assembled_path, remaining = _assemble_once(
            attempt_videos,
            target_duration,
            temp_root,
            f"{label}_try{attempt}",
            tolerance
        )
        if assembled_path or remaining <= tolerance:
            return assembled_path, max(remaining, 0.0)
        
        if not primary_scene_key or not video_base_dir:
            return None, remaining
        
        extras = find_adjacent_scene_videos(
            primary_scene_key,
            video_base_dir,
            max_extra=max_adjacent,
            exclude_paths=attempt_videos
        )
        if not extras:
            print("    ⚠️ 无法找到相邻片段，放弃原视频拼接")
            return None, remaining
        
        print(f"    ℹ 时长不足，追加相邻片段 {len(extras)} 个，重新尝试拼接")
        attempt_videos.extend(extras)
        attempt += 1

def _assemble_once(
    video_list: List[Path],
    target_duration: float,
    temp_root: Path,
    label: str,
    tolerance: float
) -> Tuple[Optional[Path], float]:
    temp_root.mkdir(parents=True, exist_ok=True)
    remaining = max(target_duration, 0.0)
    assembled_segments: List[Path] = []
    
    for idx, video in enumerate(video_list):
        src_path = Path(video)
        if not src_path.exists():
            print(f"    ⚠️ 匹配视频不存在，跳过: {src_path}")
            continue
        
        duration = get_media_duration(src_path)
        if duration <= 0:
            print(f"    ⚠️ 无法获取视频时长，跳过: {src_path}")
            continue
        
        # 保留三位小数，避免累计误差
        clip_duration = round(min(duration, remaining), 3)
        if clip_duration <= 0:
            break
        
        segment_path = temp_root / f"{label}_seg_{idx:02d}.mp4"
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        trim_video(src_path, segment_path, clip_duration)
        assembled_segments.append(segment_path)
        remaining -= clip_duration
        
        if remaining <= tolerance:
            remaining = 0.0
            break
    
    if not assembled_segments:
        return None, target_duration
    
    if remaining > tolerance:
        for seg in assembled_segments:
            if seg.exists():
                seg.unlink(missing_ok=True)
        return None, remaining
    
    if len(assembled_segments) == 1:
        return assembled_segments[0], remaining
    
    final_path = temp_root / f"{label}_assembled.mp4"
    concatenate_videos(assembled_segments, final_path)
    return (final_path if final_path.exists() else None), remaining
def find_scene_video(episode_id: str, scene_id: str, base_dir: Path) -> Optional[Path]:
    """查找场景视频文件"""
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

def run_upscale_only(
    video_path: Path,
    output_path: Optional[Path],
    config_path: Optional[Path]
) -> bool:
    """
    仅执行 Real-ESRGAN 超分处理（跳过完整生成流程）
    """
    if not video_path.exists():
        print(f"❌ 错误: 输入视频不存在: {video_path}")
        return False
    
    if not config_path or not config_path.exists():
        config_path = Path(__file__).parent.parent.parent / "gen_video" / "config.yaml"
        if not config_path.exists():
            print("❌ 错误: 未找到 gen_video/config.yaml，无法执行超分")
            return False
    
    try:
        from video_composer import VideoComposer
    except ImportError as exc:
        print(f"❌ 错误: 无法导入 VideoComposer: {exc}")
        return False
    
    composer = VideoComposer(str(config_path))
    post_cfg = composer.composition_config.get("postprocess", {})
    if not post_cfg.get("enabled", False):
        print("⚠ 提示: 配置中 postprocess.enabled = false，已临时启用以执行超分")
        post_cfg = dict(post_cfg)
        post_cfg["enabled"] = True
    
    try:
        upscaled_path = composer.postprocess_with_realesrgan(str(video_path), post_cfg)
    except Exception as exc:
        print(f"❌ Real-ESRGAN 超分失败: {exc}")
        return False
    
    final_path = Path(upscaled_path)
    if output_path:
        try:
            shutil.move(str(final_path), str(output_path))
            final_path = output_path
        except Exception as exc:
            print(f"⚠ 超分结果移动到 {output_path} 失败: {exc}")
    
    print(f"✅ 超分完成: {final_path}")
    return True

def parse_scene_key(scene_key: Optional[str]) -> Optional[Tuple[str, int]]:
    if not scene_key or '_scene_' not in scene_key.lower():
        return None
    parts = scene_key.lower().split('_scene_', 1)
    if len(parts) != 2:
        return None
    episode_part, scene_part = parts
    try:
        return episode_part.strip(), int(scene_part)
    except ValueError:
        return None

def find_adjacent_scene_videos(
    scene_key: Optional[str],
    base_dir: Path,
    max_extra: int = 3,
    exclude_paths: Optional[List[Path]] = None
) -> List[Path]:
    parsed = parse_scene_key(scene_key)
    if not parsed:
        return []
    episode_id, start_scene = parsed
    exclude_set = {Path(p).resolve() for p in (exclude_paths or [])}
    extras: List[Path] = []
    
    for offset in range(1, max_extra + 1):
        next_scene_num = start_scene + offset
        video_path = find_scene_video(episode_id, next_scene_num, base_dir)
        if video_path and video_path.exists():
            resolved = video_path.resolve()
            if resolved not in exclude_set:
                extras.append(video_path)
                exclude_set.add(resolved)
            else:
                continue
        else:
            break
    return extras

def try_alternative_candidates(
    search_results: List[Tuple[str, float, Dict]],
    audio_duration: float,
    temp_root: Path,
    label: str,
    video_base_dir: Path,
    start_index: int = 1,
    max_adjacent: int = 3
) -> Tuple[Optional[Path], Optional[str]]:
    """
    当首个检索结果无法满足时，尝试使用后续候选及其相邻片段
    """
    for candidate in search_results[start_index:]:
        scene_key = candidate[0]
        parsed = parse_scene_key(scene_key)
        if not parsed:
            continue
        episode_id, scene_num = parsed
        base_video = find_scene_video(episode_id, scene_num, video_base_dir)
        if not base_video or not base_video.exists():
            continue
        
        base_list = [base_video]
        base_list.extend(
            find_adjacent_scene_videos(
                scene_key,
                video_base_dir,
                max_extra=max_adjacent,
                exclude_paths=base_list
            )
        )
        
        assembled_path, remaining = assemble_scene_videos(
            base_list,
            audio_duration,
            temp_root,
            f"{label}_alt",
            video_base_dir,
            primary_scene_key=scene_key,
            max_adjacent=max_adjacent
        )
        if assembled_path:
            print(f"    ✅ 使用候选 {scene_key} 及相邻片段完成拼接: {assembled_path.name}")
            return assembled_path, scene_key
        elif remaining <= 0.05:
            print(f"    ⚠️ 候选 {scene_key} 拼接失败（长度已足够但合成失败），继续尝试下一个")
        else:
            print(f"    ⚠️ 候选 {scene_key} 时长仍不足，继续尝试下一个")
    
    return None

def generate_audio_for_scenes(
    scenes: List[Dict],
    output_dir: Path,
    tts_generator: TTSGenerator,
    skip_existing: bool = True
) -> Tuple[List[Path], List[float]]:
    """
    为每个场景生成配音音频
    
    Returns:
        (audio_paths, audio_durations) - 音频路径列表和时长列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = []
    audio_durations = []
    
    print("=" * 60)
    print("生成配音音频（使用声音克隆）")
    print("=" * 60)
    print()
    
    for i, scene in enumerate(scenes):
        narration = scene.get("narration", "")
        scene_id = scene.get("id", i)
        
        if not narration:
            print(f"[{i+1}/{len(scenes)}] 场景 {scene_id}: 无旁白，跳过")
            continue
        
        # 确定输出文件名
        if scene_id == 0:
            audio_file = output_dir / "audio_opening.wav"
        elif scene_id == 999:
            audio_file = output_dir / "audio_ending.wav"
        else:
            audio_file = output_dir / f"audio_scene_{scene_id:03d}.wav"
        
        # 检查是否已存在
        if skip_existing and audio_file.exists():
            duration = get_media_duration(audio_file)
            audio_paths.append(audio_file)
            audio_durations.append(duration)
            scene_label = f"场景 {scene_id}" if scene_id not in [0, 999] else ("开头" if scene_id == 0 else "结尾")
            print(f"[{i+1}/{len(scenes)}] {scene_label}: 音频已存在 ({duration:.2f}秒) - {narration[:30]}...")
            continue
        
        # 生成音频
        try:
            scene_label = f"场景 {scene_id}" if scene_id not in [0, 999] else ("开头" if scene_id == 0 else "结尾")
            print(f"[{i+1}/{len(scenes)}] {scene_label}: 生成配音 - {narration[:30]}...")
            tts_generator.generate(narration, str(audio_file))
            
            duration = get_media_duration(audio_file)
            audio_paths.append(audio_file)
            audio_durations.append(duration)
            print(f"  ✓ 生成完成: {duration:.2f}秒")
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            continue
    
    print()
    print(f"✓ 共生成 {len(audio_paths)} 个音频文件")
    print(f"  总时长: {sum(audio_durations):.2f}秒 ({sum(audio_durations)/60:.2f}分钟)")
    print()
    
    return audio_paths, audio_durations

def search_or_generate_video(
    scene: Dict,
    audio_duration: float,
    index,
    index_metadata,
    all_scenes,
    keyword_index,
    clip_model,
    video_base_dir: Path,
    scene_index: int,
    tts_generator: Optional[TTSGenerator] = None,
    image_generator: Optional[ImageGenerator] = None,
    video_generator: Optional[VideoGenerator] = None,
    ai_output_dir: Optional[Path] = None,
    used_scenes: Optional[dict] = None,
    max_reuse_count: int = 1
) -> Tuple[Optional[Path], Optional[str]]:
    """
    检索原视频或AI生成视频
    
    策略：
    1. 先尝试检索原视频（会过滤掉已达到最大使用次数的场景）
    2. 如果检索不到或分数太低，使用AI生成
    
    去重机制：
    - used_scenes: 字典，记录每个场景的使用次数 {scene_key: count}
    - max_reuse_count: 最多允许重复使用次数（默认2次）
    - 返回: (视频路径, 场景key) 元组，场景key用于记录使用次数
    
    Returns:
        Tuple[Optional[Path], Optional[str]]: (视频路径, 场景key)，场景key为None表示AI生成
    """
    description = scene.get('description', '') or scene.get('narration', '')
    scene_id = scene.get('id', scene_index)
    
    print(f"  场景 {scene_id} (目标时长: {audio_duration:.2f}秒):")
    print(f"    描述: {description[:50]}...")
    
    # 1. 尝试检索原视频
    if description:
        search_results = hybrid_search(
            description,
            index,
            index_metadata,
            all_scenes,
            keyword_index,
            clip_model,
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=30  # 增加检索数量，提高匹配概率
        )
        
        # 过滤掉已达到最大使用次数的场景
        if used_scenes and search_results:
            original_count = len(search_results)
            search_results = [
                (scene_key, score, scene_meta)
                for scene_key, score, scene_meta in search_results
                if used_scenes.get(scene_key, 0) < max_reuse_count
            ]
            filtered_count = len(search_results)
            if filtered_count < original_count:
                excluded_count = original_count - filtered_count
                print(f"    🔍 过滤掉 {excluded_count} 个已达到最大使用次数（{max_reuse_count}次）的场景")
        
        primary_scene_key = search_results[0][0] if search_results else None
        
        if search_results:
            print("    🔍 检索结果（前5个，已排除达到最大使用次数的场景）:")
            for rank, (scene_key, score, scene_meta) in enumerate(search_results[:5], 1):
                episode = scene_meta.get('episode_id')
                caption = (scene_meta.get('caption') or scene_meta.get('visual_caption') or scene_meta.get('text') or "")[:40]
                use_count = used_scenes.get(scene_key, 0) if used_scenes else 0
                used_mark = f" [已使用{use_count}次]" if use_count > 0 else ""
                print(f"      #{rank}: {scene_key} | score={score:.3f} | episode={episode} | desc={caption}{used_mark}")
        
            # 使用智能决策（大幅降低标准，优先使用任何可用的原视频）
            decision = decision_make(
                search_results=[{
                    'scene_id': r[0],
                    'score': r[1],
                    'scene_data': r[2]
                } for r in search_results],
                target_duration=audio_duration,
                base_dir=video_base_dir,
                narration_text=description,
                score_threshold_high=0.2,   # 大幅降低阈值（0.2）
                score_threshold_low=0.05,   # 极低阈值（0.05）
                duration_tolerance=1.0,     # 放宽时长差异（±100%，几乎不限制）
                avoid_ai_for_characters=False,
                prefer_retrieved=True
            )
            
            print(f"    🛈 决策: {decision.get('decision')} | 原因: {decision.get('reason')}")
            if decision.get('decision') in ('use_retrieved', 'retrieved'):
                matched_videos = decision.get('matched_videos', [])
                if matched_videos:
                    temp_root = (ai_output_dir if ai_output_dir else (video_base_dir.parent / 'temp')) / "retrieved_segments"
                    assembled_path, remaining_shortfall = assemble_scene_videos(
                        matched_videos,
                        audio_duration,
                        temp_root,
                        f"scene_{scene_id}",
                        video_base_dir,
                        primary_scene_key=primary_scene_key,
                        max_adjacent=5
                    )
                    if assembled_path:
                        print(f"    ✅ 原视频拼接完成，满足音频时长: {assembled_path.name}")
                        return assembled_path, primary_scene_key
                    else:
                        if remaining_shortfall > 0.05:
                            print(f"    ⚠️ 匹配到的原视频总时长仍短 {remaining_shortfall:.2f}s，尝试使用其他候选")
                        else:
                            print("    ⚠️ 原视频拼接失败，尝试使用其他候选")
                        alt_path, alt_scene_key = try_alternative_candidates(
                            search_results,
                            audio_duration,
                            temp_root,
                            f"scene_{scene_id}",
                            video_base_dir,
                            start_index=1,
                            max_adjacent=5
                        )
                        if alt_path:
                            return alt_path, alt_scene_key or primary_scene_key
                        else:
                            print("    ⚠️ 所有候选都无法满足时长，将回退到AI生成")
                print("    ⚠️ 决策指向检索，但 matched_videos 为空或文件不存在")
    
    # 2. 检索不到或分数太低，使用AI生成
    if not GEN_VIDEO_AVAILABLE:
        print(f"    ⚠️  无法匹配原视频，且AI生成功能不可用")
        return None, None
    
    if not image_generator or not video_generator:
        print(f"    ⚠️  无法匹配原视频，但AI生成器未加载")
        return None, None
    
    print(f"    🎨 未找到匹配的原视频，使用AI生成...")
    
    # 懒加载图像/视频生成模型，避免不必要的加载
    try:
        if image_generator and getattr(image_generator, "pipeline", None) is None:
            print("      ↻ 加载图像生成模型...")
            image_generator.load_pipeline()
    except Exception as e:
        print(f"      ✗ 图像生成模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    try:
        if video_generator and not getattr(video_generator, "model_loaded", False):
            print("      ↻ 加载视频生成模型...")
            video_generator.load_model()
    except Exception as e:
        print(f"      ✗ 视频生成模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    try:
        # 生成图片
        prompt = scene.get('prompt', '') or scene.get('description', '')
        if not prompt:
            print(f"    ✗ 缺少prompt或description，无法生成")
            return None, None
        
        image_output_dir = ai_output_dir / "images"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_output_dir / f"scene_{scene_id:03d}.png"
        
        print(f"      1/2 生成图片...")
        image_generator.generate_image(
            prompt=prompt,
            output_path=image_path,
            scene=scene
        )
        
        if not image_path.exists():
            print(f"      ✗ 图片生成失败")
            return None, None
        
        # 生成视频
        video_output_dir = ai_output_dir / "videos"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_output_dir / f"scene_{scene_id:03d}.mp4"
        
        print(f"      2/2 生成视频 (时长: {audio_duration:.2f}秒)...")
        # 设置duration以便生成匹配时长的视频
        scene_with_duration = scene.copy()
        scene_with_duration['duration'] = audio_duration
        
        video_generator.generate_video(
            image_path=str(image_path),
            output_path=str(video_path),
            scene=scene_with_duration
        )
        
        if video_path.exists():
            print(f"      ✅ AI生成完成: {video_path.name}")
            return video_path, None  # AI生成的视频没有对应的场景key
        else:
            print(f"      ✗ 视频生成失败")
            return None, None
            
    except Exception as e:
        print(f"    ✗ AI生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_video_from_json_complete(
    json_file: Path,
    index_path: Path,
    metadata_path: Path,
    scene_metadata_files: List[Path],
    video_base_dir: Path,
    opening_video: Optional[Path],
    ending_video: Optional[Path],
    output_path: Path,
    gen_video_config: Optional[Path] = None,
    skip_opening: bool = False,
    skip_ending: bool = False,
    skip_tts: bool = False
):
    """根据JSON文件生成完整视频（完整工作流）"""
    
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
    
    # 创建输出目录
    output_dir = output_path.parent / f"episode_{episode_id}_work"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audios"
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载TTS生成器和视频合成器（如果需要）
    tts_generator = None
    image_generator = None
    video_generator = None
    video_composer = None
    subtitle_generator = None
    
    if GEN_VIDEO_AVAILABLE:
        if gen_video_config and gen_video_config.exists():
            config_path = gen_video_config
        else:
            # 尝试查找默认配置
            config_path = Path(__file__).parent.parent.parent / "gen_video" / "config.yaml"
            if not config_path.exists():
                print("⚠️  警告: 未找到gen_video配置，将只支持原视频检索")
        
        if config_path.exists():
            print("加载AI生成和视频合成模块...")
            try:
                if not skip_tts:
                    tts_generator = TTSGenerator(str(config_path))
                image_generator = ImageGenerator(str(config_path))
                video_generator = VideoGenerator(str(config_path))
                video_composer = VideoComposer(str(config_path))
                subtitle_generator = SubtitleGenerator(str(config_path))
                print("✓ AI生成和视频合成模块加载成功\n")
            except Exception as e:
                print(f"⚠️  警告: AI生成模块加载失败: {e}")
                print("  将只支持原视频检索\n")
                import traceback
                traceback.print_exc()
    
    # 步骤1: 生成配音音频
    audio_paths = []
    audio_durations = []
    
    if tts_generator and not skip_tts:
        audio_paths, audio_durations = generate_audio_for_scenes(
            scenes, audio_dir, tts_generator, skip_existing=True
        )
    else:
        print("⚠️  跳过配音生成（TTS未加载或已跳过）")
        print("  将使用JSON中的duration字段\n")
        # 使用JSON中的duration
        for scene in scenes:
            duration = scene.get('duration', 0)
            if duration > 0:
                audio_durations.append(duration)
    
    # 加载索引（用于视频检索）
    print("加载场景索引...")
    index, index_metadata = load_index(index_path, metadata_path)
    # 使用与 search_scenes 相同的辅助函数，以确保每个场景都带有 episode_id/scene_id 信息
    all_scenes = load_scene_metadata(scene_metadata_files)
    
    # 为了尽可能提高命中率，这里允许使用字幕文本参与关键词索引（与 CLI 行为保持一致）
    keyword_index = build_keyword_index(all_scenes, use_subtitle=True)
    clip_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✓ 索引加载完成\n")
    
    # 步骤2: 为每个场景匹配或生成视频
    video_segments = []
    audio_idx = 0
    used_scenes = {}  # 记录已使用场景的次数 {scene_key: count}，最多允许使用2次
    max_reuse_count = 1  # 最多允许重复使用2次
    
    for i, scene in enumerate(scenes):
        scene_id = scene.get('id', i)
        narration = scene.get('narration', '')
        
        # 获取对应的音频时长
        if audio_idx < len(audio_durations):
            target_duration = audio_durations[audio_idx]
            audio_idx += 1
        else:
            target_duration = scene.get('duration', 0)
            if target_duration <= 0:
                print(f"[{i+1}/{len(scenes)}] 场景 {scene_id}: ⚠️  无法确定时长，跳过")
                continue
        
        # 开头场景（id=0）
        if scene_id == 0:
            if skip_opening:
                print(f"[{i+1}/{len(scenes)}] 跳过开头场景")
                continue
            if opening_video and opening_video.exists():
                temp_path = temp_dir / 'opening_trimmed.mp4'
                trim_video(opening_video, temp_path, target_duration)
                video_segments.append(temp_path)
                print(f"[{i+1}/{len(scenes)}] ✅ 开头视频: {target_duration:.2f}秒")
            else:
                print(f"[{i+1}/{len(scenes)}] ⚠️  开头视频不存在，跳过")
            continue
        
        # 结尾场景（id=999）
        if scene_id == 999:
            if skip_ending:
                print(f"[{i+1}/{len(scenes)}] 跳过结尾场景")
                continue
            if ending_video and ending_video.exists():
                temp_path = temp_dir / 'ending_trimmed.mp4'
                trim_video(ending_video, temp_path, target_duration)
                video_segments.append(temp_path)
                print(f"[{i+1}/{len(scenes)}] ✅ 结尾视频: {target_duration:.2f}秒")
            else:
                print(f"[{i+1}/{len(scenes)}] ⚠️  结尾视频不存在，跳过")
            continue
        
        # 普通场景：检索或生成视频
        print(f"[{i+1}/{len(scenes)}] 场景 {scene_id}")
        matched_video, used_scene_key = search_or_generate_video(
            scene,
            target_duration,
            index,
            index_metadata,
            all_scenes,
            keyword_index,
            clip_model,
            video_base_dir,
            scene_id,
            tts_generator=tts_generator,
            image_generator=image_generator,
            video_generator=video_generator,
            ai_output_dir=output_dir,
            used_scenes=used_scenes,
            max_reuse_count=max_reuse_count
        )
        
        if matched_video:
            video_segments.append(matched_video)
            # 记录已使用的场景（计数）
            if used_scene_key:
                used_scenes[used_scene_key] = used_scenes.get(used_scene_key, 0) + 1
                count = used_scenes[used_scene_key]
                status = f"第{count}次使用" if count > 1 else "首次使用"
                print(f"    📝 已记录使用场景: {used_scene_key}（{status}，最多允许{max_reuse_count}次）")
        else:
            print(f"    ⚠️  未找到/生成视频，跳过此场景")
        
        print()
    
    # 步骤3: 拼接所有视频片段（静音版本）
    if not video_segments:
        print("❌ 错误: 没有找到任何视频片段")
        return False
    
    print("=" * 60)
    print(f"拼接 {len(video_segments)} 个视频片段（静音）...")
    print("=" * 60)
    
    # 先拼接静音视频
    temp_video_silent = temp_dir / "video_silent.mp4"
    concatenate_videos(video_segments, temp_video_silent)
    
    # 步骤4: 添加配音和BGM（如果可用）
    if video_composer and audio_paths:
        print("=" * 60)
        print("添加配音和BGM...")
        print("=" * 60)
        
        # 合并所有配音音频
        merged_audio_path = None
        if len(audio_paths) > 1:
            print("合并配音音频...")
            merged_audio_path = temp_dir / "merged_audio.wav"
            concat_list = temp_dir / "audio_concat.txt"
            try:
                with open(concat_list, 'w', encoding='utf-8') as f:
                    for audio_path in audio_paths:
                        if audio_path.exists():
                            f.write(f"file '{audio_path.absolute()}'\n")
                
                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0',
                    '-i', str(concat_list),
                    '-acodec', 'pcm_s16le',
                    '-ac', '2',  # 立体声
                    '-ar', '48000',  # 采样率
                    '-y',
                    str(merged_audio_path)
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✓ 配音音频已合并: {merged_audio_path}")
            except Exception as e:
                print(f"⚠️  音频合并失败: {e}")
                merged_audio_path = audio_paths[0] if audio_paths else None
        elif len(audio_paths) == 1:
            merged_audio_path = audio_paths[0]
        
        if merged_audio_path and merged_audio_path.exists():
            # 生成字幕（使用narration文本替换识别结果）
            subtitle_path = None
            if subtitle_generator:
                print("=" * 60)
                print("生成字幕（使用narration文本替换识别结果）...")
                print("=" * 60)
                
                try:
                    # 收集所有场景的narration文本（用于字幕分段和替换）
                    # 注意：顺序必须与audio_paths和audio_durations完全一致
                    # 按照scenes的顺序遍历，只收集有narration的场景（与generate_audio_for_scenes逻辑一致）
                    scene_texts = []  # 分段文本列表
                    narration_text = ""  # 完整旁白文本
                    
                    for scene in scenes:
                        narration = scene.get("narration", "")
                        if narration:  # 只收集有narration的场景（与音频生成逻辑一致）
                            scene_texts.append(narration)
                            narration_text += narration
                    
                    # 验证数量是否匹配
                    if scene_texts and len(audio_durations) == len(scene_texts):
                        subtitle_path = temp_dir / "subtitle.srt"
                        total_duration = sum(audio_durations)
                        
                        print(f"  完整旁白文本: {len(narration_text)} 字")
                        print(f"  分段数: {len(scene_texts)} 个")
                        print(f"  总时长: {total_duration:.2f}秒")
                        
                        subtitle_generator.generate(
                            str(merged_audio_path),
                            str(subtitle_path),
                            narration=narration_text,  # 完整旁白文本（用于替换识别结果）
                            segments=scene_texts,  # 分段文本（用于字幕分段）
                            video_durations=audio_durations,  # 音频时长列表（确保时间轴对齐）
                            total_duration=total_duration,  # 总音频时长
                        )
                        
                        if subtitle_path.exists():
                            print(f"✓ 字幕已生成: {subtitle_path}")
                        else:
                            print(f"⚠️  字幕文件未生成")
                            subtitle_path = None
                    else:
                        print(f"⚠️  场景文本数量 ({len(scene_texts)}) 与音频数量 ({len(audio_durations)}) 不匹配，跳过字幕生成")
                except Exception as e:
                    print(f"⚠️  字幕生成失败: {e}")
                    import traceback
                    traceback.print_exc()
                    subtitle_path = None
            
            try:
                # 使用VideoComposer添加配音、BGM和字幕
                print("=" * 60)
                print("使用VideoComposer合成最终视频（配音+BGM+字幕）...")
                print("=" * 60)
                
                # 将拼接好的视频作为单个视频片段传入
                composed_path = video_composer.compose(
                    video_paths=[str(p) for p in video_segments],
                    audio_path=str(merged_audio_path),
                    subtitle_path=str(subtitle_path) if subtitle_path and subtitle_path.exists() else None,
                    bgm_path=None,  # 使用配置中的BGM
                    output_path=str(output_path),
                    scene_metadata=scenes,  # 传递场景元数据用于BGM选择
                )
                output_path = Path(composed_path)
                print(f"✓ 最终视频已生成（包含配音、BGM和字幕）: {output_path}")

                # 可选：执行 Real-ESRGAN 超分（需要配置可用模型）
                realesrgan_cfg = video_composer.composition_config.get("postprocess", {})
                if realesrgan_cfg.get("enabled"):
                    try:
                        print("准备执行 Real-ESRGAN 超分处理...")
                        # VideoComposer.compose 已在内部处理超分逻辑，但确保路径刷新
                        output_path = Path(composed_path)
                    except Exception as re_err:
                        print(f"⚠ Real-ESRGAN 超分失败: {re_err}")
            except Exception as e:
                print(f"⚠️  使用VideoComposer合成失败: {e}")
                print("  回退到仅拼接视频（无配音和BGM）")
                import shutil
                shutil.copy2(temp_video_silent, output_path)
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  配音音频不存在，仅拼接视频（无配音和BGM）")
            import shutil
            shutil.copy2(temp_video_silent, output_path)
    else:
        # 没有VideoComposer或配音，直接使用拼接的视频
        import shutil
        shutil.copy2(temp_video_silent, output_path)
        if not video_composer:
            print("⚠️  VideoComposer未加载，无法添加BGM和配音")
        if not audio_paths:
            print("⚠️  无配音音频，无法添加配音")
    
    total_duration = get_media_duration(output_path)
    print(f"\n✅ 视频生成完成！")
    print(f"  输出文件: {output_path}")
    print(f"  总时长: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    if audio_paths:
        print(f"  音频文件: {len(audio_paths)} 个（保存在 {audio_dir}）")
    print()
    
    if video_composer and audio_paths:
        print("✅ 已包含：")
        print("  ✓ 配音音频（TTS生成）")
        print("  ✓ 背景音乐（BGM，已智能选择和均衡）")
        if subtitle_generator:
            print("  ✓ 字幕（使用narration文本替换识别结果）")
        print()
        print("💡 视频已完整生成，可直接使用")
    else:
        print("💡 下一步：")
        if not video_composer:
            print("  1. 添加BGM")
        if not audio_paths:
            print("  2. 将配音音频添加到视频中")
        if not subtitle_generator:
            print("  3. 生成字幕")
    print()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='根据JSON文件生成完整视频（完整工作流）')
    parser.add_argument('--json', '-j', required=False,
                       help='JSON文件路径 (renjie/episode_*.json)')
    parser.add_argument('--index', required=False,
                       help='FAISS索引路径')
    parser.add_argument('--metadata', required=False,
                       help='索引metadata路径')
    parser.add_argument('--scenes', '-s', required=False, nargs='+',
                       help='场景metadata JSON文件（可多个）')
    parser.add_argument('--video-dir', required=False,
                       help='视频文件基础目录（processed/）')
    parser.add_argument('--opening', 
                       help='开头视频路径（可选）')
    parser.add_argument('--ending',
                       help='结尾视频路径（可选）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出视频路径')
    parser.add_argument('--gen-video-config',
                       help='gen_video配置文件路径（默认: gen_video/config.yaml）')
    parser.add_argument('--skip-opening', action='store_true',
                       help='跳过开头视频')
    parser.add_argument('--skip-ending', action='store_true',
                       help='跳过结尾视频')
    parser.add_argument('--skip-tts', action='store_true',
                       help='跳过TTS配音生成（使用JSON中的duration）')
    parser.add_argument('--upscale-only',
                       help='仅执行 Real-ESRGAN 超分（输入视频路径）')
    parser.add_argument('--upscale-output',
                       help='仅超分模式下的输出路径（可选）')
    
    args = parser.parse_args()
    
    if args.upscale_only:
        config_path = Path(args.gen_video_config) if args.gen_video_config else None
        output_path = Path(args.upscale_output) if args.upscale_output else None
        success = run_upscale_only(Path(args.upscale_only), output_path, config_path)
        return 0 if success else 1
    
    required_args = {
        'json': args.json,
        'index': args.index,
        'metadata': args.metadata,
        'scenes': args.scenes,
        'video_dir': args.video_dir,
        'output': args.output,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        print(f"❌ 缺少必要参数: {', '.join(missing)} （或使用 --upscale-only）")
        return 1
    
    opening_video = Path(args.opening) if args.opening else None
    ending_video = Path(args.ending) if args.ending else None
    gen_video_config = Path(args.gen_video_config) if args.gen_video_config else None
    
    success = generate_video_from_json_complete(
        Path(args.json),
        Path(args.index),
        Path(args.metadata),
        [Path(f) for f in args.scenes],
        Path(args.video_dir),
        opening_video,
        ending_video,
        Path(args.output),
        gen_video_config=gen_video_config,
        skip_opening=args.skip_opening,
        skip_ending=args.skip_ending,
        skip_tts=args.skip_tts
    )
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())