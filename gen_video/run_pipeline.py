#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行完整的视频生成流水线
"""

import os
import sys
import argparse
from typing import List
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from script_parser import ScriptParser
from main import AIVideoPipeline


def main():
    parser = argparse.ArgumentParser(description="运行AI视频生成流水线")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--markdown", type=str, default="./灵界/2.md", help="Markdown 脚本路径")
    parser.add_argument("--image-dir", type=str, default="../灵界/img2", help="图像目录路径")
    parser.add_argument("--output", type=str, default="lingjie_ep2", help="输出文件名")
    parser.add_argument("--max-scenes", type=int, default=None, help="仅处理前N个场景")
    parser.add_argument("--skip-image", action="store_true", help="跳过图像生成（使用现有图像）")
    parser.add_argument("--skip-video", action="store_true", help="跳过视频生成（使用已有视频）")
    parser.add_argument("--skip-tts", action="store_true", help="跳过TTS生成")
    parser.add_argument("--skip-subtitle", action="store_true", help="跳过字幕生成")
    parser.add_argument("--skip-compose", action="store_true", help="跳过视频合成")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI视频生成流水线")
    print("=" * 60)
    
    # 1. 解析脚本
    print("\n[1/5] 解析 Markdown 脚本...")
    script_parser = ScriptParser(args.markdown, args.image_dir)
    scenes = script_parser.parse_scenes()
    total_scenes = len(scenes)
    if args.max_scenes and args.max_scenes > 0:
        scenes = scenes[:args.max_scenes]
        print(f"  已限制为前 {len(scenes)} 个场景（原始 {total_scenes} 个）")
    else:
        print(f"  总场景数: {len(scenes)}")
    opening = script_parser.extract_opening_narration()
    ending = script_parser.extract_ending_narration()
    include_ending = not (args.max_scenes and args.max_scenes > 0 and args.max_scenes < total_scenes)
    full_narration = script_parser.get_full_narration(
        scenes,
        include_opening=True,
        include_ending=include_ending,
    )
    def _collect_subtitle_segments() -> List[str]:
        segments: List[str] = []
        if opening:
            segments.append(opening)
        segments.extend(
            [scene.get("narration", "") for scene in scenes if scene.get("narration")]
        )
        if include_ending and ending:
            segments.append(ending)
        deduped: List[str] = []
        for text in segments:
            cleaned = text.strip()
            if not cleaned:
                continue
            if not deduped or deduped[-1] != cleaned:
                deduped.append(cleaned)
        return deduped

    subtitle_segments = _collect_subtitle_segments()
    
    print(f"✓ 解析完成: {len(scenes)} 个场景")
    print(f"  开场白: {len(opening)} 字符")
    print(f"  结束语: {len(ending)} 字符")
    print(f"  总旁白: {len(full_narration)} 字符")
    
    # 检查图像
    missing_images = [s for s in scenes if not s.get('image_path') or not os.path.exists(s.get('image_path', ''))]
    if missing_images:
        print(f"⚠ 警告: {len(missing_images)} 个场景缺少图像")
        for scene in missing_images[:5]:
            print(f"  场景 {scene['scene_number']}: {scene.get('image_path', 'N/A')}")
    else:
        print("✓ 所有场景都有对应图像")
    
    # 保存脚本为 JSON
    script_json_path = f"temp/{args.output}_script.json"
    os.makedirs("temp", exist_ok=True)
    script_data = script_parser.to_json(
        script_json_path,
        scenes,
        total_scene_count=total_scenes,
    )
    
    # 2. 初始化流水线
    print("\n[2/5] 初始化流水线...")
    pipeline = AIVideoPipeline(
        args.config,
        load_image=not args.skip_image,
        load_video=not args.skip_video,
        load_tts=not args.skip_tts,
        load_subtitle=not args.skip_subtitle,
        load_composer=not args.skip_compose,
    )

    # 尝试生成/补全场景图像
    if not args.skip_image:
        scenes = pipeline.ensure_scene_images(scenes, script_json_path, args.output)
        # 重新统计缺失图像
        missing_images = [s for s in scenes if not s.get('image_path') or not os.path.exists(s.get('image_path', ''))]
        if missing_images:
            print(f"⚠ 仍有 {len(missing_images)} 个场景缺少图像")
        else:
            print("✓ 场景图像准备就绪")
    
    # 3. 生成视频片段
    video_clips = []
    if not args.skip_video:
        print("\n[3/5] 生成视频片段...")
        video_clips = pipeline.generate_video_clips(scenes, args.output)
        print(f"✓ 生成 {len(video_clips)} 个视频片段")
    else:
        print("\n[3/5] 跳过视频生成")
        # 查找已有视频
        video_dir = Path(pipeline.paths['output_dir']) / args.output / "videos"
        if video_dir.exists():
            video_clips = sorted([str(f) for f in video_dir.glob("*.mp4")])
            print(f"  使用已有视频: {len(video_clips)} 个")
    
    # 4. 生成配音
    audio_path = None
    if not args.skip_tts and full_narration:
        print("\n[4/5] 生成配音...")
        try:
            audio_path = pipeline.generate_audio(full_narration, args.output)
            if audio_path:
                print(f"✓ 配音已生成: {audio_path}")
        except Exception as e:
            print(f"✗ 配音生成失败: {e}")
            print("  继续执行（可以稍后手动添加配音）")
    else:
        print("\n[4/5] 跳过配音生成")
    
    # 5. 生成字幕
    subtitle_path = None
    if not args.skip_subtitle and audio_path and os.path.exists(audio_path):
        print("\n[5/6] 生成字幕...")
        try:
            subtitle_path = pipeline.generate_subtitle(
                audio_path,
                args.output,
                narration=full_narration,
                segments=subtitle_segments,
            )
            if subtitle_path:
                print(f"✓ 字幕已生成: {subtitle_path}")
        except Exception as e:
            print(f"✗ 字幕生成失败: {e}")
            print("  继续执行（可以稍后手动添加字幕）")
    else:
        print("\n[5/6] 跳过字幕生成")
    
    # 6. 合成最终视频
    if not args.skip_compose and video_clips:
        print("\n[6/6] 合成最终视频...")
        try:
            final_video = pipeline.compose_final_video(
                video_clips,
                audio_path,
                subtitle_path,
                args.output,
                scenes=scenes,
            )
            if final_video:
                print(f"\n{'=' * 60}")
                print(f"✓ 完成！最终视频: {final_video}")
                print(f"{'=' * 60}")
        except Exception as e:
            print(f"✗ 视频合成失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[6/6] 跳过视频合成")
    
    print("\n流水线执行完成！")


if __name__ == "__main__":
    main()

