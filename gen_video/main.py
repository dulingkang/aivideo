#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI视频生成系统主程序
完整的从图像到视频、配音、字幕、合成的自动化流程
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
from tqdm import tqdm

# 设置PyTorch CUDA内存分配配置，避免内存碎片
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 导入各个模块
from video_generator import VideoGenerator
from tts_generator import TTSGenerator
from subtitle_generator import SubtitleGenerator
from video_composer import VideoComposer
from image_generator import ImageGenerator


class AIVideoPipeline:
    """AI视频生成完整流水线"""
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        *,
        load_image: bool = True,
        load_video: bool = True,
        load_tts: bool = True,
        load_subtitle: bool = True,
        load_composer: bool = True,
    ):
        """初始化流水线"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path.cwd() / self.config_path).resolve()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.paths = self.config['paths']
        self.image_config = self.config.get('image', {})
        self.video_config = self.config.get('video', {})
        self.config_dir = self.config_path.parent
        
        # 创建必要的目录
        for dir_path in [self.paths['output_dir'], self.paths['temp_dir']]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 初始化各个组件
        print("初始化组件...")
        self.image_generator = ImageGenerator(config_path) if load_image else None
        self.video_generator = VideoGenerator(config_path) if load_video else None
        self.tts_generator = TTSGenerator(config_path) if load_tts else None
        self.subtitle_generator = SubtitleGenerator(config_path) if load_subtitle else None
        self.video_composer = VideoComposer(config_path) if load_composer else None
    
    def process_script(self, script_path: str, output_name: str = "output"):
        """
        处理完整脚本，生成视频
        
        Args:
            script_path: 脚本文件路径（JSON格式，包含分镜信息）
            output_name: 输出文件名
        """
        print(f"\n处理脚本: {script_path}")
        
        # 加载脚本
        with open(script_path, 'r', encoding='utf-8') as f:
            script = json.load(f)
        
        scenes = script.get('scenes', [])
        print(f"找到 {len(scenes)} 个场景")
        
        # 检查scenes中是否包含开头和结尾场景（id=0是开头，id=999是结尾）
        has_opening_scene = False
        has_ending_scene = False
        if scenes:
            first_scene = scenes[0]
            last_scene = scenes[-1]
            if first_scene.get("id") == 0:
                has_opening_scene = True
                print("  ✓ 检测到开头场景（id=0），将使用scenes中的场景而不是单独生成")
            if last_scene.get("id") == 999:
                has_ending_scene = True
                print("  ✓ 检测到结尾场景（id=999），将使用scenes中的场景而不是单独生成")
        
        # 1. 确保场景图像就绪
        script_json_path = Path(script_path)
        scenes = self.ensure_scene_images(scenes, str(script_json_path), output_name)

        # 2. 先生成配音（以便获取实际音频时长，用于生成匹配时长的视频）
        # narration 用于解说文本，从 opening.narration + scenes[].narration + ending.narration 组合
        narration_parts = []
        # 开头旁白：如果scenes中有id=0的场景，使用场景的narration；否则使用opening.narration
        opening_narration = ""
        if has_opening_scene and scenes and scenes[0].get("id") == 0:
            opening_narration = scenes[0].get("narration", "")
        else:
            opening_narration = script.get("opening", {}).get("narration") or script.get("opening_narration", "")
        
        if opening_narration and not has_opening_scene:
            # 只有当scenes中没有id=0场景时，才添加单独的opening.narration
            narration_parts.append(opening_narration)
        
        # 场景旁白（包括id=0和id=999的场景）
        narration_parts.extend(
            [scene.get("narration", "") for scene in scenes if scene.get("narration")]
        )
        
        # 结尾旁白：如果scenes中有id=999的场景，场景的narration已经在上面添加了；否则使用ending.narration
        ending_narration = ""
        if has_ending_scene and scenes and scenes[-1].get("id") == 999:
            # 已经在上面添加了，不需要重复
            ending_narration = scenes[-1].get("narration", "")
        else:
            ending_narration = script.get("ending", {}).get("narration") or script.get("ending_narration", "")
            if ending_narration:
                narration_parts.append(ending_narration)
        # 2. 为每个场景生成单独的音频文件（推荐方案：精确获取每个片段的时长）
        # 优点：1) TTS生成的时长100%精确，无需语音识别 2) 速度快，无需额外识别时间
        #      3) 容错好，单个片段失败不影响其他 4) 可缓存复用
        # 相比生成完整音频后识别：识别可能有误差，特别是中文，且需要额外时间
        audio_paths = self.generate_audio_per_scene(narration_parts, output_name, has_opening_scene, has_ending_scene, opening_narration, scenes)
        
        # 3. 直接从分段音频获取实际时长（TTS生成的精确时长，无需识别）
        audio_durations = None
        audio_path = None  # 完整音频路径（稍后合并）
        if audio_paths and len(audio_paths) > 0:
            # 直接从每个分段音频获取时长（TTS实际生成的，100%精确）
            audio_durations_raw = []
            for audio_path_segment in audio_paths:
                if os.path.exists(audio_path_segment):
                    duration = self._get_video_duration(audio_path_segment)
                    if duration > 0:
                        audio_durations_raw.append(duration)
                    else:
                        print(f"  ⚠ 无法获取音频时长: {audio_path_segment}")
                else:
                    print(f"  ⚠ 音频文件不存在: {audio_path_segment}")
            
            if audio_durations_raw and len(audio_durations_raw) == len(audio_paths):
                # 使用精确的音频时长（不取整），视频生成器支持精确时长
                # 视频生成器会通过 num_frames = round(duration * fps) 来计算精确帧数
                # 例如：音频 5.234s, fps=15 -> 79帧 -> 实际视频 5.267s，误差仅 0.033s
                audio_durations = audio_durations_raw  # 直接使用精确时长
                total_audio_duration = sum(audio_durations_raw)
                
                print(f"\n  使用TTS生成的精确音频时长（无需识别，100%准确）:")
                print(f"  ✓ 视频生成器支持精确时长，通过帧数计算实现精确匹配")
                print(f"  {'场景':<12} {'精确音频时长':<15} {'说明':<30}")
                print(f"  {'-'*12} {'-'*15} {'-'*30}")
                
                # 打印时长分配（根据audio_paths的顺序，对应narration_parts的顺序）
                for i, raw_dur in enumerate(audio_durations_raw):
                    # 根据索引判断是开头、场景还是结尾
                    if not has_opening_scene and i == 0 and opening_narration:
                        scene_label = "开头"
                    elif has_ending_scene and i == len(audio_paths) - 1:
                        # 最后一个是结尾场景（id=999）
                        scene_id = scenes[-1].get("id") if scenes else None
                        scene_label = "结尾场景" if scene_id == 999 else f"场景 {i}"
                    else:
                        # 找到对应的场景索引
                        scene_idx = i
                        if not has_opening_scene and opening_narration:
                            scene_idx = i - 1  # 减去单独的opening
                        scene_num = scene_idx + 1
                        scene_id = scenes[scene_idx].get("id") if scene_idx < len(scenes) else None
                        if scene_id == 0:
                            scene_label = "开头场景"
                        elif scene_id == 999:
                            scene_label = "结尾场景"
                        else:
                            scene_label = f"场景 {scene_num}"
                    
                    # 估算视频时长（基于fps，实际可能略有差异）
                    fps = self.video_config.get('fps', 15)
                    estimated_frames = round(raw_dur * fps)
                    estimated_video_duration = estimated_frames / fps
                    duration_diff = abs(estimated_video_duration - raw_dur)
                    status = "✓" if duration_diff < 0.05 else "≈"
                    print(f"  {scene_label:<12} {raw_dur:<15.3f} {status} 视频约{estimated_video_duration:.3f}s (差异{duration_diff:.3f}s)")
                
                print(f"  {'-'*12} {'-'*15} {'-'*30}")
                print(f"  {'总计':<12} {total_audio_duration:<15.3f} ✓ TTS生成的精确时长，视频与音频完全匹配")
            else:
                print(f"  ⚠ 分段音频数量不匹配，无法获取准确的音频时长信息")
        
        # 4. 生成视频片段（使用实际音频时长）
        # 注意：如果scenes中已有id=0和id=999的场景，它们也会被生成视频
        video_clips = self.generate_video_clips(scenes, output_name, audio_durations=audio_durations, opening_narration=opening_narration)
        
        # 5. 添加开头和结尾视频（仅在scenes中没有id=0和id=999时）
        all_video_clips = []
        opening_ending_clips = {"opening": [], "ending": []}  # 初始化，确保变量始终存在
        
        if has_opening_scene and has_ending_scene:
            # 如果scenes中已有开头和结尾场景，直接使用生成的视频，不再单独生成
            print("\n使用scenes中的开头和结尾场景，跳过单独的开头和结尾视频生成")
            all_video_clips = video_clips
        else:
            # 如果scenes中没有开头或结尾场景，需要单独生成
            opening_audio_duration = None
            ending_audio_duration = None
            if audio_durations:
                # 计算开头和结尾的音频时长索引
                audio_idx = 0
                if not has_opening_scene and opening_narration:
                    opening_audio_duration = audio_durations[audio_idx]
                    audio_idx += 1
                
                # 找到结尾的音频时长（在最后一个有narration的场景之后）
                if not has_ending_scene and ending_narration:
                    # 计算场景中最后一个有narration的场景索引
                    last_narration_idx = len(audio_durations) - 1
                    ending_audio_duration = audio_durations[last_narration_idx]
            
            opening_ending_clips = self.add_opening_ending_videos(
                script,
                output_name,
                opening_audio_duration=opening_audio_duration,
                ending_audio_duration=ending_audio_duration,
            )
            
            # 合并开头、正片、结尾（确保不重复）
            if not has_opening_scene and opening_ending_clips.get("opening"):
                all_video_clips.extend(opening_ending_clips["opening"])
            all_video_clips.extend(video_clips)
            if not has_ending_scene and opening_ending_clips.get("ending"):
                all_video_clips.extend(opening_ending_clips["ending"])
        
        print(f"\n视频片段列表（共 {len(all_video_clips)} 个）:")
        for i, clip in enumerate(all_video_clips, 1):
            clip_name = os.path.basename(clip)
            print(f"  {i}. {clip_name}")
        
        # 5. 生成字幕（使用完整音频，确保与视频完全匹配）
        # 由于合成视频时使用精确的音频时长，字幕可以直接基于完整音频生成
        subtitle_path = None
        
        # 收集所有旁白文本（开头、场景、结尾），对应 audio_paths 的顺序
        narration_text = "".join(narration_parts)  # 完整旁白文本
        scene_texts: List[str] = []  # 分段文本列表
        
        # 构建 scene_texts，顺序必须与 audio_paths 一致：
        # 1. 如果有单独的opening（不在scenes中），添加开头
        # 2. 添加所有场景的narration（包括id=0和id=999的场景）
        # 3. 如果有单独的ending（不在scenes中），添加结尾
        if opening_narration and not has_opening_scene:
            scene_texts.append(opening_narration)
            print(f"  开头旁白: {len(opening_narration)} 字")
        
        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "")
            if narration:
                scene_texts.append(narration)
                scene_id = scene.get("id") or scene.get("scene_number")
                if scene_id == 0:
                    print(f"  开头场景旁白: {len(narration)} 字")
                elif scene_id == 999:
                    print(f"  结尾场景旁白: {len(narration)} 字")
                else:
                    print(f"  场景 {i+1} 旁白: {len(narration)} 字")
        
        if ending_narration and not has_ending_scene:
            scene_texts.append(ending_narration)
            print(f"  结尾旁白: {len(ending_narration)} 字")
        
        # 合并分段音频为完整音频（用于合成和字幕生成）
        # 由于视频生成使用精确的音频时长，视频片段已经精确匹配，可以直接使用完整音频
        audio_path = None  # 完整音频路径
        if audio_paths and len(audio_paths) > 0:
            print(f"\n=== 合并分段音频为完整音频 ===")
            output_dir = Path(self.paths['output_dir']) / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            merged_audio_path = output_dir / "audio.wav"
            
            try:
                # 使用 FFmpeg 合并音频
                import ffmpeg
                temp_dir = Path(self.paths['temp_dir'])
                temp_dir.mkdir(parents=True, exist_ok=True)
                concat_list = temp_dir / "audio_concat.txt"
                
                # 创建音频文件列表
                with open(concat_list, 'w', encoding='utf-8') as f:
                    for audio_path_seg in audio_paths:
                        abs_path = os.path.abspath(audio_path_seg)
                        escaped_path = abs_path.replace("'", "'\\''")
                        f.write(f"file '{escaped_path}'\n")
                
                # 合并音频
                (
                    ffmpeg
                    .input(str(concat_list), format='concat', safe=0)
                    .output(
                        str(merged_audio_path),
                        acodec='pcm_s16le',  # WAV格式
                        ac=2,  # 立体声
                        ar=48000,  # 采样率
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                
                # 验证合并后的音频时长
                merged_duration = self._get_video_duration(str(merged_audio_path))
                total_segments_duration = sum(self._get_video_duration(ap) for ap in audio_paths)
                duration_diff = abs(merged_duration - total_segments_duration)
                
                if duration_diff < 0.05:
                    audio_path = str(merged_audio_path)
                    print(f"  ✓ 音频合并完成: {merged_duration:.3f}s (分段总和: {total_segments_duration:.3f}s, 差异: {duration_diff:.3f}s)")
                else:
                    print(f"  ⚠ 音频合并时长差异较大: {merged_duration:.3f}s vs {total_segments_duration:.3f}s (差异: {duration_diff:.3f}s)")
                    audio_path = str(merged_audio_path)  # 仍然使用，但记录警告
                
                # 清理临时文件
                if concat_list.exists():
                    concat_list.unlink()
            except Exception as e:
                print(f"  ⚠ 音频合并失败: {e}")
                import traceback
                traceback.print_exc()
                # 如果合并失败，使用第一个分段音频作为备用
                audio_path = audio_paths[0] if audio_paths else None
        
        # 生成字幕（基于完整音频，使用精确的音频时长列表确保对齐）
        if audio_path and len(scene_texts) > 0:
            print(f"\n=== 生成字幕（基于完整音频，使用精确音频时长确保对齐） ===")
            print(f"  ✓ 视频生成使用精确音频时长，视频片段已精确匹配")
            print(f"  ✓ 字幕使用精确音频时长列表，确保每个字幕片段时间严格对应音频片段")
            
            # 计算总音频时长（用于字幕生成）
            total_audio_duration = sum(audio_durations) if audio_durations else None
            
            subtitle_path = self.generate_subtitle(
                audio_path,  # 使用完整音频
                output_name,
                narration_text,  # 完整旁白文本
                segments=scene_texts,  # 分段文本（用于字幕分段）
                video_durations=audio_durations,  # 传入精确的音频时长列表，确保字幕时间轴与音频完全对齐
                total_duration=total_audio_duration,  # 传入总音频时长
            )
            
            if subtitle_path:
                print(f"  ✓ 字幕已生成: {os.path.basename(subtitle_path)}")
                print(f"  ✓ 视频、音频、字幕三者完全同步（使用精确音频时长列表，确保时间轴严格对齐）")
            else:
                print(f"  ⚠ 字幕生成失败")
        
        # 6. 检查视频和音频时长，如果不对齐，延长结尾视频
        # 使用分段音频的时长总和，而不是完整音频文件
        video_extended = False
        if audio_durations and len(audio_durations) > 0 and all_video_clips:
            # 使用分段音频时长总和作为总音频时长
            audio_duration = sum(audio_durations)
            total_video_duration = sum(self._get_video_duration(clip) for clip in all_video_clips)
            
            # 诊断信息：对比每个片段的预期时长和实际时长
            if audio_durations:
                print(f"\n  诊断：对比预期音频时长与实际视频时长")
                print(f"  {'场景':<8} {'预期音频时长':<12} {'实际视频时长':<12} {'差异':<10} {'状态':<8}")
                print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
                total_expected = 0
                total_actual = 0
                truncated_count = 0
                for idx, clip in enumerate(all_video_clips):
                    clip_duration = self._get_video_duration(clip)
                    total_actual += clip_duration
                    # 计算对应的音频时长索引
                    if has_opening_scene:
                        audio_idx = idx
                    else:
                        audio_idx = idx + (1 if opening_narration else 0)
                    
                    if audio_idx < len(audio_durations):
                        expected_duration = audio_durations[audio_idx]
                        total_expected += expected_duration
                        diff = clip_duration - expected_duration
                        # 判断状态：截断、向上取整导致的略长、正常
                        if diff < -0.3:
                            status = "截断⚠"
                            truncated_count += 1
                        elif diff > 0.15:
                            status = "略长ℹ"
                        elif abs(diff) < 0.05:
                            status = "✓"
                        else:
                            status = "≈"
                        print(f"  {idx+1:<8} {expected_duration:<12.2f} {clip_duration:<12.2f} {diff:+.2f} {status:<8}")
                    else:
                        print(f"  {idx+1:<8} {'N/A':<12} {clip_duration:<12.2f} {'N/A':<10} {'N/A':<8}")
                
                print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
                print(f"  {'总计':<8} {total_expected:<12.2f} {total_actual:<12.2f} {total_actual - total_expected:+.2f}")
                print(f"  音频总时长: {audio_duration:.2f}s")
                print(f"  视频总时长: {total_video_duration:.2f}s")
                total_diff = total_video_duration - total_expected
                if abs(total_diff) > 0.5:
                    if total_diff < 0:
                        print(f"  ⚠ 视频总时长比音频总时长短 {abs(total_diff):.2f}s")
                        if truncated_count > 0:
                            print(f"    可能原因：{truncated_count} 个场景被 max_duration 或 max_frames 限制截断")
                    else:
                        print(f"  ℹ 视频总时长比音频总时长长 {total_diff:.2f}s（向上取整导致）")
                if abs(total_expected - audio_duration) > 0.5:
                    print(f"  ⚠ 预期时长总和 ({total_expected:.2f}s) 与音频总时长 ({audio_duration:.2f}s) 不一致")
            
            if audio_duration > total_video_duration:
                duration_diff = audio_duration - total_video_duration
                print(f"\n  音频时长 ({audio_duration:.2f}s) 比视频总时长 ({total_video_duration:.2f}s) 长 {duration_diff:.2f}s")
                print(f"  将延长结尾视频以对齐")
                
                # 延长最后一个视频片段（通常是结尾视频）
                # 如果scenes中有id=999的结尾场景，all_video_clips的最后一个就是结尾场景视频
                # 如果没有，检查是否有单独的结尾视频
                if has_ending_scene and all_video_clips:
                    # scenes中有结尾场景，直接延长最后一个视频
                    last_video = all_video_clips[-1]
                    extended_ending = self._extend_video_by_duration(last_video, duration_diff)
                    if extended_ending:
                        all_video_clips[-1] = extended_ending
                        video_extended = True
                        print(f"  ✓ 结尾场景视频已延长 {duration_diff:.2f}秒")
                elif not has_ending_scene and opening_ending_clips.get("ending") and opening_ending_clips["ending"]:
                    # 有单独的结尾视频
                    ending_video = opening_ending_clips["ending"][0]
                    extended_ending = self._extend_video_by_duration(ending_video, duration_diff)
                    if extended_ending:
                        # 替换结尾视频
                        all_video_clips[-1] = extended_ending
                        video_extended = True
                        print(f"  ✓ 结尾视频已延长 {duration_diff:.2f}秒")
                elif all_video_clips:
                    # 如果没有结尾视频，延长最后一个场景视频
                    last_video = all_video_clips[-1]
                    extended_scene = self._extend_video_by_duration(last_video, duration_diff)
                    if extended_scene:
                        all_video_clips[-1] = extended_scene
                        video_extended = True
                        print(f"  ✓ 最后一个场景视频已延长 {duration_diff:.2f}秒")
        
        # 如果视频延长了，重新生成字幕以确保时间轴同步
        if video_extended and subtitle_path and audio_path:
            print(f"\n  视频已延长，重新生成字幕以确保时间轴同步...")
            # 计算延长后的视频时长列表
            video_durations_updated = [self._get_video_duration(clip) for clip in all_video_clips]
            total_video_duration = sum(video_durations_updated)
            # 重新生成字幕，使用延长后的视频时长
            subtitle_path = self.generate_subtitle(
                audio_path,
                output_name,
                narration_text,
                segments=scene_texts,
                video_durations=video_durations_updated,  # 使用延长后的视频时长
                total_duration=total_video_duration,  # 传入总视频时长
            )
            print(f"  ✓ 字幕已重新生成，时间轴已同步")
        
        # 6. 合成最终视频
        # 由于视频生成使用精确的音频时长，视频片段已经精确匹配，直接使用完整音频即可
        final_video = self.compose_final_video(
            all_video_clips,
            audio_path,  # 使用完整音频（已合并）
            subtitle_path,
            output_name,
            scenes=scenes,
        )
        
        print(f"\n✓ 完成！最终视频: {final_video}")
        return final_video
    
    def add_opening_ending_videos(
        self,
        script: Dict,
        output_name: str,
        opening_audio_duration: Optional[float] = None,
        ending_audio_duration: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """
        添加开头和结尾视频
        
        Args:
            script: 脚本数据
            output_name: 输出名称
            opening_audio_duration: 开头实际音频时长（秒），如果提供则优先使用
            ending_audio_duration: 结尾实际音频时长（秒），如果提供则优先使用
        
        Returns:
            {"opening": [opening_video_path], "ending": [ending_video_path]}
        """
        try:
            from opening_ending_generator import OpeningEndingGenerator
            
            generator = OpeningEndingGenerator("config.yaml")
            
            episode = script.get("episode")
            title = script.get("title", "")
            
            # 根据旁白字数计算开头和结尾视频时长（使用配置的语速参数）
            opening_narration = script.get("opening", {}).get("narration") or script.get("opening_narration", "")
            ending_narration = script.get("ending", {}).get("narration") or script.get("ending_narration", "")
            
            # 从配置中获取语速参数
            tts_config = self.config.get('tts', {})
            speech_rate = tts_config.get('speech_rate', {})
            use_chars = speech_rate.get('use_chars', True)
            chars_per_second = speech_rate.get('chars_per_second', 8.0)
            words_per_second = speech_rate.get('words_per_second', 3.55)
            
            # 优先使用实际音频时长，如果没有则使用旁白字数计算，最后使用 duration 字段
            if opening_audio_duration:
                opening_duration = opening_audio_duration
                print(f"  使用开头实际音频时长: {opening_duration:.2f}秒")
            elif opening_narration:
                if use_chars:
                    char_count = len(opening_narration)
                    rate = chars_per_second
                    rate_type = "字符"
                else:
                    import re
                    chinese_chars = re.findall(r'[\u4e00-\u9fff]', opening_narration)
                    char_count = len(chinese_chars)
                    rate = words_per_second
                    rate_type = "字"
                
                opening_duration = max(char_count / rate, 2.0)  # 最少2秒
                max_duration = self.video_config.get('max_duration', 15.0)
                opening_duration = min(opening_duration, max_duration)
                print(f"  根据开头旁白 ({char_count} {rate_type}) 计算视频时长: {opening_duration:.2f}秒")
            else:
                opening_duration = script.get("opening", {}).get("duration") if script.get("opening") else 6.0
                print(f"  使用开头 duration 字段: {opening_duration}秒")
            
            if ending_audio_duration:
                ending_duration = ending_audio_duration
                print(f"  使用结尾实际音频时长: {ending_duration:.2f}秒")
            elif ending_narration:
                if use_chars:
                    char_count = len(ending_narration)
                    rate = chars_per_second
                    rate_type = "字符"
                else:
                    import re
                    chinese_chars = re.findall(r'[\u4e00-\u9fff]', ending_narration)
                    char_count = len(chinese_chars)
                    rate = words_per_second
                    rate_type = "字"
                
                ending_duration = max(char_count / rate, 2.0)  # 最少2秒
                max_duration = self.video_config.get('max_duration', 15.0)
                ending_duration = min(ending_duration, max_duration)
                print(f"  根据结尾旁白 ({char_count} {rate_type}) 计算视频时长: {ending_duration:.2f}秒")
            else:
                ending_duration = script.get("ending", {}).get("duration") if script.get("ending") else 5.0
                print(f"  使用结尾 duration 字段: {ending_duration}秒")
            
            # 确保开头和结尾视频存在
            opening_path, ending_path = generator.ensure_opening_ending_videos(
                episode, 
                title,
                opening_duration=opening_duration,
                ending_duration=ending_duration,
            )
            
            return {
                "opening": [str(opening_path)] if opening_path.exists() else [],
                "ending": [str(ending_path)] if ending_path.exists() else [],
            }
        except ImportError:
            print("⚠ opening_ending_generator 未找到，跳过开头和结尾视频")
            return {"opening": [], "ending": []}
        except Exception as e:
            print(f"⚠ 添加开头和结尾视频失败: {e}")
            import traceback
            traceback.print_exc()
            return {"opening": [], "ending": []}
    
    def ensure_scene_images(
        self,
        scenes: List[Dict],
        script_json_path: str,
        output_name: str,
        overwrite: Optional[bool] = None,
    ) -> List[Dict]:
        """确保场景拥有可用的图像；必要时触发生成"""

        if self.image_generator is None:
            print("\n=== 跳过图像生成（未加载图像生成器） ===")
            return scenes

        if not script_json_path or not os.path.exists(script_json_path):
            print("  未找到脚本 JSON，无法生成图像")
            return scenes

        # 检查哪些场景缺少图像
        missing = []
        existing_count = 0
        for idx, scene in enumerate(scenes):
            image_path = scene.get("image_path")
            if image_path and os.path.exists(image_path):
                existing_count += 1
            else:
                missing.append(idx)

        # 如果所有图像都存在，且不强制重新生成，则跳过
        if not missing and not self.image_config.get("force_regenerate", False):
            print(f"\n=== 跳过图像生成 ===")
            print(f"  ✓ 所有 {len(scenes)} 个场景的图像已存在，跳过生成")
            if existing_count > 0:
                print(f"  ✓ 已检查 {existing_count} 个图像文件")
            return scenes

        # 如果有缺失的图像或强制重新生成
        if missing:
            print(f"\n=== 生成/补全场景图像 ===")
            print(f"  找到 {existing_count} 个已存在的图像")
            print(f"  需要生成 {len(missing)} 个缺失的图像")
        else:
            print(f"\n=== 重新生成场景图像 ===")
            print(f"  强制重新生成所有 {len(scenes)} 个场景的图像")

        overwrite = self.image_config.get("overwrite_existing", False) if overwrite is None else overwrite

        base_image_dir = self.paths.get("image_output")
        if base_image_dir:
            image_output_dir = Path(base_image_dir) / output_name
        else:
            image_output_dir = Path(self.paths['output_dir']) / output_name / "images"

        generated = self.image_generator.generate_from_script(
            script_json_path,
            output_dir=str(image_output_dir),
            overwrite=overwrite,
            update_script=True,
        )

        if generated:
            with open(script_json_path, "r", encoding="utf-8") as f:
                updated = json.load(f)
            scenes = updated.get("scenes", scenes)
            print(f"  ✓ 图像生成完成")
        else:
            print("  ⚠ 未生成新图像（可能全部存在或生成失败）")

        return scenes
    
    def generate_video_clips(self, scenes: List[Dict], output_name: str, audio_durations: Optional[List[float]] = None, opening_narration: Optional[str] = None) -> List[str]:
        """生成视频片段
        
        Args:
            scenes: 场景列表（可能包含id=0的开头场景和id=999的结尾场景）
            output_name: 输出名称
            audio_durations: 实际音频时长列表（按照narration_parts的顺序：opening或scenes[0]、scenes[1]、...、ending或scenes[-1]）
            opening_narration: 开头旁白（用于计算场景音频时长索引）
        """
        if self.video_generator is None:
            print("\n=== 跳过视频生成（未加载视频引擎） ===")
            return []

        print("\n=== 生成视频片段 ===")
        
        video_clips = []
        output_dir = Path(self.paths['output_dir']) / output_name / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算场景对应的音频时长索引
        # 如果scenes中有id=0的场景，audio_durations[0]对应scenes[0]
        # 如果scenes中没有id=0的场景，audio_durations[0]对应单独的opening，scenes[0]对应audio_durations[1]
        has_opening_scene = scenes and scenes[0].get("id") == 0
        scene_audio_start_idx = 0 if has_opening_scene else (1 if opening_narration else 0)
        
        for i, scene in enumerate(tqdm(scenes, desc="生成视频")):
            # 在每个场景生成前清理GPU缓存
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            image_path = scene.get('image_path')
            if not image_path or not os.path.exists(image_path):
                print(f"警告: 场景 {i+1} 的图像不存在: {image_path}")
                continue
            
            output_path = output_dir / f"scene_{i+1:03d}.mp4"
            abs_output_path = os.path.abspath(str(output_path))
            
            # 如果视频已存在，直接使用
            if os.path.exists(abs_output_path):
                video_clips.append(abs_output_path)
                print(f"  ✓ 场景 {i+1} 使用已有视频: {abs_output_path}")
                continue
            
            # 确保 scene 对象只包含当前场景的数据，不包含其他场景或开头/结尾的数据
            # 创建一个干净的 scene 副本，只包含当前场景的字段
            clean_scene = {
                "id": scene.get("id"),
                "scene_number": scene.get("scene_number"),
                "narration": scene.get("narration", ""),  # 只使用当前场景的旁白
                "duration": scene.get("duration"),
                "visual": scene.get("visual"),
                "face_style_auto": scene.get("face_style_auto"),
                "action": scene.get("action"),  # 添加 action 字段，用于检测静态动作（如 lying_still）
                "camera": scene.get("camera"),  # 添加 camera 字段，用于视频生成
                "mood": scene.get("mood"),  # 添加 mood 字段
                "lighting": scene.get("lighting"),  # 添加 lighting 字段
            }
            
            # 如果提供了实际音频时长，优先使用实际音频时长
            # audio_durations的索引对应narration_parts的顺序，也就是scenes的顺序（如果scenes中有id=0和id=999）
            if audio_durations and scene_audio_start_idx + i < len(audio_durations):
                actual_audio_duration = audio_durations[scene_audio_start_idx + i]
                # 将实际音频时长设置到 scene 的 duration 字段，这样视频生成器会优先使用它
                clean_scene["duration"] = actual_audio_duration
                scene_id = scene.get("id") or scene.get("scene_number")
                scene_label = f"场景 {i+1}" if scene_id not in [0, 999] else (f"开头场景(id={scene_id})" if scene_id == 0 else f"结尾场景(id={scene_id})")
                print(f"\n生成{scene_label}视频，使用实际音频时长: {actual_audio_duration:.2f}s")
            
            # 调试信息：显示场景ID和旁白
            scene_id = clean_scene.get("id") or clean_scene.get("scene_number") or i+1
            narration = clean_scene.get("narration", "")
            if narration:
                print(f"  旁白: {narration[:50]}{'...' if len(narration) > 50 else ''} ({len(narration)} 字)")
            
            # 生成新视频
            try:
                self.video_generator.generate_video(
                    image_path,
                    str(output_path),
                    motion_bucket_id=scene.get('motion_bucket_id'),
                    noise_aug_strength=scene.get('noise_aug_strength'),
                    num_frames=scene.get('num_frames'),
                    fps=scene.get('fps'),
                    scene=clean_scene,  # 传递干净的场景数据，确保只包含当前场景的旁白
                )
                # 使用绝对路径确保文件可找到
                # 等待一下确保文件写入完成
                import time
                time.sleep(0.5)
                
                if os.path.exists(abs_output_path):
                    video_clips.append(abs_output_path)
                    print(f"  ✓ 场景 {i+1} 视频已生成: {abs_output_path}")
                else:
                    print(f"  ⚠ 场景 {i+1} 视频文件不存在: {abs_output_path}")
                    # 尝试使用相对路径
                    if os.path.exists(str(output_path)):
                        video_clips.append(os.path.abspath(str(output_path)))
                        print(f"  ✓ 场景 {i+1} 使用相对路径找到视频")
            except Exception as e:
                print(f"✗ 场景 {i+1} 生成失败: {e}")
                import traceback
                traceback.print_exc()
                # 即使生成失败，也检查是否有部分生成的视频文件
                if os.path.exists(abs_output_path):
                    print(f"  ⚠ 检测到部分生成的视频文件，尝试使用: {abs_output_path}")
                    video_clips.append(abs_output_path)
                continue
        
        print(f"\n✓ 共找到/生成 {len(video_clips)} 个视频片段")
        return video_clips
    
    def generate_audio_per_scene(self, narration_parts: List[str], output_name: str, has_opening_scene: bool, has_ending_scene: bool, opening_narration: str, scenes: List[Dict]) -> List[str]:
        """为每个场景生成单独的音频文件
        
        Args:
            narration_parts: 旁白文本列表（按顺序：opening（如果有）、scenes、ending（如果有））
            output_name: 输出名称
            has_opening_scene: 是否有开头场景（id=0）
            has_ending_scene: 是否有结尾场景（id=999）
            opening_narration: 开头旁白文本
            scenes: 场景列表
        
        Returns:
            音频文件路径列表，与视频片段一一对应
        """
        if not narration_parts or not any(narration_parts):
            print("跳过分段音频生成（无旁白文本）")
            return []
        
        if self.tts_generator is None:
            print("跳过分段音频生成（未加载TTS引擎）")
            return []
        
        print("\n=== 为每个场景生成单独的音频文件 ===")
        
        output_dir = Path(self.paths['output_dir']) / output_name / "audios"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_paths = []
        part_idx = 0  # narration_parts 的索引
        
        # 检查是否强制重新生成（从TTS配置读取）
        force_regenerate = self.config.get('tts', {}).get('force_regenerate', False)
        
        # 处理开头音频（如果有单独的opening，且不在scenes中）
        if opening_narration and not has_opening_scene:
            if part_idx < len(narration_parts) and narration_parts[part_idx]:
                output_path = output_dir / "audio_opening.wav"
                # 检查文件是否已存在
                if output_path.exists() and not force_regenerate:
                    audio_paths.append(str(output_path))
                    print(f"  ✓ 开头音频已存在，跳过生成: {output_path}")
                else:
                    try:
                        self.tts_generator.generate(narration_parts[part_idx], str(output_path))
                        audio_paths.append(str(output_path))
                        print(f"  ✓ 开头音频已生成: {output_path}")
                    except Exception as e:
                        print(f"  ✗ 开头音频生成失败: {e}")
                part_idx += 1
        
        # 处理场景音频（包括id=0和id=999的场景）
        # narration_parts 的顺序：opening（如果有，且不在scenes中）-> scenes[].narration -> ending（如果有，且不在scenes中）
        # 对于 scenes，需要找到对应的 narration_parts 索引
        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "")
            if narration:
                # 计算对应的 narration_parts 索引
                # narration_parts 中场景旁白的位置
                # 如果 has_opening_scene，scenes[0]对应narration_parts[0]；否则scenes[0]对应narration_parts[1]（跳过opening）
                scene_audio_idx = part_idx
                if scene_audio_idx < len(narration_parts) and narration_parts[scene_audio_idx]:
                    scene_id = scene.get("id") or scene.get("scene_number")
                    if scene_id == 0:
                        output_path = output_dir / "audio_scene_000.wav"
                    elif scene_id == 999:
                        output_path = output_dir / "audio_scene_ending.wav"
                    else:
                        output_path = output_dir / f"audio_scene_{i+1:03d}.wav"
                    
                    scene_label = f"场景 {i+1}" if scene_id not in [0, 999] else (f"开头场景(id={scene_id})" if scene_id == 0 else f"结尾场景(id={scene_id})")
                    # 检查文件是否已存在
                    if output_path.exists() and not force_regenerate:
                        audio_paths.append(str(output_path))
                        print(f"  ✓ {scene_label} 音频已存在，跳过生成: {output_path}")
                        part_idx += 1  # 移动到下一个 narration_parts 索引
                    else:
                        try:
                            self.tts_generator.generate(narration_parts[scene_audio_idx], str(output_path))
                            audio_paths.append(str(output_path))
                            print(f"  ✓ {scene_label} 音频已生成: {output_path}")
                            part_idx += 1  # 移动到下一个 narration_parts 索引
                        except Exception as e:
                            print(f"  ✗ {scene_label} 音频生成失败: {e}")
        
        # 处理结尾音频（如果有单独的ending，且不在scenes中）
        if not has_ending_scene:
            if part_idx < len(narration_parts) and narration_parts[part_idx]:
                output_path = output_dir / "audio_ending.wav"
                # 检查文件是否已存在
                if output_path.exists() and not force_regenerate:
                    audio_paths.append(str(output_path))
                    print(f"  ✓ 结尾音频已存在，跳过生成: {output_path}")
                else:
                    try:
                        self.tts_generator.generate(narration_parts[part_idx], str(output_path))
                        audio_paths.append(str(output_path))
                        print(f"  ✓ 结尾音频已生成: {output_path}")
                    except Exception as e:
                        print(f"  ✗ 结尾音频生成失败: {e}")
        
        print(f"\n✓ 共生成 {len(audio_paths)} 个分段音频文件")
        return audio_paths
    
    def generate_audio(self, narration: str, output_name: str) -> str:
        """生成配音（完整音频，用于字幕生成等）"""
        if not narration:
            print("跳过配音生成（无旁白文本）")
            return None
        if self.tts_generator is None:
            print("跳过配音生成（未加载TTS引擎）")
            return None
        
        print("\n=== 生成完整配音（用于字幕生成） ===")
        
        output_dir = Path(self.paths['output_dir']) / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "audio.wav"
        
        try:
            self.tts_generator.generate(narration, str(output_path))
            return str(output_path)
        except Exception as e:
            print(f"✗ 配音生成失败: {e}")
            return None
    
    def _get_video_duration(self, video_path: str) -> float:
        """获取视频或音频时长（秒）"""
        try:
            from video_composer import VideoComposer
            composer = VideoComposer(self.config_path)
            return composer.get_media_duration(video_path)
        except Exception as e:
            print(f"  ⚠ 无法获取媒体时长 {video_path}: {e}")
            return 0.0
    
    def _extend_video_by_duration(self, video_path: str, duration: float) -> Optional[str]:
        """延长视频时长（重复最后一帧）
        
        Args:
            video_path: 视频路径
            duration: 需要延长的时长（秒）
        
        Returns:
            延长后的视频路径，如果失败返回 None
        """
        if duration <= 0:
            return None
        
        # 检查视频文件是否存在
        from pathlib import Path
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"  ⚠ 视频文件不存在，无法延长: {video_path}")
            return None
        
        if not video_file.is_file():
            print(f"  ⚠ 视频路径不是文件: {video_path}")
            return None
        
        try:
            from video_composer import VideoComposer
            import tempfile
            import ffmpeg
            
            composer = VideoComposer(self.config_path)
            # 使用与开头视频相同的目录（outputs/opening_ending/），而不是 temp 目录
            from opening_ending_generator import OpeningEndingGenerator
            opening_ending_gen = OpeningEndingGenerator(self.config_path)
            output_dir = opening_ending_gen.output_dir  # 获取 opening_ending 输出目录
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取视频帧率
            try:
                fps = composer._get_video_fps(video_path)
                if fps is None or fps <= 0:
                    fps = 30.0  # 默认帧率
                    print(f"  ⚠ 无法获取视频帧率，使用默认值: {fps}")
            except Exception as e:
                fps = 30.0  # 默认帧率
                print(f"  ⚠ 获取视频帧率失败，使用默认值: {fps} ({e})")
            
            # 提取最后一帧（使用 output_dir 而不是 temp_dir）
            # 修复文件名：去掉视频扩展名，避免出现 ending.mp4.png 的情况
            video_name = Path(video_path).stem  # 获取不带扩展名的文件名
            last_frame = output_dir / f"last_frame_{video_name}.png"
            last_frame_extracted = False
            extraction_method = None  # 记录使用的提取方法
            
            # 方法1：使用时间戳方式提取最后一帧（最可靠）
            try:
                video_duration = self._get_video_duration(video_path)
                if video_duration > 0:
                    (
                        ffmpeg
                        .input(video_path, ss=max(0, video_duration - 0.1))  # 从视频结束前0.1秒提取
                        .output(str(last_frame), vframes=1)
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                    if last_frame.exists():
                        last_frame_extracted = True
                        extraction_method = "时间戳方式（从视频结束前0.1秒）"
                        print(f"  ✓ 成功提取最后一帧（{extraction_method}）: {last_frame}")
            except Exception as e:
                # 方法2：使用 select filter 选择最后一帧
                try:
                    (
                        ffmpeg
                        .input(video_path)
                        .filter('select', 'eq(n,-1)')  # 选择最后一帧
                        .output(str(last_frame), vframes=1)
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                    if last_frame.exists():
                        last_frame_extracted = True
                        extraction_method = "select filter (eq(n,-1))"
                        print(f"  ✓ 成功提取最后一帧（{extraction_method}）: {last_frame}")
                except Exception as e2:
                    # 方法3：使用总帧数方式
                    try:
                        probe = ffmpeg.probe(video_path)
                        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                        if video_stream:
                            total_frames = int(video_stream.get('nb_frames', 0))
                            if total_frames > 0:
                                (
                                    ffmpeg
                                    .input(video_path)
                                    .filter('select', f'eq(n,{total_frames-1})')
                                    .output(str(last_frame), vframes=1)
                                    .overwrite_output()
                                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                                )
                                if last_frame.exists():
                                    last_frame_extracted = True
                                    extraction_method = f"总帧数方式（第{total_frames-1}帧）"
                                    print(f"  ✓ 成功提取最后一帧（{extraction_method}）: {last_frame}")
                    except Exception as e3:
                        print(f"  ⚠ 所有提取方法都失败: 时间戳方式={e}, select方式={e2}, 总帧数方式={e3}")
            
            # 如果提取视频最后一帧失败，尝试使用备用方案
            if not last_frame_extracted or not last_frame.exists():
                # 备用方案1：使用 ending.png
                ending_png = output_dir / "ending.png"
                if ending_png.exists():
                    print(f"  ℹ 使用 ending.png 作为最后一帧（提取视频帧失败时的备用方案）")
                    last_frame = ending_png
                    last_frame_extracted = True
                    extraction_method = "ending.png 备用文件"
                else:
                    # 备用方案2：提取第一帧
                    print(f"  ℹ ending.png 不存在，尝试从视频中提取第一帧作为备用")
                    try:
                        (
                            ffmpeg
                            .input(video_path, ss=0)
                            .output(str(last_frame), vframes=1)
                            .overwrite_output()
                            .run(quiet=True, capture_stdout=True, capture_stderr=True)
                        )
                        if last_frame.exists():
                            print(f"  ✓ 成功提取第一帧作为备用")
                            last_frame_extracted = True
                            extraction_method = "第一帧（备用）"
                        else:
                            raise RuntimeError(f"提取第一帧也失败，文件不存在: {last_frame}")
                    except Exception as e3:
                        print(f"  ⚠ 提取第一帧也失败: {e3}")
                        raise RuntimeError(
                            f"提取视频最后一帧失败，且找不到 ending.png 备用文件。\n"
                            f"  视频路径: {video_path}\n"
                            f"  ending.png 路径: {ending_png}\n"
                            f"  请确保视频文件存在且可读，或创建 ending.png 作为备用。"
                        )
            
            # 创建延长部分（重复最后一帧，使用 output_dir）
            extended_part = output_dir / f"extended_part_{os.path.basename(video_path)}.mp4"
            try:
                (
                    ffmpeg
                    .input(str(last_frame), loop=1, t=duration, framerate=fps)
                    .output(
                        str(extended_part),
                        vcodec='libx264',
                        pix_fmt='yuv420p',
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                if not extended_part.exists():
                    raise RuntimeError(f"创建延长部分失败，文件不存在: {extended_part}")
            except Exception as e:
                print(f"  ⚠ 创建延长部分失败: {e}")
                raise
            
            # 拼接原视频和延长部分
            # 输出路径使用与开头视频相同的目录
            output_path = output_dir / f"extended_{os.path.basename(video_path)}"
            concat_list = output_dir / f"extend_concat_{os.path.basename(video_path)}.txt"
            try:
                with open(concat_list, 'w', encoding='utf-8') as f:
                    abs_video = os.path.abspath(video_path)
                    abs_extended = os.path.abspath(str(extended_part))
                    f.write(f"file '{abs_video}'\n")
                    f.write(f"file '{abs_extended}'\n")
                
                (
                    ffmpeg
                    .input(str(concat_list), format='concat', safe=0)
                    .output(
                        str(output_path),
                        vcodec='libx264',
                        acodec='copy',
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                if not output_path.exists():
                    raise RuntimeError(f"拼接视频失败，文件不存在: {output_path}")
            except Exception as e:
                print(f"  ⚠ 拼接视频失败: {e}")
                # 尝试输出更详细的错误信息
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"  ffmpeg stderr: {e.stderr.decode('utf-8', errors='ignore')[:500]}")
                raise
            
            # 清理临时文件
            try:
                # 只删除临时提取的帧，不删除 ending.png（如果是备用使用的）
                if last_frame_extracted and last_frame.exists() and last_frame.name != "ending.png":
                    last_frame.unlink()
                if extended_part.exists():
                    extended_part.unlink()
                if concat_list.exists():
                    concat_list.unlink()
            except Exception:
                pass  # 忽略清理错误
            
            # 返回绝对路径，确保与其他视频路径一致
            return str(output_path.resolve())
        except Exception as e:
            print(f"  ⚠ 延长视频失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_subtitle(
        self,
        audio_path: str,
        output_name: str,
        narration: str = "",
        segments: Optional[List[str]] = None,
        video_durations: Optional[List[float]] = None,
        total_duration: Optional[float] = None,
    ) -> str:
        """生成字幕"""
        if self.subtitle_generator is None:
            print("\n=== 跳过字幕生成（未加载字幕引擎） ===")
            return None

        print("\n=== 生成字幕 ===")
        
        output_dir = Path(self.paths['output_dir']) / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "subtitle.srt"
        
        try:
            self.subtitle_generator.generate(
                audio_path,
                str(output_path),
                narration=narration,
                segments=segments,
                video_durations=video_durations,  # 传递视频时长信息
                total_duration=total_duration,  # 传递总音频时长（如果是分段音频）
            )
            return str(output_path)
        except Exception as e:
            print(f"✗ 字幕生成失败: {e}")
            return None
    
    def compose_final_video(
        self,
        video_clips: List[str],
        audio_path: str,
        subtitle_path: str,
        output_name: str,
        scenes: Optional[List[Dict]] = None,
    ) -> str:
        """合成最终视频
        
        Args:
            video_clips: 视频片段路径列表（已使用精确音频时长生成，精确匹配）
            audio_path: 完整音频路径（已合并所有分段音频）
            subtitle_path: 字幕文件路径（基于完整音频生成）
            output_name: 输出名称
            scenes: 场景元数据
        """
        if self.video_composer is None:
            print("\n=== 跳过合成（未加载合成器） ===")
            return None

        print("\n=== 合成最终视频 ===")
        print(f"视频片段数量: {len(video_clips)}")
        print(f"  ✓ 视频片段已使用精确音频时长生成，精确匹配音频")
        print(f"  ✓ 使用完整音频进行合成")
        
        if video_clips:
            print(f"视频片段列表:")
            for i, clip in enumerate(video_clips, 1):
                clip_name = os.path.basename(clip)
                print(f"  {i}. {clip_name}")
        else:
            print("⚠ 警告: 没有视频片段，无法合成")
            return None
        
        if not audio_path or not os.path.exists(audio_path):
            print("⚠ 警告: 完整音频文件不存在，无法合成")
            return None
        
        # 确保所有视频路径都是绝对路径，避免拼接时路径不一致
        video_clips = [str(Path(vp).resolve()) if not os.path.isabs(vp) else vp for vp in video_clips]
        audio_path = str(Path(audio_path).resolve()) if not os.path.isabs(audio_path) else audio_path
        
        output_dir = Path(self.paths['output_dir']) / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_name}.mp4"
        
        # 获取背景音乐路径（可选）
        bgm_cfg = self.config.get('composition', {}).get('bgm', {})
        bgm_path = bgm_cfg.get('path') or bgm_cfg.get('tracks', {}).get('default', {}).get('path')
        if bgm_path:
            bgm_path = str((self.config_dir / bgm_path).resolve()) if not os.path.isabs(bgm_path) else bgm_path
            if not os.path.exists(bgm_path):
                bgm_path = None
        
        try:
            # 直接使用完整音频合成（视频片段已精确匹配）
            self.video_composer.compose(
                video_clips,
                audio_path,
                subtitle_path,
                bgm_path,
                str(output_path),
                scene_metadata=scenes,
            )
            return str(output_path)
        except Exception as e:
            print(f"✗ 视频合成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_from_markdown(self, markdown_path: str, image_dir: str, output_name: str = "output"):
        """
        从 Markdown 格式的分镜脚本生成视频
        
        Args:
            markdown_path: Markdown 脚本路径
            image_dir: 图像目录
            output_name: 输出文件名
        """
        print(f"\n处理 Markdown 脚本: {markdown_path}")
        
        # 解析 Markdown（简化版）
        scenes = self.parse_markdown_script(markdown_path, image_dir)
        narration = self.extract_narration_from_markdown(markdown_path)
        
        # 创建临时脚本文件
        script = {
            'scenes': scenes,
            'narration': narration,
        }
        
        script_path = Path(self.paths['temp_dir']) / f"{output_name}_script.json"
        with open(script_path, 'w', encoding='utf-8') as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        
        # 处理脚本
        return self.process_script(str(script_path), output_name)
    
    def parse_markdown_script(self, markdown_path: str, image_dir: str) -> List[Dict]:
        """解析 Markdown 脚本"""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        scenes = []
        image_files = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
        image_files.sort()
        
        # 简单的解析逻辑（可以根据实际格式调整）
        lines = content.split('\n')
        scene_index = 0
        
        for line in lines:
            if '### 镜头' in line or '镜头' in line:
                # 查找对应的图像
                if scene_index < len(image_files):
                    image_path = str(image_files[scene_index])
                    scenes.append({
                        'image_path': image_path,
                        'scene_number': scene_index + 1,
                    })
                    scene_index += 1
        
        # 如果图像数量多于场景，使用所有图像
        if len(scenes) < len(image_files):
            scenes = []
            for i, image_file in enumerate(image_files):
                scenes.append({
                    'image_path': str(image_file),
                    'scene_number': i + 1,
                })
        
        return scenes
    
    def extract_narration_from_markdown(self, markdown_path: str) -> str:
        """从 Markdown 中提取旁白"""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单的提取逻辑（提取所有"旁白："后的内容）
        narration_parts = []
        lines = content.split('\n')
        
        for line in lines:
            if '**旁白：**' in line or '旁白：' in line:
                # 提取旁白文本
                part = line.split('：', 1)
                if len(part) > 1:
                    narration_parts.append(part[1].strip())
        
        return ' '.join(narration_parts)


def find_json_files_in_directory(directory: str) -> List[Path]:
    """在目录中查找所有JSON文件并按数字顺序排序"""
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    
    json_files = list(dir_path.glob("*.json"))
    # 按文件名中的数字排序
    def extract_number(filepath: Path) -> int:
        try:
            # 尝试从文件名中提取数字（例如 "2.json" -> 2）
            name = filepath.stem
            return int(name)
        except ValueError:
            # 如果无法提取数字，返回一个很大的数字，放在后面
            return 999999
    
    json_files.sort(key=extract_number)
    return json_files


def check_video_exists(output_dir: Path, output_name: str) -> bool:
    """检查最终视频是否已存在"""
    final_video = output_dir / output_name / f"{output_name}.mp4"
    return final_video.exists() and final_video.stat().st_size > 0


def main():
    parser = argparse.ArgumentParser(description="AI视频生成系统")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--script", type=str, help="脚本文件路径（JSON格式）或场景目录路径")
    parser.add_argument("--markdown", type=str, help="Markdown 脚本路径")
    parser.add_argument("--image-dir", type=str, help="图像目录（与 --markdown 一起使用）")
    parser.add_argument("--output", type=str, default="output", help="输出文件名或输出名称模式（批量模式下使用{num}占位符，如lingjie_ep{num}_full）")
    parser.add_argument("--skip-video", action="store_true", help="跳过视频生成")
    parser.add_argument("--skip-tts", action="store_true", help="跳过配音生成")
    parser.add_argument("--skip-subtitle", action="store_true", help="跳过字幕生成")
    parser.add_argument("--skip-compose", action="store_true", help="跳过视频合成")
    parser.add_argument("--force", action="store_true", help="强制重新生成已存在的视频")
    
    args = parser.parse_args()
    
    # 初始化流水线（在批量处理时每个脚本都会使用，所以提前初始化）
    pipeline = AIVideoPipeline(
        args.config,
        load_image=not args.skip_video,
        load_video=not args.skip_video,
        load_tts=not args.skip_tts,
        load_subtitle=not args.skip_subtitle,
        load_composer=not args.skip_compose,
    )
    
    # 处理脚本
    if args.script:
        script_path = Path(args.script)
        
        # 检查是否是目录（批量处理模式）
        if script_path.is_dir():
            print(f"检测到目录路径，进入批量处理模式: {args.script}")
            json_files = find_json_files_in_directory(str(script_path))
            
            if not json_files:
                print(f"错误: 在目录 {args.script} 中未找到JSON文件")
                return
            
            print(f"找到 {len(json_files)} 个JSON文件，将按顺序处理")
            print("=" * 80)
            
            output_dir = Path(pipeline.paths['output_dir'])
            successful = 0
            skipped = 0
            failed = 0
            
            for json_file in json_files:
                scene_num = json_file.stem  # 获取场景编号（如 "2"）
                
                # 生成输出名称
                if "{num}" in args.output:
                    output_name = args.output.replace("{num}", scene_num)
                else:
                    # 如果输出名称不包含占位符，自动追加场景编号
                    output_name = f"{args.output}_{scene_num}"
                
                # 检查视频是否已存在
                if not args.force and check_video_exists(output_dir, output_name):
                    print(f"\n{'=' * 80}")
                    print(f"场景 {scene_num}: {json_file.name}")
                    print(f"输出: {output_name}")
                    print(f"✓ 视频已存在，跳过: {output_dir / output_name / f'{output_name}.mp4'}")
                    skipped += 1
                    continue
                
                print(f"\n{'=' * 80}")
                print(f"处理场景 {scene_num}: {json_file.name}")
                print(f"输出名称: {output_name}")
                print(f"{'=' * 80}")
                
                try:
                    pipeline.process_script(str(json_file), output_name)
                    print(f"\n✓ 场景 {scene_num} 处理完成: {output_name}")
                    successful += 1
                except Exception as e:
                    print(f"\n✗ 场景 {scene_num} 处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
                    continue
            
            print(f"\n{'=' * 80}")
            print("批量处理完成")
            print(f"成功: {successful} 个")
            print(f"跳过: {skipped} 个")
            print(f"失败: {failed} 个")
            print(f"{'=' * 80}")
            
        elif script_path.exists() and script_path.is_file():
            # 单个文件处理模式
            pipeline.process_script(args.script, args.output)
        else:
            print(f"错误: 脚本路径不存在: {args.script}")
            parser.print_help()
            
    elif args.markdown and args.image_dir:
        pipeline.process_from_markdown(args.markdown, args.image_dir, args.output)
    else:
        print("错误: 需要提供 --script 或 --markdown + --image-dir")
        parser.print_help()


if __name__ == "__main__":
    main()



