#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频合成脚本
将视频片段、音频、字幕合成最终视频
"""

import os
import math
import tempfile
import yaml
import argparse
from pathlib import Path
import ffmpeg
from typing import Dict, List, Optional, Any, Tuple
import subprocess


class VideoComposer:
    """视频合成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化视频合成器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path.cwd() / self.config_path).resolve()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.composition_config = self.config['composition']
        self.subtitle_config = self.config['subtitle']
        self.config_dir = self.config_path.parent
        self._bgm_cache: Dict[str, Any] = {}
        
        # 创建输出目录
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)
    
    def compose(
        self,
        video_paths: List[str],
        audio_path: Optional[str] = None,
        subtitle_path: Optional[str] = None,
        bgm_path: Optional[str] = None,
        output_path: str = "output.mp4",
        scene_metadata: Optional[List[Dict]] = None,
    ) -> str:
        """
        合成视频
        
        Args:
            video_paths: 视频片段路径列表
            audio_path: 配音音频路径
            subtitle_path: 字幕文件路径
            bgm_path: 背景音乐路径
            output_path: 输出视频路径
            
        Returns:
            输出视频路径
        """
        print(f"\n合成视频: {len(video_paths)} 个片段")
        
        # 方法1: 使用 FFmpeg（更快速，适合批量处理）
        if self.composition_config.get('use_ffmpeg', True):
            return self.compose_ffmpeg(
                video_paths,
                audio_path,
                subtitle_path,
                bgm_path,
                output_path,
                scene_metadata=scene_metadata,
            )
        else:
            # 方法2: 使用 MoviePy（更灵活，适合复杂编辑）
            return self.compose_moviepy(
                video_paths,
                audio_path,
                subtitle_path,
                bgm_path,
                output_path,
                scene_metadata=scene_metadata,
            )
    
    def compose_with_segment_audio(
        self,
        video_paths: List[str],
        audio_paths: List[str],
        subtitle_path: Optional[str] = None,
        bgm_path: Optional[str] = None,
        output_path: str = "output.mp4",
        *,
        scene_metadata: Optional[List[Dict]] = None,
        audio_durations: Optional[List[float]] = None,
    ) -> str:
        """使用分段音频合成视频（每个视频片段对应一个音频文件）
        
        Args:
            video_paths: 视频片段路径列表
            audio_paths: 音频片段路径列表（与video_paths一一对应）
            subtitle_path: 字幕文件路径
            bgm_path: 背景音乐路径
            output_path: 输出视频路径
            scene_metadata: 场景元数据
        """
        if len(video_paths) != len(audio_paths):
            raise ValueError(f"视频片段数量 ({len(video_paths)}) 与音频片段数量 ({len(audio_paths)}) 不一致")
        
        print(f"\n使用分段音频合成视频: {len(video_paths)} 个片段")
        temp_dir = Path(self.config['paths']['temp_dir'])
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 为每个视频片段添加对应的音频片段（确保时长对齐）
        print("为每个视频片段添加对应的音频片段...")
        video_with_audio_paths = []
        for i, (video_path, audio_path) in enumerate(zip(video_paths, audio_paths)):
            if not os.path.exists(video_path):
                print(f"  ⚠ 视频文件不存在: {video_path}")
                continue
            if not os.path.exists(audio_path):
                print(f"  ⚠ 音频文件不存在: {audio_path}")
                continue
            
            # 获取视频和音频时长，确保对齐
            video_duration = self.get_media_duration(video_path)
            # 使用精确的音频时长（不取整），确保完全匹配
            audio_duration = self.get_media_duration(audio_path)
            
            # 如果提供了 audio_durations，用于日志显示（但实际使用精确时长）
            if audio_durations and i < len(audio_durations):
                audio_duration_rounded = audio_durations[i]
                print(f"  片段 {i+1}: 音频精确时长 {audio_duration:.3f}s (取整后: {audio_duration_rounded:.0f}s)")
            else:
                print(f"  片段 {i+1}: 音频精确时长 {audio_duration:.3f}s")
            
            temp_video_with_audio = temp_dir / f"temp_video_audio_{i:03d}.mp4"
            try:
                # 为每个视频片段添加对应的音频片段，使用精确的音频时长作为目标
                # 视频时长必须匹配音频时长（精确到毫秒）
                target_duration = audio_duration  # 使用精确的音频时长作为目标
                
                import ffmpeg
                # 创建视频和音频输入流
                video_stream = ffmpeg.input(video_path)
                audio_stream = ffmpeg.input(audio_path)
                # 组合视频和音频，输出到目标文件
                (
                    ffmpeg
                    .output(
                        video_stream,
                        audio_stream,
                        str(temp_video_with_audio),
                        vcodec='copy',
                        acodec='aac',
                        ac=2,  # 立体声
                        ar=48000,  # 采样率
                        t=target_duration,  # 限制时长为较长的一个
                        shortest=None,  # 使用 shortest=False，让两个流都达到目标时长
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                
                if temp_video_with_audio.exists():
                    video_with_audio_paths.append(str(temp_video_with_audio))
                    # 验证实际输出时长
                    actual_duration = self.get_media_duration(str(temp_video_with_audio))
                    duration_diff = abs(actual_duration - target_duration)
                    status = "✓" if duration_diff < 0.05 else "⚠"
                    print(f"  {status} 片段 {i+1}: {os.path.basename(video_path)} ({video_duration:.3f}s) + {os.path.basename(audio_path)} ({audio_duration:.3f}s) -> {actual_duration:.3f}s (目标: {target_duration:.3f}s, 差异: {duration_diff:.3f}s)")
                else:
                    print(f"  ✗ 片段 {i+1} 添加音频失败: 输出文件不存在")
            except Exception as e:
                print(f"  ✗ 片段 {i+1} 添加音频失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not video_with_audio_paths:
            raise RuntimeError("没有成功添加音频的视频片段")
        
        # 2. 拼接所有带音频的视频片段
        temp_concat = temp_dir / "temp_concat_with_audio.mp4"
        print(f"\n拼接 {len(video_with_audio_paths)} 个带音频的视频片段...")
        self.concat_videos_ffmpeg(video_with_audio_paths, str(temp_concat))
        
        # 3. 添加BGM和字幕到最终视频（视频已经有音频了，需要与BGM混合）
        print("添加BGM和字幕到最终视频...")
        bgm_mix_path, bgm_cleanup, bgm_meter = self.prepare_bgm_tracks(
            video_paths,
            scene_metadata=scene_metadata,
            default_bgm_path=bgm_path,
        )
        
        # 添加BGM和字幕（视频已经有音频了，需要从视频中提取音频，与BGM混合）
        # 直接使用 add_audio_subtitle_ffmpeg，传入视频路径，它会从视频中提取音频
        # 但我们需要传入 None 作为单独的音频路径，然后让方法从视频中提取音频与BGM混合
        self.add_audio_subtitle_ffmpeg(
            str(temp_concat),
            None,  # 不使用单独的音频文件（视频中已有音频）
            subtitle_path,
            bgm_mix_path,
            output_path,
            bgm_pre_scaled=bool(bgm_meter.get("pre_scaled") if bgm_meter else False),
        )
        
        # 4. Real-ESRGAN 超分后处理
        post_cfg = self.composition_config.get("postprocess", {})
        if post_cfg.get("enabled"):
            try:
                output_path = self.postprocess_with_realesrgan(output_path, post_cfg)
            except Exception as exc:
                print(f"⚠ 视频后处理失败: {exc}")
        
        # 清理临时文件
        try:
            for path in video_with_audio_paths:
                if os.path.exists(path):
                    os.unlink(path)
            if temp_concat.exists():
                temp_concat.unlink()
        except Exception:
            pass
        
        return output_path
    
    def compose_ffmpeg(
        self,
        video_paths: List[str],
        audio_path: Optional[str] = None,
        subtitle_path: Optional[str] = None,
        bgm_path: Optional[str] = None,
        output_path: str = "output.mp4",
        *,
        scene_metadata: Optional[List[Dict]] = None,
    ) -> str:
        """使用 FFmpeg 合成视频"""
        temp_dir = Path(self.config['paths']['temp_dir'])
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_video = temp_dir / "temp_concat.mp4"
        
        # 1. 拼接视频片段
        print("拼接视频片段...")
        print(f"  视频片段数量: {len(video_paths)}")
        # 检查是否有重复的视频片段
        seen = set()
        unique_paths = []
        for path in video_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
            else:
                print(f"  ⚠ 警告: 发现重复的视频片段: {os.path.basename(path)}")
        
        if len(unique_paths) != len(video_paths):
            print(f"  ⚠ 已移除 {len(video_paths) - len(unique_paths)} 个重复片段")
            video_paths = unique_paths
        
        self.concat_videos_ffmpeg(video_paths, str(temp_video))
        
        # 2. 预处理背景音乐
        bgm_mix_path, bgm_cleanup, bgm_meter = self.prepare_bgm_tracks(
            video_paths,
            scene_metadata=scene_metadata,
            default_bgm_path=bgm_path,
        )

        # 3. 添加音频和字幕
        print("添加音频和字幕...")
        self.add_audio_subtitle_ffmpeg(
            str(temp_video),
            audio_path,
            subtitle_path,
            bgm_mix_path,
            output_path,
            bgm_pre_scaled=bool(bgm_meter.get("pre_scaled") if bgm_meter else False),
        )

        # 4. Real-ESRGAN 超分后处理
        post_cfg = self.composition_config.get("postprocess", {})
        if post_cfg.get("enabled"):
            try:
                output_path = self.postprocess_with_realesrgan(output_path, post_cfg)
            except Exception as exc:
                print(f"⚠ 视频后处理失败: {exc}")
        
        # 清理临时文件
        if temp_video.exists():
            temp_video.unlink()
        for temp_file in bgm_cleanup:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass
        
        print(f"✓ 视频已合成: {output_path}")
        return output_path
    
    def concat_videos_ffmpeg(self, video_paths: List[str], output_path: str):
        """使用 FFmpeg 拼接视频（确保时长不丢失）"""
        if not video_paths:
            raise ValueError("视频片段列表为空，无法拼接")
        
        # 检查所有视频文件是否存在
        missing_files = []
        video_durations = []
        total_expected_duration = 0.0
        
        for video_path in video_paths:
            abs_path = os.path.abspath(video_path)
            if not os.path.exists(abs_path):
                missing_files.append(abs_path)
                continue
            # 获取每个视频片段的时长
            duration = self.get_media_duration(abs_path)
            video_durations.append((abs_path, duration))
            total_expected_duration += duration
        
        if missing_files:
            raise FileNotFoundError(f"以下视频文件不存在: {missing_files}")
        
        print(f"  预期总时长: {total_expected_duration:.3f}s (共 {len(video_paths)} 个片段)")
        
        print(f"  拼接 {len(video_paths)} 个视频片段...")
        for i, (vp, dur) in enumerate(video_durations, 1):
            print(f"    {i}. {os.path.basename(vp)} ({dur:.3f}s)")
        
        # 直接使用 filter_complex 方式拼接（更可靠，确保时长不丢失）
        # concat demuxer 方式在视频格式不一致时可能丢失时长，filter_complex 更可靠
        print(f"  使用 filter_complex 方式拼接（确保时长不丢失）...")
        try:
            self._concat_videos_with_filter_complex(video_paths, output_path, total_expected_duration)
        except Exception as e:
            print(f"✗ filter_complex 拼接失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _concat_videos_with_filter_complex(self, video_paths: List[str], output_path: str, expected_duration: float):
        """使用 filter_complex 方式拼接视频（更可靠，确保时长不丢失）"""
        print(f"  使用 filter_complex 方式拼接（确保时长不丢失）...")
        
        # 检查哪些视频有音频流
        has_audio_list = []
        for video_path in video_paths:
            try:
                probe = ffmpeg.probe(video_path)
                has_audio = any(s.get('codec_type') == 'audio' for s in probe.get('streams', []))
                has_audio_list.append(has_audio)
            except:
                has_audio_list.append(False)
        
        # 构建输入流
        inputs = []
        for video_path in video_paths:
            inputs.append(ffmpeg.input(video_path))
        
        # 使用 concat filter 拼接
        video_streams = [inp['v'] for inp in inputs]
        
        # 拼接视频流
        if len(video_streams) > 1:
            video_concat = ffmpeg.concat(*video_streams, v=1, a=0)
        else:
            video_concat = video_streams[0]
        
        # 拼接音频流（如果有）
        if any(has_audio_list):
            audio_streams = []
            for i, inp in enumerate(inputs):
                if has_audio_list[i]:
                    try:
                        audio_streams.append(inp['a'])
                    except:
                        pass  # 如果获取音频流失败，跳过
            
            if len(audio_streams) > 1:
                audio_concat = ffmpeg.concat(*audio_streams, v=0, a=1)
            elif len(audio_streams) == 1:
                audio_concat = audio_streams[0]
            else:
                audio_concat = None
        else:
            audio_concat = None
        
        # 输出
        # 获取编码参数
        video_codec = self.composition_config.get('video_codec', 'libx264')
        video_bitrate = self.composition_config.get('video_bitrate', '8000k')
        video_preset = self.composition_config.get('video_preset', 'medium')
        video_crf = self.composition_config.get('video_crf')
        
        # 构建输出参数
        output_kwargs = {'vcodec': video_codec}
        if video_crf is not None and video_codec == 'libx264':
            # 使用 CRF 质量模式
            output_kwargs['crf'] = str(video_crf)
            if video_preset:
                output_kwargs['preset'] = video_preset
            print(f"  使用 CRF 质量模式: {video_crf}, preset: {video_preset}")
        else:
            # 使用比特率模式
            output_kwargs['b:v'] = video_bitrate
            if video_preset and video_codec == 'libx264':
                output_kwargs['preset'] = video_preset
            print(f"  使用比特率模式: {video_bitrate}, preset: {video_preset if video_preset else 'default'}")
        
        if audio_concat is not None:
            out = ffmpeg.output(video_concat, audio_concat, output_path, 
                              acodec='aac',
                              **output_kwargs)
        else:
            out = ffmpeg.output(video_concat, output_path,
                              **output_kwargs)
        
        out.overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True)
        
        # 验证时长
        actual_duration = self.get_media_duration(output_path)
        duration_diff = abs(actual_duration - expected_duration)
        if duration_diff < 0.1:
            print(f"  ✓ filter_complex 拼接成功: 实际时长 {actual_duration:.3f}s (预期: {expected_duration:.3f}s, 差异: {duration_diff:.3f}s)")
        else:
            print(f"  ⚠ filter_complex 拼接完成但时长仍有差异: 实际时长 {actual_duration:.3f}s (预期: {expected_duration:.3f}s, 差异: {duration_diff:.3f}s)")
    
    def get_media_duration(self, media_path: str) -> float:
        """获取媒体文件时长（秒）"""
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', media_path],
                capture_output=True,
                text=True,
                check=True
            )
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"⚠ 无法获取 {media_path} 的时长: {e}")
            return 0.0
    
    def _get_video_fps(self, video_path: str) -> float:
        """获取视频帧率"""
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True,
                text=True,
                check=True
            )
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                return num / den if den > 0 else 25.0
            return float(fps_str) if fps_str else 25.0
        except Exception as e:
            print(f"⚠ 无法获取 {video_path} 的帧率: {e}")
            return 25.0  # 默认帧率
    
    def add_audio_subtitle_ffmpeg(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        subtitle_path: Optional[str] = None,
        bgm_path: Optional[str] = None,
        output_path: str = "output.mp4",
        *,
        bgm_pre_scaled: bool = False,
    ):
        """使用 FFmpeg 添加音频和字幕"""
        import subprocess
        
        # 获取视频和音频的实际时长
        video_duration = self.get_media_duration(video_path)
        # 如果 audio_path 为 None，说明视频中已有音频，从视频中提取音频时长
        if audio_path and os.path.exists(audio_path):
            audio_duration = self.get_media_duration(audio_path)
        else:
            # 视频中已有音频，获取视频的音频轨道时长
            audio_duration = video_duration  # 通常视频中的音频时长等于视频时长
        
        # 如果音频比视频长，需要延长视频（重复最后一帧，而不是循环整个视频）
        if audio_path and audio_duration > video_duration and audio_duration > 0:
            duration_diff = audio_duration - video_duration
            print(f"  音频时长 ({audio_duration:.2f}s) 比视频时长 ({video_duration:.2f}s) 长 {duration_diff:.2f}s，将延长视频（重复最后一帧）")
            # 使用 FFmpeg 延长视频：重复最后一帧，而不是循环整个视频
            # 这样可以避免重复开头，只延长结尾
            temp_extended = Path(self.config['paths']['temp_dir']) / "temp_extended_video.mp4"
            temp_extended.parent.mkdir(parents=True, exist_ok=True)
            
            # 方法：使用 filter_complex 重复最后一帧来延长视频
            # 这样可以避免循环整个视频（导致重复开头），只延长结尾
            # 获取编码参数
            video_codec = self.composition_config.get('video_codec', 'libx264')
            video_bitrate = self.composition_config.get('video_bitrate', '8000k')
            video_preset = self.composition_config.get('video_preset', 'medium')
            video_crf = self.composition_config.get('video_crf')
            
            try:
                # 构建输出参数
                output_kwargs = {'vcodec': video_codec, 'acodec': 'copy'}
                if video_crf is not None and video_codec == 'libx264':
                    output_kwargs['crf'] = str(video_crf)
                    if video_preset:
                        output_kwargs['preset'] = video_preset
                else:
                    output_kwargs['b:v'] = video_bitrate
                    if video_preset and video_codec == 'libx264':
                        output_kwargs['preset'] = video_preset
                
                (
                    ffmpeg
                    .input(video_path)
                    .filter('tpad', stop_mode='clone', stop_duration=duration_diff)  # 重复最后一帧
                    .output(
                        str(temp_extended),
                        **output_kwargs,
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
            except Exception as e:
                # 如果 tpad 不支持，使用替代方法：提取最后一帧并重复
                print(f"  ⚠ tpad filter 失败，使用替代方法: {e}")
                # 提取最后一帧
                last_frame = temp_extended.parent / "last_frame.png"
                (
                    ffmpeg
                    .input(video_path)
                    .filter('select', 'eq(n,-1)')  # 选择最后一帧
                    .output(str(last_frame), vframes=1)
                    .overwrite_output()
                    .run(quiet=True)
                )
                # 创建延长视频：原视频 + 重复最后一帧
                fps = self._get_video_fps(video_path)
                num_frames_to_add = int(duration_diff * fps)
                # 获取编码参数（在链式调用之前）
                video_codec = self.composition_config.get('video_codec', 'libx264')
                video_bitrate = self.composition_config.get('video_bitrate', '8000k')
                video_preset = self.composition_config.get('video_preset', 'medium')
                video_crf = self.composition_config.get('video_crf')
                
                # 构建输出参数
                output_kwargs = {'vcodec': video_codec}
                if video_crf is not None and video_codec == 'libx264':
                    output_kwargs['crf'] = str(video_crf)
                    if video_preset:
                        output_kwargs['preset'] = video_preset
                else:
                    output_kwargs['b:v'] = video_bitrate
                    if video_preset and video_codec == 'libx264':
                        output_kwargs['preset'] = video_preset
                
                # 使用 loop filter 重复最后一帧
                (
                    ffmpeg
                    .input(str(last_frame), loop=1, t=duration_diff, framerate=fps)
                    .output(
                        str(temp_extended.parent / "extended_part.mp4"),
                        **output_kwargs,
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
                # 拼接原视频和延长部分
                concat_list = temp_extended.parent / "extend_concat.txt"
                with open(concat_list, 'w') as f:
                    f.write(f"file '{video_path}'\n")
                    f.write(f"file '{temp_extended.parent / 'extended_part.mp4'}'\n")
                (
                    ffmpeg
                    .input(str(concat_list), format='concat', safe=0)
                    .output(
                        str(temp_extended),
                        vcodec=self.composition_config['video_codec'],
                        acodec='copy',
                        **{'b:v': self.composition_config['video_bitrate']},
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
            video_path = str(temp_extended)
            print(f"  ✓ 视频已延长至 {audio_duration:.2f}s（重复最后一帧）")
        elif video_duration > audio_duration and audio_duration > 0:
            duration_diff = video_duration - audio_duration
            # 视频时长比音频时长长，通常是因为向上取整导致的累计误差
            # 由于音频是准确的语音时长，应该裁剪视频到音频时长，而不是延长音频
            # 这样可以确保视频时长与语音完全对应，避免末尾出现静音
            print(f"  视频时长 ({video_duration:.2f}s) 比音频时长 ({audio_duration:.2f}s) 长 {duration_diff:.2f}s")
            print(f"  ℹ 将视频裁剪到音频时长（音频是准确的语音时长，向上取整导致的累计误差）")
            # 裁剪视频到音频时长：使用 -t 参数限制输出时长
            temp_cropped_video = self.config['paths']['temp_dir'] + "/temp_cropped_video.mp4"
            (
                ffmpeg
                .input(video_path)
                .output(
                    temp_cropped_video,
                    vcodec='copy',  # 使用 copy 避免重新编码
                    acodec='copy',
                    t=audio_duration,  # 限制输出时长为音频时长
                )
                .overwrite_output()
                .run(quiet=True)
            )
            video_path = temp_cropped_video
            video_duration = audio_duration  # 更新视频时长为音频时长
            print(f"  ✓ 视频已裁剪至 {audio_duration:.2f}s（与音频时长一致）")
        
        # 构建输入列表
        inputs = []
        input_index = 0
        
        # 视频输入
        inputs.append(('-i', video_path))
        video_index = input_index
        input_index += 1
        
        # 音频输入
        audio_index = None
        if audio_path and os.path.exists(audio_path):
            inputs.append(('-i', audio_path))
            audio_index = input_index
            input_index += 1
        
        # 背景音乐输入
        bgm_index = None
        if bgm_path and os.path.exists(bgm_path):
            inputs.append(('-i', bgm_path))
            bgm_index = input_index
            input_index += 1
        
        # 构建基础命令
        cmd = ['ffmpeg', '-y']
        for flag, path in inputs:
            cmd.extend([flag, path])
        
        # 构建滤镜
        video_filters = []
        audio_filters = []
        filter_complex_parts = []
        
        # 视频滤镜：分辨率提升 / 锐化 / 字幕
        upscale_cfg = self.composition_config.get("upscale", {})
        if upscale_cfg.get("enabled"):
            up_width = upscale_cfg.get("width")
            up_height = upscale_cfg.get("height")
            if up_width and up_height:
                scale_flags = upscale_cfg.get("flags", "lanczos")
                video_filters.append(f"scale={up_width}:{up_height}:flags={scale_flags}")

        sharpen_cfg = self.composition_config.get("sharpen", {})
        if sharpen_cfg.get("enabled"):
            lx = sharpen_cfg.get("luma_msize_x", 5)
            ly = sharpen_cfg.get("luma_msize_y", 5)
            la = sharpen_cfg.get("luma_amount", 1.0)
            cx = sharpen_cfg.get("chroma_msize_x", 5)
            cy = sharpen_cfg.get("chroma_msize_y", 5)
            ca = sharpen_cfg.get("chroma_amount", 0.0)
            video_filters.append(f"unsharp={lx}:{ly}:{la}:{cx}:{cy}:{ca}")

        if subtitle_path and os.path.exists(subtitle_path):
            subtitle_path_escaped = subtitle_path.replace('\\', '\\\\').replace(':', '\\:')
            video_filters.append(f"subtitles='{subtitle_path_escaped}'")
        
        audio_volume = float(self.composition_config.get("audio_volume", 1.0))
        bgm_config = self.composition_config.get('bgm', {})

        # 音频滤镜：混合音频
        # 首先处理配音音频：转换为 48000 Hz 立体声
        target_sample_rate = 48000
        if audio_index is not None:
            # 使用单独的音频文件
            filter_complex_parts.append(
                f"[{audio_index}:a]aresample={target_sample_rate},"
                f"aformat=sample_rates={target_sample_rate}:channel_layouts=stereo,"
                f"volume={audio_volume}[a1_processed]"
            )
            processed_audio_index = "[a1_processed]"
            # 使用单独音频文件的时长
            audio_duration_actual = audio_duration if audio_duration > 0 else (self.get_media_duration(audio_path) if audio_path else video_duration)
        elif video_duration > 0:
            # 从视频中提取音频（视频中已有音频轨道）
            # 视频输入在 video_index，提取其音频流
            filter_complex_parts.append(
                f"[{video_index}:a]aresample={target_sample_rate},"
                f"aformat=sample_rates={target_sample_rate}:channel_layouts=stereo,"
                f"volume={audio_volume}[a1_processed]"
            )
            processed_audio_index = "[a1_processed]"
            # 使用视频时长（视频中的音频时长应该等于视频时长）
            audio_duration_actual = video_duration
        else:
            processed_audio_index = None
            audio_duration_actual = 0.0

        if processed_audio_index is not None and bgm_index is not None:
            # BGM 一直播放，与配音音频时长匹配
            
            if bgm_pre_scaled:
                # 预处理过的 BGM 直接使用，但需要确保采样率匹配
                filter_complex_parts.append(f"[{bgm_index}:a]aresample=48000[a2_raw]")
            else:
                bgm_volume = bgm_config.get('volume', 0.3)
                filter_complex_parts.append(f"[{bgm_index}:a]aresample=48000,volume={bgm_volume}[a2_raw]")
            
            # BGM 循环播放直到达到音频时长
            # 如果BGM时长小于音频时长，循环BGM
            if audio_duration_actual > 0:
                # 使用aloop循环BGM，然后截断到音频时长
                filter_complex_parts.append(
                    f"[a2_raw]aloop=loop=-1:size=2e+09,atrim=0:{audio_duration_actual},asetpts=PTS-STARTPTS[a2_final]"
                )
            else:
                # 如果无法获取音频时长，使用duration=first让BGM跟随音频
                filter_complex_parts.append("[a2_raw]anull[a2_final]")
            
            # 混合配音和BGM
            filter_complex_parts.append(f"[a1_processed][a2_final]amix=inputs=2:duration=first:dropout_transition=2[aout]")
            audio_output = "[aout]"
            print(f"  BGM 配置: 全程播放，音量 {bgm_config.get('volume', 0.3)}")
        elif processed_audio_index is not None:
            audio_output = processed_audio_index
        elif bgm_index is not None:
            if bgm_pre_scaled:
                filter_complex_parts.append(f"[{bgm_index}:a]anull[aout]")
            else:
                bgm_volume = bgm_config.get('volume', 0.3)
                filter_complex_parts.append(f"[{bgm_index}:a]volume={bgm_volume}[aout]")
            audio_output = "[aout]"
        else:
            audio_output = None
        
        # 添加滤镜
        # 如果只有视频滤镜，使用 -vf；如果有音频混合，使用 -filter_complex
        if filter_complex_parts:
            # 有音频混合，使用 filter_complex
            if video_filters:
                # 视频和音频都有滤镜
                video_filter_str = f"[{video_index}:v]{','.join(video_filters)}[vout]"
                filter_complex_parts.insert(0, video_filter_str)
            cmd.extend(['-filter_complex', ';'.join(filter_complex_parts)])
            # 映射将在后面处理
        elif video_filters:
            # 只有视频滤镜
            cmd.extend(['-vf', ','.join(video_filters)])
        
        # 映射流
        # 根据是否有 filter_complex 决定映射方式
        if filter_complex_parts:
            # 使用了 filter_complex
            if video_filters:
                # 视频有滤镜，已在 filter_complex 中处理为 [vout]
                cmd.extend(['-map', '[vout]'])
            else:
                # 视频无滤镜，直接映射
                cmd.extend(['-map', f'{video_index}:v'])
            
            # 音频映射（从 filter_complex 输出）
            if audio_output:
                cmd.extend(['-map', audio_output])
        else:
            # 没有 filter_complex
            cmd.extend(['-map', f'{video_index}:v'])  # 视频映射
            
            if audio_index is not None:
                cmd.extend(['-map', f'{audio_index}:a'])
            elif bgm_index is not None:
                cmd.extend(['-map', f'{bgm_index}:a'])
            else:
                cmd.extend(['-an'])  # 无音频
        
        # 输出参数
        video_codec = self.composition_config.get('video_codec', 'libx264')
        video_bitrate = self.composition_config.get('video_bitrate', '8000k')
        video_preset = self.composition_config.get('video_preset', 'medium')
        video_crf = self.composition_config.get('video_crf')
        
        cmd.extend([
            '-c:v', video_codec,
            '-c:a', self.composition_config['audio_codec'] if audio_output else 'copy',
        ])
        
        # 优先使用 CRF（质量模式），如果没有配置则使用比特率模式
        if video_crf is not None and video_codec == 'libx264':
            cmd.extend(['-crf', str(video_crf)])
            # 如果配置了 preset，使用它
            if video_preset:
                cmd.extend(['-preset', video_preset])
            print(f"  使用 CRF 质量模式: {video_crf}, preset: {video_preset}")
        else:
            cmd.extend(['-b:v', video_bitrate])
            # 如果配置了 preset，使用它
            if video_preset and video_codec == 'libx264':
                cmd.extend(['-preset', video_preset])
            print(f"  使用比特率模式: {video_bitrate}, preset: {video_preset if video_preset else 'default'}")
        
        if audio_output:
            cmd.extend(['-b:a', self.composition_config['audio_bitrate']])
        
        cmd.extend([
            '-s', f"{self.composition_config['output_width']}x{self.composition_config['output_height']}",
            '-shortest',  # 以最短流为准
            output_path
        ])
        
        # 执行命令
        print(f"执行 FFmpeg 命令...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ FFmpeg 执行成功")
        except subprocess.CalledProcessError as e:
            print("✗ FFmpeg 执行失败")
            print(f"命令: {' '.join(cmd)}")
            if e.stderr:
                snippet = e.stderr if len(e.stderr) <= 4000 else e.stderr[:4000] + "..."
                print("错误输出:\n" + snippet)
            raise

    def postprocess_with_realesrgan(self, video_path: str, cfg: dict) -> str:
        """在合成后调用 Real-ESRGAN 进行超分"""
        try:
            from realesrgan_upscale import build_model, upscale_video  # type: ignore
        except ImportError as exc:
            raise RuntimeError("无法导入 realesrgan_upscale，请确认脚本存在并已安装依赖。") from exc

        input_path = Path(video_path)
        model_path = Path(cfg.get("model_path", "models/realesrgan/RealESRGAN_x4plus.pth"))
        scale = int(cfg.get("model_scale", 4))
        outscale = float(cfg.get("outscale", 2.0))
        target_resolution = cfg.get("target_resolution")  # 格式: "1920x1080" 或 None
        tile = int(cfg.get("tile", 0))
        full_precision = bool(cfg.get("full_precision", False))
        codec = cfg.get("codec", "mp4v")
        suffix = cfg.get("suffix", "_upscaled")

        raw_upscaled_path = input_path.with_name(input_path.stem + suffix + "_video" + input_path.suffix)
        final_output_path = input_path.with_name(input_path.stem + suffix + input_path.suffix)
        preserve_audio = cfg.get("preserve_audio", True)
        print("\n=== Real-ESRGAN 视频后处理 ===")
        print(f"输入: {input_path}")
        print(f"输出: {final_output_path}")
        print(f"模型: {model_path.name}, outscale={outscale}x, tile={tile}")
        if target_resolution:
            print(f"目标分辨率: {target_resolution} (超分后将缩放到此分辨率)")

        # 直接使用配置的tile值，不自动调整（让用户手动控制）
        # tile=0 表示不使用瓦片，适合x2模型（计算量小，整图处理更快）
        # tile>0 用于显存不足的情况，但会增加处理时间
        upscaler = build_model(
            model_path=model_path,
            scale=scale,
            half=not full_precision,
            tile=tile,  # 使用配置的tile值，不自动调整
            verbose=False,  # 关闭详细日志，减少输出提高速度
        )
        
        # 获取并行工作线程数（从配置中读取，默认1）
        num_workers = cfg.get("num_workers", 1)
        upscale_video(
            upscaler=upscaler,
            src_path=input_path,
            dst_path=raw_upscaled_path,
            outscale=outscale,
            fps=None,
            codec=codec,
            num_workers=num_workers,
        )

        if preserve_audio:
            try:
                self.copy_audio_track(
                    source_video=input_path,
                    processed_video=raw_upscaled_path,
                    output_video=final_output_path,
                    target_resolution=target_resolution,
                )
                raw_upscaled_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"⚠ 音频合并失败，保留无声版本: {exc}")
                return str(raw_upscaled_path)

        return str(final_output_path)

    def copy_audio_track(self, source_video: Path, processed_video: Path, output_video: Path, target_resolution: Optional[str] = None) -> None:
        """将原始视频的音频轨合并到处理后的视频，并重新编码视频以优化文件大小
        
        Args:
            source_video: 原始视频（用于提取音频）
            processed_video: 处理后的视频（超分后的视频）
            output_video: 输出视频路径
            target_resolution: 目标分辨率，格式 "宽度x高度"，例如 "1920x1080"。如果设置，会先缩放到此分辨率
        """
        # 获取配置参数
        video_codec = self.composition_config.get("video_codec", "libx264")
        video_bitrate = self.composition_config.get("video_bitrate", "8000k")
        video_preset = self.composition_config.get("video_preset", "medium")
        video_crf = self.composition_config.get("video_crf", 23)  # 默认使用 23（平衡点）
        audio_codec = self.composition_config.get("audio_codec", "aac")
        audio_bitrate = self.composition_config.get("audio_bitrate", "192k")
        
        # 构建 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(processed_video),
            "-i",
            str(source_video),
            "-c:v",
            video_codec,
            "-crf",
            str(video_crf),  # 使用配置中的 CRF 值（18-28，越小质量越好文件越大）
            "-preset",
            video_preset,  # 使用配置中的 preset 值（编码速度和质量平衡）
        ]
        
        # 如果指定了目标分辨率，添加缩放滤镜
        if target_resolution:
            try:
                width, height = map(int, target_resolution.split('x'))
                cmd.extend([
                    "-vf",
                    f"scale={width}:{height}:flags=lanczos",  # 使用 lanczos 算法进行高质量缩放
                ])
                print(f"  将视频缩放到目标分辨率: {width}x{height}")
            except ValueError:
                print(f"⚠ 警告: 无效的目标分辨率格式 '{target_resolution}'，应格式为 '宽度x高度'，跳过缩放")
        
        cmd.extend([
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:a",
            audio_codec,
            "-b:a",
            audio_bitrate,
            "-shortest",
            str(output_video),
        ])
        print("合并 Real-ESRGAN 视频与原始音频（重新编码优化文件大小）...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ 音频已保留，视频已重新编码: {output_video}")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr[:500] if exc.stderr else ""
            print(f"✗ 音频合并失败: {stderr}")
            raise
    
    def resolve_path(self, maybe_path: Optional[str]) -> Optional[str]:
        if not maybe_path:
            return None
        path = Path(maybe_path)
        if not path.is_absolute():
            path = (self.config_dir / path).resolve()
        if path.exists():
            return str(path)
        return None

    def get_media_duration(self, media_path: str) -> float:
        try:
            probe = ffmpeg.probe(media_path)
            if probe.get("format", {}).get("duration"):
                return float(probe["format"]["duration"])
            for stream in probe.get("streams", []):
                if stream.get("duration"):
                    return float(stream["duration"])
        except Exception as exc:
            print(f"⚠ 无法获取媒体时长 {media_path}: {exc}")
        return 0.0

    def prepare_bgm_tracks(
        self,
        video_paths: List[str],
        scene_metadata: Optional[List[Dict]] = None,
        default_bgm_path: Optional[str] = None,
    ) -> Tuple[Optional[str], List[str], Dict[str, Any]]:
        """根据场景信息生成背景音乐时间线，返回可用于混音的临时文件"""
        bgm_cfg = self.composition_config.get("bgm", {})
        if not bgm_cfg.get("enabled", False):
            return default_bgm_path, [], {"pre_scaled": False}

        tracks_cfg: Dict[str, Dict[str, Any]] = bgm_cfg.get("tracks", {})
        resolved_default = (
            self.resolve_path(tracks_cfg.get("default", {}).get("path"))
            or self.resolve_path(bgm_cfg.get("path"))
            or self.resolve_path(default_bgm_path)
        )

        if not resolved_default:
            print("⚠ 未找到默认背景音乐文件，使用静音背景。")
            return None, [], {"pre_scaled": False}

        try:
            from pydub import AudioSegment
        except ImportError:
            print("⚠ 未安装 pydub，无法自动混合 BGM，使用默认背景音乐。")
            return resolved_default, [], {"pre_scaled": False}

        clip_durations = [self.get_media_duration(p) for p in video_paths]
        if not any(d > 0 for d in clip_durations):
            return resolved_default, [], {"pre_scaled": False}

        total_duration_ms = int(sum(clip_durations) * 1000)
        if total_duration_ms <= 0:
            return resolved_default, [], {"pre_scaled": False}

        # 预加载并缓存音频，并进行响度标准化
        def load_track(path: str, normalize: bool = True) -> AudioSegment:
            """加载音轨并可选地进行响度标准化"""
            if path not in self._bgm_cache:
                audio = AudioSegment.from_file(path)
                # 对每个BGM文件进行响度标准化，确保所有BGM音量一致
                if normalize:
                    # 计算RMS（均方根）响度
                    raw_audio = audio.get_array_of_samples()
                    if len(raw_audio) > 0:
                        import numpy as np
                        audio_array = np.array(raw_audio, dtype=np.float32)
                        # 归一化到[-1, 1]
                        if audio.sample_width == 1:
                            audio_array = (audio_array - 128) / 128.0
                        elif audio.sample_width == 2:
                            audio_array = audio_array / 32768.0
                        elif audio.sample_width == 4:
                            audio_array = audio_array / 2147483648.0
                        
                        # 计算RMS
                        rms = np.sqrt(np.mean(audio_array ** 2))
                        # 目标RMS（-18dBFS，适合背景音乐）
                        target_rms = 0.125  # 约等于 -18dBFS
                        
                        if rms > 0:
                            # 计算需要的增益
                            gain_factor = target_rms / rms
                            # 限制增益范围，避免过度放大或缩小
                            gain_factor = max(0.1, min(10.0, gain_factor))
                            gain_db = 20 * math.log10(gain_factor)
                            audio = audio + gain_db
                
                self._bgm_cache[path] = audio
            return self._bgm_cache[path]

        result_audio = None
        timeline: List[Tuple[str, Dict[str, Any], int]] = []
        # 跟踪每个音轨的播放位置，确保背景音乐连续播放
        track_positions: Dict[str, int] = {}  # {track_path: current_position_ms}
        
        for idx, duration in enumerate(clip_durations):
            if duration <= 0:
                continue
            scene = scene_metadata[idx] if scene_metadata and idx < len(scene_metadata) else {}
            track_cfg = self.select_bgm_track(
                index=idx,
                scene=scene,
                tracks_cfg=tracks_cfg,
            )
            track_path = self.resolve_path(track_cfg.get("path")) if track_cfg else None
            if not track_path or not Path(track_path).exists():
                track_path = resolved_default
                track_cfg = tracks_cfg.get("default", {}) or {}
            scene_id_dbg = scene.get("id") if isinstance(scene, dict) else None
            label_dbg = scene.get("label") if isinstance(scene, dict) else None
            # print(f"  🎵 场景 {idx+1} (id={scene_id_dbg}, label={label_dbg}) 选择BGM: {Path(track_path).name if track_path else 'None'}")
            timeline.append((track_path, track_cfg or {}, int(duration * 1000)))

        if not timeline:
            return resolved_default, [], {"pre_scaled": False}

        from pydub import AudioSegment  # type: ignore

        master_volume = float(bgm_cfg.get("volume", 0.3))
        global_fade_in = int(bgm_cfg.get("fade_in", 600))
        global_fade_out = int(bgm_cfg.get("fade_out", 600))
        global_crossfade = int(bgm_cfg.get("crossfade", 250))

        for idx, (track_path, track_cfg, duration_ms) in enumerate(timeline):
            try:
                base_audio = load_track(track_path)
            except Exception as exc:
                print(f"⚠ 加载背景音乐失败 {track_path}: {exc}")
                base_audio = AudioSegment.silent(duration=duration_ms)

            if len(base_audio) <= 0:
                segment_audio = AudioSegment.silent(duration=duration_ms)
            else:
                # 获取当前音轨的播放位置（如果之前播放过）
                current_pos = track_positions.get(track_path, 0)
                
                # 如果当前位置已经超过音轨长度，从头开始循环
                if current_pos >= len(base_audio):
                    current_pos = current_pos % len(base_audio)
                
                # 从当前位置开始截取需要的时长
                remaining = len(base_audio) - current_pos
                if remaining >= duration_ms:
                    # 剩余部分足够，直接截取
                    segment_audio = base_audio[current_pos:current_pos + duration_ms]
                    track_positions[track_path] = current_pos + duration_ms
                else:
                    # 剩余部分不足，需要循环
                    segment_parts = [base_audio[current_pos:]]
                    needed = duration_ms - remaining
                    next_pos = 0  # 记录下一个场景应该从哪个位置开始
                    while needed > 0:
                        if needed >= len(base_audio):
                            segment_parts.append(base_audio)
                            needed -= len(base_audio)
                            next_pos = 0  # 完整循环后，下一个场景从头开始
                        else:
                            segment_parts.append(base_audio[:needed])
                            next_pos = needed  # 记录下一个场景应该从 needed 位置开始
                            needed = 0
                    segment_audio = sum(segment_parts)[:duration_ms]
                    track_positions[track_path] = next_pos

            fade_in = int(track_cfg.get("fade_in", global_fade_in))
            fade_out = int(track_cfg.get("fade_out", global_fade_out))
            if fade_in > 0:
                segment_audio = segment_audio.fade_in(min(fade_in, duration_ms // 2))
            if fade_out > 0:
                segment_audio = segment_audio.fade_out(min(fade_out, duration_ms // 2))

            if result_audio is None:
                result_audio = segment_audio
            else:
                crossfade = int(track_cfg.get("crossfade", global_crossfade))
                crossfade = max(0, min(crossfade, min(len(segment_audio), len(result_audio)) // 2))
                result_audio = result_audio.append(segment_audio, crossfade=crossfade)

        if result_audio is None:
            return resolved_default, [], {"pre_scaled": False}

        if len(result_audio) < total_duration_ms:
            pad = total_duration_ms - len(result_audio)
            result_audio += AudioSegment.silent(duration=pad)

        if master_volume <= 0:
            result_audio = result_audio - 90
        else:
            gain_db = 20 * math.log10(master_volume)
            result_audio = result_audio + gain_db

        result_audio = result_audio.set_channels(2)

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix="bgm_mix_", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        result_audio.export(temp_file_path, format="wav")

        return temp_file_path, [temp_file_path], {"pre_scaled": True}

    def select_bgm_track(
        self,
        index: int,
        scene: Dict[str, Any],
        tracks_cfg: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not tracks_cfg:
            return None

        mood = (scene.get("mood") or "").lower()
        title = (scene.get("title") or "").lower()

        if index == 0 and "start" in tracks_cfg:
            return tracks_cfg["start"]

        # 结尾优先：场景ID=999 或 label/type 标记 ending
        scene_id_raw = scene.get("id") if isinstance(scene, dict) else None
        if scene_id_raw is None and isinstance(scene, dict):
            scene_id_raw = scene.get("scene_number")
        scene_id_str = str(scene_id_raw).strip() if scene_id_raw is not None else ""
        try:
            scene_id_val = int(scene_id_str)
        except (ValueError, TypeError):
            scene_id_val = None
        scene_type = (scene.get("type") or "").lower()
        label = (scene.get("label") or "").lower()
        if (
            ("ending" in tracks_cfg)
            and (
                scene_id_val == 999
                or scene_id_str == "999"
                or scene_type == "ending"
                or label == "ending"
            )
        ):
            return tracks_cfg["ending"]

        # 其它自定义标签优先
        if label and label in tracks_cfg:
            return tracks_cfg[label]

        def match_keywords(candidate: Dict[str, Any], text: str) -> bool:
            keywords = candidate.get("match_moods") or candidate.get("match_keywords") or []
            for kw in keywords:
                if kw and kw.lower() in text:
                    return True
            return False

        for key in ["tense", "intense", "battle"]:
            candidate = tracks_cfg.get(key)
            if candidate and (match_keywords(candidate, mood) or match_keywords(candidate, title)):
                return candidate

        for name, candidate in tracks_cfg.items():
            if name in ("default", "start", "ending"):
                continue
            if match_keywords(candidate, mood) or match_keywords(candidate, title):
                return candidate

        return tracks_cfg.get("default")
    
    def compose_moviepy(
        self,
        video_paths: List[str],
        audio_path: Optional[str] = None,
        subtitle_path: Optional[str] = None,
        bgm_path: Optional[str] = None,
        output_path: str = "output.mp4",
        scene_metadata: Optional[List[Dict]] = None,
    ) -> str:
        """使用 MoviePy 合成视频（待实现）"""
        raise NotImplementedError("MoviePy 合成方式待实现")


def main():
    parser = argparse.ArgumentParser(description="视频合成")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--videos", type=str, nargs="+", required=True, help="视频片段路径列表")
    parser.add_argument("--audio", type=str, help="配音音频路径")
    parser.add_argument("--subtitle", type=str, help="字幕文件路径")
    parser.add_argument("--bgm", type=str, help="背景音乐路径")
    parser.add_argument("--output", type=str, default="output.mp4", help="输出视频路径")
    
    args = parser.parse_args()
    
    # 初始化合成器
    composer = VideoComposer(args.config)
    
    # 合成视频
    composer.compose(
        args.videos,
        args.audio,
        args.subtitle,
        args.bgm,
        args.output,
    )


if __name__ == "__main__":
    main()



