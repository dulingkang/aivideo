#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间轴同步管理器
确保视频、音频、字幕完全匹配的完美方案
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess


class TimelineSyncManager:
    """时间轴同步管理器 - 确保视频、音频、字幕完全匹配"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化时间轴同步管理器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path.cwd() / self.config_path).resolve()
        
        # 时间轴数据：存储每个片段的精确时长
        self.timeline_data: List[Dict] = []
    
    def get_media_duration(self, media_path: str) -> float:
        """获取媒体文件的实际时长（秒，精确到毫秒）"""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                 '-of', 'default=noprint_wrappers=1:nokey=1', media_path],
                capture_output=True,
                text=True,
                check=True
            )
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"⚠ 无法获取 {media_path} 的时长: {e}")
            return 0.0
    
    def build_timeline(
        self,
        video_paths: List[str],
        audio_paths: List[str],
        scene_texts: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        构建统一的时间轴
        
        Args:
            video_paths: 视频片段路径列表
            audio_paths: 音频片段路径列表（与video_paths一一对应）
            scene_texts: 场景文本列表（可选，用于字幕）
        
        Returns:
            时间轴数据列表，每个元素包含：
            {
                'index': int,  # 片段索引（从0开始）
                'video_path': str,  # 视频路径
                'audio_path': str,  # 音频路径
                'video_duration': float,  # 视频实际时长（秒）
                'audio_duration': float,  # 音频实际时长（秒）
                'target_duration': float,  # 目标时长（使用音频时长作为基准）
                'start_time': float,  # 在最终视频中的开始时间（秒）
                'end_time': float,  # 在最终视频中的结束时间（秒）
                'text': str,  # 场景文本（如果有）
                'needs_adjustment': bool,  # 是否需要调整
            }
        """
        if len(video_paths) != len(audio_paths):
            raise ValueError(
                f"视频片段数量 ({len(video_paths)}) 与音频片段数量 ({len(audio_paths)}) 不一致"
            )
        
        self.timeline_data = []
        current_time = 0.0
        
        for idx, (video_path, audio_path) in enumerate(zip(video_paths, audio_paths)):
            # 获取实际时长（精确到毫秒）
            video_duration = self.get_media_duration(video_path)
            audio_duration = self.get_media_duration(audio_path)
            
            # 使用音频时长作为基准（因为音频是准确的语音时长）
            # 视频时长应该匹配音频时长
            target_duration = audio_duration
            
            # 检查是否需要调整
            duration_diff = abs(video_duration - audio_duration)
            needs_adjustment = duration_diff > 0.05  # 如果差异超过50ms，需要调整
            
            # 获取场景文本
            text = scene_texts[idx] if scene_texts and idx < len(scene_texts) else ""
            
            timeline_entry = {
                'index': idx,
                'video_path': video_path,
                'audio_path': audio_path,
                'video_duration': video_duration,
                'audio_duration': audio_duration,
                'target_duration': target_duration,  # 使用音频时长作为目标
                'start_time': current_time,
                'end_time': current_time + target_duration,
                'text': text,
                'needs_adjustment': needs_adjustment,
                'duration_diff': duration_diff,
            }
            
            self.timeline_data.append(timeline_entry)
            current_time += target_duration
        
        return self.timeline_data
    
    def adjust_video_durations(self, output_dir: Optional[Path] = None) -> List[str]:
        """
        调整视频时长以匹配音频时长
        
        Args:
            output_dir: 输出目录（用于保存调整后的视频）
        
        Returns:
            调整后的视频路径列表
        """
        if not self.timeline_data:
            raise ValueError("时间轴数据为空，请先调用 build_timeline()")
        
        adjusted_videos = []
        temp_dir = output_dir or Path("/tmp/timeline_sync")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        import ffmpeg
        
        for entry in self.timeline_data:
            video_path = entry['video_path']
            audio_duration = entry['audio_duration']
            video_duration = entry['video_duration']
            target_duration = entry['target_duration']
            
            # 如果时长差异很小（< 50ms），不需要调整
            if not entry['needs_adjustment']:
                adjusted_videos.append(video_path)
                continue
            
            # 生成调整后的视频路径
            video_name = Path(video_path).stem
            adjusted_path = temp_dir / f"adjusted_{entry['index']:03d}_{video_name}.mp4"
            
            try:
                if video_duration < audio_duration:
                    # 视频比音频短，需要延长（重复最后一帧）
                    print(f"  片段 {entry['index']+1}: 视频 ({video_duration:.3f}s) < 音频 ({audio_duration:.3f}s)，延长视频")
                    self._extend_video(video_path, str(adjusted_path), target_duration)
                else:
                    # 视频比音频长，需要裁剪
                    print(f"  片段 {entry['index']+1}: 视频 ({video_duration:.3f}s) > 音频 ({audio_duration:.3f}s)，裁剪视频")
                    self._trim_video(video_path, str(adjusted_path), target_duration)
                
                # 验证调整后的时长
                adjusted_duration = self.get_media_duration(str(adjusted_path))
                if abs(adjusted_duration - target_duration) < 0.05:
                    adjusted_videos.append(str(adjusted_path))
                    print(f"    ✓ 调整成功: {adjusted_duration:.3f}s (目标: {target_duration:.3f}s)")
                else:
                    print(f"    ⚠ 调整后时长仍有差异: {adjusted_duration:.3f}s (目标: {target_duration:.3f}s)")
                    adjusted_videos.append(str(adjusted_path))  # 仍然使用，但记录警告
            except Exception as e:
                print(f"    ✗ 调整失败: {e}，使用原视频")
                adjusted_videos.append(video_path)
        
        return adjusted_videos
    
    def _extend_video(self, video_path: str, output_path: str, target_duration: float):
        """延长视频到目标时长（重复最后一帧）"""
        import ffmpeg
        
        video_duration = self.get_media_duration(video_path)
        duration_diff = target_duration - video_duration
        
        if duration_diff <= 0:
            # 不需要延长，直接复制
            import shutil
            shutil.copy2(video_path, output_path)
            return
        
        # 获取视频帧率
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if video_stream:
                fps_str = video_stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den > 0 else 30.0
                else:
                    fps = float(fps_str) if fps_str else 30.0
            else:
                fps = 30.0
        except:
            fps = 30.0
        
        # 使用 tpad filter 延长视频（重复最后一帧）
        try:
            (
                ffmpeg
                .input(video_path)
                .filter('tpad', stop_mode='clone', stop_duration=duration_diff)
                .output(
                    output_path,
                    vcodec='libx264',
                    acodec='copy',
                    t=target_duration,  # 限制输出时长为目标时长
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except Exception as e:
            # 如果 tpad 不支持，使用替代方法
            print(f"    ⚠ tpad filter 失败，使用替代方法: {e}")
            # 提取最后一帧
            temp_dir = Path(output_path).parent
            last_frame = temp_dir / f"last_frame_{Path(video_path).stem}.png"
            
            (
                ffmpeg
                .input(video_path)
                .filter('select', 'eq(n,-1)')
                .output(str(last_frame), vframes=1)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 创建延长部分
            extended_part = temp_dir / f"extended_{Path(video_path).stem}.mp4"
            (
                ffmpeg
                .input(str(last_frame), loop=1, t=duration_diff, framerate=fps)
                .output(
                    str(extended_part),
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 拼接
            concat_list = temp_dir / f"concat_{Path(video_path).stem}.txt"
            with open(concat_list, 'w') as f:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
                f.write(f"file '{os.path.abspath(str(extended_part))}'\n")
            
            (
                ffmpeg
                .input(str(concat_list), format='concat', safe=0)
                .output(
                    output_path,
                    vcodec='libx264',
                    acodec='copy',
                    t=target_duration,
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 清理临时文件
            for temp_file in [last_frame, extended_part, concat_list]:
                if temp_file.exists():
                    temp_file.unlink()
    
    def _trim_video(self, video_path: str, output_path: str, target_duration: float):
        """裁剪视频到目标时长"""
        import ffmpeg
        
        (
            ffmpeg
            .input(video_path)
            .output(
                output_path,
                vcodec='copy',
                acodec='copy',
                t=target_duration,  # 限制输出时长为目标时长
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    
    def generate_subtitle_timeline(self) -> List[Dict]:
        """
        生成字幕时间轴（基于统一的时间轴数据）
        
        Returns:
            字幕条目列表，每个元素包含：
            {
                'start': float,  # 开始时间（秒）
                'end': float,  # 结束时间（秒）
                'text': str,  # 字幕文本
            }
        """
        if not self.timeline_data:
            raise ValueError("时间轴数据为空，请先调用 build_timeline()")
        
        subtitle_entries = []
        
        for entry in self.timeline_data:
            if entry['text']:
                subtitle_entries.append({
                    'start': entry['start_time'],
                    'end': entry['end_time'],
                    'text': entry['text'],
                })
        
        return subtitle_entries
    
    def save_subtitle_srt(self, output_path: str):
        """保存字幕为SRT格式（基于统一的时间轴）"""
        subtitle_entries = self.generate_subtitle_timeline()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, entry in enumerate(subtitle_entries, 1):
                start_time = self._format_timestamp(entry['start'])
                end_time = self._format_timestamp(entry['end'])
                text = entry['text'].strip()
                
                f.write(f"{idx}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
    
    def _format_timestamp(self, seconds: float) -> str:
        """格式化时间戳为SRT格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def verify_sync(self) -> Tuple[bool, Dict]:
        """
        验证时间轴同步
        
        Returns:
            (是否同步, 诊断信息)
        """
        if not self.timeline_data:
            return False, {"error": "时间轴数据为空"}
        
        total_video_duration = sum(entry['video_duration'] for entry in self.timeline_data)
        total_audio_duration = sum(entry['audio_duration'] for entry in self.timeline_data)
        total_target_duration = sum(entry['target_duration'] for entry in self.timeline_data)
        
        max_diff = max(entry['duration_diff'] for entry in self.timeline_data)
        needs_adjustment_count = sum(1 for entry in self.timeline_data if entry['needs_adjustment'])
        
        diagnostics = {
            'total_video_duration': total_video_duration,
            'total_audio_duration': total_audio_duration,
            'total_target_duration': total_target_duration,
            'max_segment_diff': max_diff,
            'needs_adjustment_count': needs_adjustment_count,
            'segment_count': len(self.timeline_data),
        }
        
        # 检查是否同步（允许50ms的误差）
        is_synced = (
            abs(total_video_duration - total_audio_duration) < 0.05 and
            max_diff < 0.05 and
            needs_adjustment_count == 0
        )
        
        return is_synced, diagnostics
    
    def print_diagnostics(self):
        """打印时间轴诊断信息"""
        if not self.timeline_data:
            print("⚠ 时间轴数据为空")
            return
        
        is_synced, diagnostics = self.verify_sync()
        
        print("\n" + "=" * 80)
        print("时间轴同步诊断")
        print("=" * 80)
        print(f"片段数量: {diagnostics['segment_count']}")
        print(f"视频总时长: {diagnostics['total_video_duration']:.3f}s")
        print(f"音频总时长: {diagnostics['total_audio_duration']:.3f}s")
        print(f"目标总时长: {diagnostics['total_target_duration']:.3f}s")
        print(f"最大片段差异: {diagnostics['max_segment_diff']:.3f}s")
        print(f"需要调整的片段: {diagnostics['needs_adjustment_count']}")
        
        print("\n各片段详情:")
        print(f"{'索引':<6} {'视频时长':<12} {'音频时长':<12} {'目标时长':<12} {'差异':<10} {'状态':<10}")
        print("-" * 80)
        
        for entry in self.timeline_data:
            status = "✓ 同步" if not entry['needs_adjustment'] else "⚠ 需调整"
            print(
                f"{entry['index']+1:<6} "
                f"{entry['video_duration']:<12.3f} "
                f"{entry['audio_duration']:<12.3f} "
                f"{entry['target_duration']:<12.3f} "
                f"{entry['duration_diff']:<10.3f} "
                f"{status:<10}"
            )
        
        print("=" * 80)
        if is_synced:
            print("✓ 时间轴完全同步")
        else:
            print("⚠ 时间轴需要调整")
        print("=" * 80)


def main():
    """测试时间轴同步管理器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="时间轴同步管理器")
    parser.add_argument("--videos", nargs="+", required=True, help="视频片段路径列表")
    parser.add_argument("--audios", nargs="+", required=True, help="音频片段路径列表")
    parser.add_argument("--texts", nargs="+", help="场景文本列表（可选）")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--subtitle", help="输出字幕文件路径")
    
    args = parser.parse_args()
    
    manager = TimelineSyncManager()
    
    # 构建时间轴
    timeline = manager.build_timeline(
        args.videos,
        args.audios,
        scene_texts=args.texts if args.texts else None,
    )
    
    # 打印诊断信息
    manager.print_diagnostics()
    
    # 调整视频时长
    if args.output_dir:
        adjusted_videos = manager.adjust_video_durations(Path(args.output_dir))
        print(f"\n✓ 已调整 {len(adjusted_videos)} 个视频片段")
    
    # 生成字幕
    if args.subtitle:
        manager.save_subtitle_srt(args.subtitle)
        print(f"\n✓ 字幕已保存: {args.subtitle}")


if __name__ == "__main__":
    main()

