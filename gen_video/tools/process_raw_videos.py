#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 raw_videos 目录批量截帧并生成素材元数据

示例：
    python process_raw_videos.py --interval 1.5 --output-dir ../assets/library/stills --episodes 1 2
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "gen_video" / "raw_videos"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "gen_video" / "assets" / "library" / "stills"
DEFAULT_METADATA_PATH = REPO_ROOT / "gen_video" / "assets" / "library" / "metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量截取 raw_videos 中的素材帧")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="原始视频所在目录（默认: gen_video/raw_videos）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="截帧输出目录（默认: gen_video/assets/library/stills）",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="素材元数据表路径（CSV，默认: gen_video/assets/library/metadata.csv）",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        help="仅处理指定编号或文件名的剧集（例如: 1 2 3 或 episode01）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="截帧时间间隔（秒），默认 1.0 秒抽一帧",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="每个视频最多保留的帧数，默认不限制",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="1280x720",
        help="输出尺寸，格式为 宽x高（默认 1280x720），设置为空字符串表示不缩放",
    )
    parser.add_argument(
        "--suffix",
        choices=("png", "jpg"),
        default="png",
        help="导出图片格式（默认 png）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的帧文件",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    return parser.parse_args()


def list_videos(raw_dir: Path, episodes: Optional[Sequence[str]] = None) -> List[Path]:
    candidates = []
    patterns = ("*.mp4", "*.mov", "*.mkv", "*.ts", "*.flv")
    for pattern in patterns:
        candidates.extend(sorted(raw_dir.glob(pattern)))

    if not episodes:
        return candidates

    normalized = {str(ep).lower() for ep in episodes}
    filtered = []
    for video in candidates:
        stem = video.stem.lower()
        if stem in normalized or any(stem.endswith(f"_{ep}") for ep in normalized):
            filtered.append(video)
        else:
            name_only = "".join(filter(str.isdigit, stem))
            if name_only and name_only in normalized:
                filtered.append(video)
    return filtered


def build_ffmpeg_command(
    video_path: Path,
    output_pattern: Path,
    interval: float,
    scale: Optional[str],
    suffix: str,
    max_frames: Optional[int],
    overwrite: bool,
) -> List[str]:
    vf_parts = []
    if interval > 0:
        vf_parts.append(f"fps=1/{interval}")
    if scale:
        # force_original_aspect_ratio=decrease 保持原始比例，再填充到目标尺寸
        scale_width, scale_height = scale.split("x")
        vf_parts.append(
            f"scale={scale}:force_original_aspect_ratio=decrease,"
            f"pad={scale_width}:{scale_height}:(ow-iw)/2:(oh-ih)/2"
        )
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
    ]
    if vf:
        cmd.extend(["-vf", vf])

    if max_frames:
        cmd.extend(["-vframes", str(max_frames)])

    if suffix == "jpg":
        cmd.extend(["-qscale:v", "2"])

    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")

    cmd.append(str(output_pattern))
    return cmd


def run_ffmpeg(cmd: Sequence[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg 运行失败: {' '.join(cmd)}") from exc


def iter_generated_frames(target_dir: Path, suffix: str) -> Iterable[Path]:
    yield from sorted(target_dir.glob(f"*.{suffix}"))


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_metadata(
    metadata_path: Path,
    records: Iterable[Tuple[str, str, float, str]],
) -> None:
    ensure_directory(metadata_path.parent)
    fieldnames = ["episode", "frame_path", "timestamp_sec", "source_video", "tags", "notes"]
    file_exists = metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for episode, rel_path, timestamp, source in records:
            writer.writerow(
                {
                    "episode": episode,
                    "frame_path": rel_path,
                    "timestamp_sec": f"{timestamp:.2f}",
                    "source_video": source,
                    "tags": "",
                    "notes": "",
                }
            )


def process_video(
    video_path: Path,
    output_dir: Path,
    metadata_path: Path,
    interval: float,
    scale: Optional[str],
    suffix: str,
    max_frames: Optional[int],
    overwrite: bool,
    quiet: bool,
) -> None:
    episode_name = video_path.stem
    target_dir = output_dir / episode_name
    ensure_directory(target_dir)

    output_pattern = target_dir / f"frame_%06d.{suffix}"

    cmd = build_ffmpeg_command(
        video_path,
        output_pattern,
        interval=interval,
        scale=scale,
        suffix=suffix,
        max_frames=max_frames,
        overwrite=overwrite,
    )

    if not quiet:
        print(f"\n=== 处理 {video_path.name} ===")
        print("FFmpeg 命令:", " ".join(cmd))

    run_ffmpeg(cmd)

    frames = list(iter_generated_frames(target_dir, suffix))
    if not frames:
        print(f"⚠ 未生成任何帧: {video_path}")
        return

    records = []
    for idx, frame_path in enumerate(frames, start=1):
        timestamp = max(0.0, (idx - 1) * interval)
        rel_path = os.path.relpath(frame_path, metadata_path.parent)
        records.append((episode_name, rel_path, timestamp, video_path.name))

    write_metadata(metadata_path, records)

    if not quiet:
        print(f"✓ 已生成 {len(frames)} 张素材帧 -> {target_dir}")


def main() -> None:
    args = parse_args()

    raw_dir: Path = args.raw_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    metadata_path: Path = args.metadata.resolve()

    ensure_directory(raw_dir)
    ensure_directory(output_dir)

    videos = list_videos(raw_dir, args.episodes)
    if not videos:
        print(f"⚠ 未在 {raw_dir} 找到待处理的视频")
        return

    for video_path in videos:
        process_video(
            video_path=video_path,
            output_dir=output_dir,
            metadata_path=metadata_path,
            interval=args.interval,
            scale=args.scale or None,
            suffix=args.suffix,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )

    print("\n任务完成 ✅")
    print(f"素材输出目录: {output_dir}")
    print(f"元数据文件: {metadata_path}")


if __name__ == "__main__":
    sys.exit(main())

