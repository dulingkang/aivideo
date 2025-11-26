#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动分析素材帧，计算清晰度/亮度等指标并生成自动标签。

示例：
    python auto_tag_stills.py \
        --metadata ../assets/library/metadata.csv \
        --output ../assets/library/metadata.csv \
        --selected-dir ../assets/library/selected \
        --per-episode 60
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "gen_video" / "assets" / "library" / "metadata.csv"
DEFAULT_OUTPUT = DEFAULT_METADATA


@dataclass
class Metrics:
    episode: str
    frame_path: str
    timestamp_sec: float
    source_video: str
    brightness: float
    saturation: float
    sharpness: float
    hue_deg: float
    auto_tags: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为素材帧生成分析指标与自动标签")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="素材元数据 CSV（默认: assets/library/metadata.csv）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 CSV 路径（默认覆盖原文件）",
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=None,
        help="可选：筛选的优质素材复制到此目录（按剧集划分）",
    )
    parser.add_argument(
        "--per-episode",
        type=int,
        default=0,
        help="如设置 >0，则按清晰度挑选每集前 N 张复制到 selected-dir",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=10.0,
        help="判定为清晰帧的 Laplacian 方差阈值（默认 10.0）",
    )
    parser.add_argument(
        "--min-saturation",
        type=float,
        default=0.18,
        help="判定色彩鲜艳的饱和度阈值（默认 0.18）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    return parser.parse_args()


def load_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def compute_metrics(img: Image.Image) -> Dict[str, float]:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("期望 RGB 图像")

    # 亮度：灰度平均
    gray = arr.mean(axis=2)
    brightness = float(gray.mean())

    # 饱和度：HSV 中 S 分量的平均
    hsv = rgb_to_hsv(arr)
    saturation = float(hsv[:, :, 1].mean())

    # Hue（平均色相），转换到 0-360
    hue_channel = hsv[:, :, 0]
    # 处理环绕问题：将 hue 转换到 0-2π，使用单位向量平均
    hue_rad = hue_channel * 2.0 * math.pi
    mean_x = float(np.cos(hue_rad).mean())
    mean_y = float(np.sin(hue_rad).mean())
    hue_deg = (math.degrees(math.atan2(mean_y, mean_x)) + 360.0) % 360.0

    # 清晰度：自定义 Laplacian 方差
    sharpness = float(laplacian_var(gray))

    return {
        "brightness": brightness,
        "saturation": saturation,
        "sharpness": sharpness,
        "hue_deg": hue_deg,
    }


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    # arr: (..., 3) in range [0,1]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    v = maxc
    delta = maxc - minc

    s = np.where(maxc == 0, 0, delta / np.clip(maxc, 1e-8, None))

    # Hue calculation
    rc = np.where(delta != 0, (maxc - r) / delta, 0)
    gc = np.where(delta != 0, (maxc - g) / delta, 0)
    bc = np.where(delta != 0, (maxc - b) / delta, 0)

    h = np.zeros_like(maxc)
    mask = delta != 0
    idx = mask & (maxc == r)
    h[idx] = (bc[idx] - gc[idx]) % 6.0
    idx = mask & (maxc == g)
    h[idx] = 2.0 + (rc[idx] - bc[idx])
    idx = mask & (maxc == b)
    h[idx] = 4.0 + (gc[idx] - rc[idx])

    h = (h / 6.0) % 1.0

    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def laplacian_var(gray: np.ndarray) -> float:
    # 自定义 3x3 Laplacian 核
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    result = (
        kernel[1, 1] * padded[1:-1, 1:-1]
        + kernel[0, 1] * padded[0:-2, 1:-1]
        + kernel[2, 1] * padded[2:, 1:-1]
        + kernel[1, 0] * padded[1:-1, 0:-2]
        + kernel[1, 2] * padded[1:-1, 2:]
    )
    return float(result.var())


def generate_tags(
    metrics: Dict[str, float],
    min_sharpness: float,
    min_saturation: float,
) -> List[str]:
    brightness = metrics["brightness"]
    saturation = metrics["saturation"]
    sharpness = metrics["sharpness"]
    hue = metrics["hue_deg"]

    tags: List[str] = []
    if sharpness >= min_sharpness:
        tags.append("sharp")
    elif sharpness <= min_sharpness * 0.4:
        tags.append("blurry")

    if brightness >= 0.65:
        tags.append("bright")
    elif brightness <= 0.35:
        tags.append("dark")

    if saturation >= min_saturation:
        tags.append("vivid")
    else:
        tags.append("muted")

    # 色调分类
    if 20 <= hue <= 65:
        tags.append("warm")
        if "bright" in tags and saturation >= min_saturation:
            tags.append("desert_like")
    elif 80 <= hue <= 160:
        tags.append("greenish")
    elif 180 <= hue <= 260:
        tags.append("blue")
        if "bright" in tags:
            tags.append("sky_like")
    elif hue >= 320 or hue <= 20:
        tags.append("reddish")

    if saturation >= min_saturation * 1.5 and "bright" in tags:
        tags.append("high_contrast")

    return sorted(set(tags))


def load_metadata(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 metadata 文件: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_frames(
    rows: List[Dict[str, str]],
    metadata_dir: Path,
    min_sharpness: float,
    min_saturation: float,
    quiet: bool,
) -> List[Metrics]:
    metrics_list: List[Metrics] = []
    for row in tqdm(rows, desc="分析素材", disable=quiet):
        frame_rel = row.get("frame_path", "")
        frame_path = (metadata_dir / frame_rel).resolve()
        img = load_image(frame_path)
        if img is None:
            if not quiet:
                print(f"⚠ 无法读取图像: {frame_path}")
            continue

        metrics = compute_metrics(img)
        tags = generate_tags(metrics, min_sharpness, min_saturation)

        metrics_list.append(
            Metrics(
                episode=row.get("episode", ""),
                frame_path=frame_rel,
                timestamp_sec=float(row.get("timestamp_sec", 0) or 0),
                source_video=row.get("source_video", ""),
                brightness=metrics["brightness"],
                saturation=metrics["saturation"],
                sharpness=metrics["sharpness"],
                hue_deg=metrics["hue_deg"],
                auto_tags=",".join(tags),
            )
        )
    return metrics_list


def merge_rows(
    original_rows: List[Dict[str, str]],
    metrics: List[Metrics],
) -> List[Dict[str, str]]:
    metrics_map = {m.frame_path: m for m in metrics}
    merged: List[Dict[str, str]] = []
    for row in original_rows:
        frame_rel = row.get("frame_path", "")
        metric = metrics_map.get(frame_rel)
        if not metric:
            merged.append(row)
            continue
        new_row = dict(row)
        new_row.update(
            {
                "brightness": f"{metric.brightness:.4f}",
                "saturation": f"{metric.saturation:.4f}",
                "sharpness": f"{metric.sharpness:.2f}",
                "dominant_hue": f"{metric.hue_deg:.1f}",
                "auto_tags": metric.auto_tags,
            }
        )
        merged.append(new_row)
    return merged


def write_metadata(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_top_frames(
    metrics: List[Metrics],
    target_dir: Path,
    per_episode: int,
    metadata_dir: Path,
    quiet: bool,
) -> None:
    if per_episode <= 0:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    by_episode: Dict[str, List[Metrics]] = {}
    for m in metrics:
        by_episode.setdefault(m.episode, []).append(m)

    for episode, items in by_episode.items():
        items.sort(key=lambda x: (x.sharpness, x.brightness, x.saturation), reverse=True)
        selected = items[:per_episode]
        episode_dir = target_dir / episode
        episode_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for m in selected:
            src = (metadata_dir / m.frame_path).resolve()
            if not src.exists():
                continue
            dst = episode_dir / Path(m.frame_path).name
            shutil.copy2(src, dst)
            copied += 1
        if not quiet:
            print(f"✓ 已拷贝 {copied} 张优选帧 -> {episode_dir}")


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    output_path = args.output.resolve()
    metadata_dir = metadata_path.parent.resolve()

    rows = load_metadata(metadata_path)
    if not rows:
        print("⚠ metadata.csv 为空，未进行处理")
        return

    metrics = analyze_frames(
        rows,
        metadata_dir=metadata_dir,
        min_sharpness=args.min_sharpness,
        min_saturation=args.min_saturation,
        quiet=args.quiet,
    )

    if not metrics:
        print("⚠ 未成功分析任何帧")
        return

    merged_rows = merge_rows(rows, metrics)
    write_metadata(output_path, merged_rows)

    if args.selected_dir:
        select_top_frames(
            metrics,
            target_dir=args.selected_dir.resolve(),
            per_episode=args.per_episode,
            metadata_dir=metadata_dir,
            quiet=args.quiet,
        )

    if not args.quiet:
        print("\n分析完成 ✅")
        print(f"已更新元数据文件: {output_path}")
        print(f"样例前 3 帧: {[m.frame_path for m in metrics[:3]]}")


if __name__ == "__main__":
    main()

