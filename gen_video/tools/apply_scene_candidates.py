#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 scene_candidates 目录中的首选素材自动复制到流水线图像目录。

示例：
    python apply_scene_candidates.py \
        --candidates-dir assets/library/scene_candidates/ep5 \
        --target-dir ../lingjie/img2 \
        --ext .png
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="应用场景候选图像到目标目录")
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        required=True,
        help="候选素材根目录（包含 scene_001 等子目录）",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="目标输出目录（如 lingjie/img2）",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".png",
        help="输出文件扩展名（默认 .png）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的目标文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要复制的列表，不真正执行",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    return parser.parse_args()


def find_candidate_image(scene_dir: Path) -> Path | None:
    images: List[Path] = sorted(
        [p for p in scene_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    )
    return images[0] if images else None


def main() -> None:
    args = parse_args()

    if not args.candidates_dir.exists():
        raise FileNotFoundError(f"候选目录不存在: {args.candidates_dir}")

    target_dir = args.target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(
        [p for p in args.candidates_dir.iterdir() if p.is_dir() and p.name.startswith("scene_")]
    )

    if not scene_dirs:
        print(f"⚠ 未在 {args.candidates_dir} 中找到 scene_* 子目录")
        return

    copied = 0
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name  # scene_001
        scene_idx = scene_name.replace("scene_", "")
        src = find_candidate_image(scene_dir)
        if src is None:
            print(f"⚠ 场景 {scene_name} 未找到候选图像")
            continue

        dst = target_dir / f"{scene_name}{args.ext}"
        if not args.overwrite and dst.exists():
            print(f"跳过 {dst}（已存在，可加 --overwrite 覆盖）")
            continue

        if args.dry_run:
            print(f"[dry-run] {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            copied += 1
            if not args.quiet:
                print(f"✓ 复制 {src.name} -> {dst.name}")

    if not args.dry_run:
        print(f"\n已更新 {copied} 个场景素材到 {target_dir}")


if __name__ == "__main__":
    main()



