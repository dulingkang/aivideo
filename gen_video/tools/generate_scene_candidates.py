#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据场景脚本和素材标签，自动挑选候选帧供后续人工确认。

示例：
    python generate_scene_candidates.py \
        --scene-json ../lingjie/5-青罗沙漠.json \
        --metadata assets/library/metadata.csv \
        --output-dir assets/library/scene_candidates/ep5 \
        --per-scene 6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_JSON = REPO_ROOT / "lingjie" / "5-青罗沙漠.json"
DEFAULT_METADATA = REPO_ROOT / "gen_video" / "assets" / "library" / "metadata.csv"
DEFAULT_OUTPUT = REPO_ROOT / "gen_video" / "assets" / "library" / "scene_candidates" / "ep5"


KEYWORD_RULES = [
    ("desert", ["沙漠", "沙", "黄沙", "风沙", "sand", "desert", "沙海"]),
    ("palace", ["宫殿", "殿堂", "大厅", "宫", "殿", "temple", "palace", "hall"]),
    ("battle", ["战斗", "激战", "交锋", "对峙", "battle", "clash", "冲击", "碰撞"]),
    ("mystic", ["灵光", "灵气", "阵", "幻影", "artifact", "mystical", "portal", "法阵", "光团"]),
    ("sunset", ["夕阳", "余晖", "落日", "sunset", "twilight"]),
    ("night", ["夜", "黑暗", "星空", "night", "moon"]),
]

TAG_MATCHERS: Dict[str, List[str]] = {
    "desert": ["desert", "sand"],
    "palace": ["palace", "temple", "hall"],
    "battle": ["battle", "sword", "explosion", "beam", "duel", "shockwave"],
    "mystic": ["mystical", "artifact", "glowing", "portal", "gate", "aura"],
    "sunset": ["sunset", "twilight", "warm"],
    "night": ["night", "moonlight", "dark sky"],
}

FALLBACK_PRIORITY = ["battle", "desert", "palace", "mystic", "sunset", "night"]


@dataclass
class SceneInfo:
    index: int
    title: str
    description: str
    tags: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为每个场景挑选候选素材帧")
    parser.add_argument(
        "--scene-json",
        type=Path,
        default=DEFAULT_SCENE_JSON,
        help="场景脚本 JSON（默认 lingjie/5-青罗沙漠.json）",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="素材元数据 CSV（默认 assets/library/metadata.csv）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="候选帧输出目录（默认 assets/library/scene_candidates/ep5）",
    )
    parser.add_argument(
        "--per-scene",
        type=int,
        default=6,
        help="每个场景挑选的候选帧数量（默认 6）",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制（默认移动）候选帧",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    return parser.parse_args()


def load_scene_json(path: Path) -> List[SceneInfo]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        scenes_raw = data.get("scenes") or data.get("Scenes") or []
    elif isinstance(data, list):
        scenes_raw = data
    else:
        raise ValueError(f"未知脚本格式: {type(data)}")
    scenes: List[SceneInfo] = []

    for scene in scenes_raw:
        idx = scene.get("scene_number") or scene.get("scene_id")
        if idx is None:
            continue

        text_parts = [
            scene.get("title", ""),
            scene.get("description", ""),
            scene.get("environment", ""),
            scene.get("action", ""),
            scene.get("mood", ""),
            scene.get("narration", ""),
        ]
        combined = " ".join(text_parts).lower()

        tags = infer_tags_from_text(combined)
        scenes.append(
            SceneInfo(
                index=int(idx),
                title=scene.get("title", f"场景{idx}"),
                description=" ".join(text_parts),
                tags=tags,
            )
        )

    scenes.sort(key=lambda x: x.index)
    return scenes


def infer_tags_from_text(text: str) -> List[str]:
    matched = []
    for tag, keywords in KEYWORD_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            matched.append(tag)
    if not matched:
        # fallback: if contains "沙" etc
        if "沙" in text or "sand" in text:
            matched.append("desert")
        elif "战" in text or "battle" in text:
            matched.append("battle")
        else:
            matched.append("mystic")
    return matched


def load_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_clip_tags(tag_string: str) -> List[str]:
    if not tag_string:
        return []
    tags = []
    for segment in tag_string.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if "(" in segment:
            segment = segment.split("(", 1)[0]
        tags.append(segment.lower())
    return tags


def select_candidates_for_scene(
    rows: List[Dict[str, str]],
    tags: List[str],
    per_scene: int,
) -> List[Dict[str, str]]:
    candidates: List[Tuple[float, float, Dict[str, str]]] = []
    desired = list(dict.fromkeys(tags))  # preserve order

    # Build search tokens for clip tags
    tokens: List[str] = []
    for tag in desired:
        tokens.extend(TAG_MATCHERS.get(tag, []))

    for row in rows:
        clip_tags = parse_clip_tags(row.get("clip_tags", ""))
        auto_tags = [t.strip().lower() for t in (row.get("auto_tags", "") or "").split(",")]

        score = 0
        for token in tokens:
            if any(token in ct for ct in clip_tags):
                score += 2
        for tag in desired:
            if tag in auto_tags:
                score += 1

        if score == 0:
            continue

        try:
            sharpness = float(row.get("sharpness", "0") or 0.0)
        except ValueError:
            sharpness = 0.0
        try:
            brightness = float(row.get("brightness", "0") or 0.0)
        except ValueError:
            brightness = 0.0

        weight = score * 10 + sharpness * 0.6 + brightness * 0.2
        candidates.append((weight, sharpness, row))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    if len(candidates) < per_scene:
        # Fallback: pick top sharpness regardless of tag
        extra = sorted(
            rows,
            key=lambda r: float(r.get("sharpness", "0") or 0.0),
            reverse=True,
        )
        for row in extra:
            if any(row is c[2] for c in candidates):
                continue
            candidates.append((0.0, float(row.get("sharpness", "0") or 0.0), row))
            if len(candidates) >= per_scene:
                break

    return [c[2] for c in candidates[:per_scene]]


def copy_candidates(
    candidates: List[Dict[str, str]],
    metadata_dir: Path,
    scene_output_dir: Path,
    copy: bool,
) -> int:
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in candidates:
        frame_rel = row.get("frame_path", "")
        src = (metadata_dir / frame_rel).resolve()
        if not src.exists():
            continue
        dst = scene_output_dir / Path(frame_rel).name
        if copy:
            shutil.copy2(src, dst)
        else:
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        count += 1
    return count


def main() -> None:
    args = parse_args()

    scenes = load_scene_json(args.scene_json.resolve())
    rows = load_metadata(args.metadata.resolve())
    metadata_dir = args.metadata.resolve().parent

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for scene in scenes:
        candidates = select_candidates_for_scene(rows, scene.tags, args.per_scene)
        scene_dir = args.output_dir / f"scene_{scene.index:03d}"
        copied = copy_candidates(candidates, metadata_dir, scene_dir, args.copy)
        total += copied
        if not args.quiet:
            print(
                f"场景 {scene.index:02d} ({scene.title}) "
                f"标签 {scene.tags} -> {copied} 张候选"
            )

    if not args.quiet:
        print(f"\n已为 {len(scenes)} 个场景生成 {total} 张候选帧，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

