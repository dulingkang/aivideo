#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 metadata.csv 中的标签关键字自动归档素材帧。

示例：
    python sort_stills_by_tags.py \
        --metadata ../assets/library/metadata.csv \
        --output-dir ../assets/library/by_tag \
        --config ../assets/library/tag_groups.yaml \
        --source-column clip_tags \
        --copy

配置文件 (YAML) 示例：
    desert:
      keywords: ["desert", "sand dune", "desert_like"]
    palace:
      keywords: ["palace", "temple", "hall"]
      include_auto_tags: true
    battle:
      keywords: ["battle", "explosion"]
      min_sharpness: 15
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "gen_video" / "assets" / "library" / "metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "gen_video" / "assets" / "library" / "by_tag"
DEFAULT_CONFIG = REPO_ROOT / "gen_video" / "assets" / "library" / "tag_groups.yaml"


@dataclass
class TagRule:
    name: str
    keywords: List[str]
    include_auto_tags: bool = False
    include_notes: bool = False
    min_sharpness: Optional[float] = None
    min_brightness: Optional[float] = None
    max_brightness: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据标签关键字自动归档素材帧")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="素材元数据 CSV（默认: assets/library/metadata.csv）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="归档输出目录（默认: assets/library/by_tag）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="标签规则配置 YAML（默认: assets/library/tag_groups.yaml）",
    )
    parser.add_argument(
        "--source-column",
        type=str,
        default="clip_tags",
        help="使用哪一列提取关键字（默认 clip_tags，可选 auto_tags/tags/notes）",
    )
    parser.add_argument(
        "--auto-column",
        type=str,
        default="auto_tags",
        help="启用 include_auto_tags 时使用的列（默认 auto_tags）",
    )
    parser.add_argument(
        "--notes-column",
        type=str,
        default="notes",
        help="启用 include_notes 时使用的列（默认 notes）",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制文件（默认移动文件）",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="不按剧集分子目录，所有匹配帧直接存入标签目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印匹配结果，不执行复制/移动",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 metadata 文件: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_rules(config_path: Path) -> List[TagRule]:
    if not config_path.exists():
        # 默认规则
        return [
            TagRule(name="desert", keywords=["desert", "sand", "desert_like", "sand dune"]),
            TagRule(name="palace", keywords=["palace", "temple", "hall", "宫殿"]),
            TagRule(name="battle", keywords=["battle", "explosion", "激烈"]),
            TagRule(name="mystic", keywords=["mystical", "artifact", "glowing", "灵气"]),
        ]
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    rules: List[TagRule] = []
    for name, cfg in data.items():
        keywords = cfg.get("keywords", [])
        include_auto = bool(cfg.get("include_auto_tags", False))
        include_notes = bool(cfg.get("include_notes", False))
        min_sharp = cfg.get("min_sharpness")
        min_brightness = cfg.get("min_brightness")
        max_brightness = cfg.get("max_brightness")
        rules.append(
            TagRule(
                name=name,
                keywords=keywords,
                include_auto_tags=include_auto,
                include_notes=include_notes,
                min_sharpness=min_sharp,
                min_brightness=min_brightness,
                max_brightness=max_brightness,
            )
        )
    return rules


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def tokenise_tags(text: str) -> List[str]:
    if not text:
        return []
    # 支持 clip_tags 格式: "xxx(0.65);yyy(0.32)"
    parts = []
    for segment in text.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if "(" in segment:
            segment = segment.split("(", 1)[0]
        parts.append(segment.lower())
    return parts


def match_rule(
    row: Dict[str, str],
    rule: TagRule,
    source_column: str,
    auto_column: str,
    notes_column: str,
) -> bool:
    text_sources: List[str] = []
    if source_column in row:
        text_sources.append(row[source_column])
    if rule.include_auto_tags and auto_column in row:
        text_sources.append(row[auto_column])
    if rule.include_notes and notes_column in row:
        text_sources.append(row[notes_column])

    combined_text = " ".join(text_sources).lower()
    if not combined_text:
        return False

    if rule.keywords:
        pattern = r"|".join(re.escape(k.lower()) for k in rule.keywords)
        if not re.search(pattern, combined_text):
            return False

    def float_or_none(value: str) -> Optional[float]:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    if rule.min_sharpness is not None:
        sharpness = float_or_none(row.get("sharpness", ""))
        if sharpness is None or sharpness < rule.min_sharpness:
            return False

    if rule.min_brightness is not None:
        brightness = float_or_none(row.get("brightness", ""))
        if brightness is None or brightness < rule.min_brightness:
            return False

    if rule.max_brightness is not None:
        brightness = float_or_none(row.get("brightness", ""))
        if brightness is None or brightness > rule.max_brightness:
            return False

    return True


def archive_frames(
    rows: List[Dict[str, str]],
    rules: List[TagRule],
    metadata_dir: Path,
    output_dir: Path,
    source_column: str,
    auto_column: str,
    notes_column: str,
    copy: bool,
    flat_output: bool,
    dry_run: bool,
    quiet: bool,
) -> None:
    ensure_directory(output_dir)
    summary: Dict[str, int] = {rule.name: 0 for rule in rules}

    for row in rows:
        frame_rel = row.get("frame_path", "")
        frame_path = (metadata_dir / frame_rel).resolve()
        if not frame_path.exists():
            if not quiet:
                print(f"⚠ 素材不存在: {frame_path}")
            continue

        for rule in rules:
            if not match_rule(row, rule, source_column, auto_column, notes_column):
                continue

            if flat_output:
                target_dir = output_dir / rule.name
            else:
                target_dir = output_dir / rule.name / row.get("episode", "unknown")
            ensure_directory(target_dir)
            target_path = target_dir / Path(frame_rel).name

            summary[rule.name] += 1
            if dry_run:
                if not quiet:
                    print(f"[dry-run] {frame_path} -> {target_path}")
                continue

            if copy:
                shutil.copy2(frame_path, target_path)
            else:
                shutil.move(frame_path, target_path)

    if not quiet:
        print("\n归档完成 ✅")
        for name, count in summary.items():
            print(f"  {name}: {count} 张")


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    output_dir = args.output_dir.resolve()
    metadata_dir = metadata_path.parent.resolve()

    rows = load_metadata(metadata_path)
    if not rows:
        print("⚠ metadata.csv 为空，未进行处理")
        return

    rules = load_rules(args.config)
    if not rules:
        print("⚠ 未定义任何规则，使用 --config 添加配置")
        return

    archive_frames(
        rows=rows,
        rules=rules,
        metadata_dir=metadata_dir,
        output_dir=output_dir,
        source_column=args.source_column,
        auto_column=args.auto_column,
        notes_column=args.notes_column,
        copy=args.copy,
        flat_output=args.flat_output,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

