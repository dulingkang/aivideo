#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 CLIP 语义相似度对素材帧进行初筛。

- 读取素材元数据 CSV（默认 `assets/library/metadata.csv`）
- 为每张图片计算与“主角提示词”和“排除提示词”的相似度
- 输出筛选标记，可选择复制通过的图片到指定目录

示例：
    python filter_stills_clip.py \
        --metadata ../assets/library/metadata.csv \
        --positive "close-up portrait of Han Li, male cultivator in green robe" \
        --positive-file hanli_positive.txt \
        --negative "crowd of people" \
        --positive-threshold 0.26 \
        --negative-threshold 0.22 \
        --append-column clip_flag \
        --selected-dir ../assets/library/clip_selected
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import open_clip
from PIL import Image
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "gen_video" / "assets" / "library" / "metadata.csv"


@dataclass
class ClipConfig:
    checkpoint: str
    pretrained: str
    device: str
    batch_size: int
    quiet: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="利用 CLIP 对素材帧进行语义初筛")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="素材元数据 CSV（默认: assets/library/metadata.csv）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="筛选后 CSV 输出路径（默认覆盖 metadata 文件）",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="可选：图片相对路径的根目录，默认取元数据里的路径",
    )
    parser.add_argument(
        "--positive",
        type=str,
        nargs="*",
        default=[],
        help="主角正向提示词（命令行指定，可多条）",
    )
    parser.add_argument(
        "--positive-file",
        type=Path,
        help="包含正向提示词的文本文件（每行一个，或 YAML 的 prompts 列）",
    )
    parser.add_argument(
        "--negative",
        type=str,
        nargs="*",
        default=[],
        help="排除提示词列表，可多条",
    )
    parser.add_argument(
        "--negative-file",
        type=Path,
        help="包含排除提示词的文本文件",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.25,
        help="判定为主角画面的最小相似度阈值（默认 0.25）",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=0.23,
        help="判定为排除画面的最大相似度阈值（默认 0.23）",
    )
    parser.add_argument(
        "--append-column",
        type=str,
        default="clip_flag",
        help="在 CSV 中写入的标记列名（默认 clip_flag）",
    )
    parser.add_argument(
        "--score-columns",
        nargs=2,
        metavar=("POS_COL", "NEG_COL"),
        default=("clip_pos_score", "clip_neg_score"),
        help="在 CSV 中写入正向 / 负向相似度的列名（默认 clip_pos_score clip_neg_score）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ViT-L-14",
        help="CLIP 模型名称（默认 ViT-L-14）",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="预训练权重来源（默认 openai）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备（默认自动选择 cuda）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批量大小（默认 32）",
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        help="可选：复制通过筛选的图片到此目录（会按原相对路径层级保存）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅计算分数，不写入 CSV 或复制文件",
    )
    return parser.parse_args()


def load_prompts_from_file(path: Path) -> List[str]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "prompts" in data:
            prompts = data["prompts"]
        else:
            prompts = data
        if not isinstance(prompts, list):
            raise ValueError(f"{path} 中 prompts 不是列表")
        return [str(p).strip() for p in prompts if str(p).strip()]
    else:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def load_metadata(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 metadata 文件: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_metadata(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def batch_iter(seq: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def encode_texts(
    prompts: Sequence[str],
    model,
    tokenizer,
    device: str,
) -> np.ndarray:
    if not prompts:
        return np.zeros((0, model.visual.output_dim), dtype=np.float32)
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


def encode_images(
    paths: Sequence[Path],
    preprocess,
    model,
    device: str,
    batch_size: int,
    quiet: bool,
) -> np.ndarray:
    features: List[np.ndarray] = []
    batches = list(batch_iter(paths, batch_size))
    for batch_paths in tqdm(batches, desc="编码图像", disable=quiet):
        imgs = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(preprocess(img))
            except Exception:
                imgs.append(torch.zeros((3, 224, 224)))
        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats /= feats.norm(dim=-1, keepdim=True)
        features.append(feats.cpu().numpy())
    if not features:
        return np.zeros((0, model.visual.output_dim), dtype=np.float32)
    return np.concatenate(features, axis=0)


def resolve_image_path(row: dict, image_root: Optional[Path]) -> Optional[Path]:
    path_str = row.get("frame_path") or row.get("image_path")
    if not path_str:
        return None
    p = Path(path_str)
    if image_root:
        p = (image_root / p).resolve() if not p.is_absolute() else p
    return p if p.exists() else None


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata
    output_path = args.output or metadata_path

    rows = load_metadata(metadata_path)
    if not rows:
        print("⚠ metadata 为空，无需处理")
        return

    pos_prompts = list(args.positive)
    if args.positive_file:
        pos_prompts.extend(load_prompts_from_file(args.positive_file))
    neg_prompts = list(args.negative)
    if args.negative_file:
        neg_prompts.extend(load_prompts_from_file(args.negative_file))

    pos_prompts = [p.strip() for p in pos_prompts if p.strip()]
    neg_prompts = [p.strip() for p in neg_prompts if p.strip()]
    if not pos_prompts:
        print("✗ 未提供正向提示词 --positive/--positive-file", file=sys.stderr)
        sys.exit(1)

    image_root = args.image_root.resolve() if args.image_root else None
    image_paths: List[Path] = []
    valid_indices: List[int] = []
    for idx, row in enumerate(rows):
        path = resolve_image_path(row, image_root)
        if path:
            image_paths.append(path)
            valid_indices.append(idx)
        else:
            rows[idx][args.append_column] = "missing"

    if not image_paths:
        print("✗ 未找到任何图片路径，请检查 metadata 的 frame_path/image_path 字段", file=sys.stderr)
        sys.exit(1)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.checkpoint,
        pretrained=args.pretrained,
        device=args.device,
    )
    tokenizer = open_clip.get_tokenizer(args.checkpoint)
    pos_features = encode_texts(pos_prompts, model, tokenizer, args.device)
    neg_features = encode_texts(neg_prompts, model, tokenizer, args.device)

    img_features = encode_images(
        image_paths,
        preprocess,
        model,
        device=args.device,
        batch_size=args.batch_size,
        quiet=args.quiet,
    )

    pos_scores = img_features @ pos_features.T if pos_features.size else np.zeros((len(img_features), 0))
    neg_scores = img_features @ neg_features.T if neg_features.size else np.zeros((len(img_features), 0))

    pos_best = pos_scores.max(axis=1) if pos_scores.size else np.zeros(len(img_features))
    neg_best = neg_scores.max(axis=1) if neg_scores.size else np.zeros(len(img_features))

    pos_col, neg_col = args.score_columns
    selected_indices: List[int] = []
    for feat_idx, row_idx in enumerate(valid_indices):
        row = rows[row_idx]
        row[pos_col] = f"{pos_best[feat_idx]:.4f}"
        row[neg_col] = f"{neg_best[feat_idx]:.4f}" if neg_scores.size else ""
        keep = pos_best[feat_idx] >= args.positive_threshold
        if neg_scores.size:
            keep = keep and neg_best[feat_idx] < args.negative_threshold
        row[args.append_column] = "keep" if keep else "reject"
        if keep:
            selected_indices.append(row_idx)

    kept = len(selected_indices)
    print(
        f"✓ CLIP 筛选完成: 保留 {kept}/{len(valid_indices)} 张 "
        f"(pos>= {args.positive_threshold}, neg<{args.negative_threshold})"
    )

    if args.dry_run:
        print("🍀 dry-run 模式，不写入文件")
        return

    if args.selected_dir and kept:
        dst_root = args.selected_dir
        dst_root.mkdir(parents=True, exist_ok=True)
        from shutil import copy2

        for row_idx in selected_indices:
            src = resolve_image_path(rows[row_idx], image_root)
            if not src:
                continue
            if image_root and src.is_relative_to(image_root):
                rel = src.relative_to(image_root)
            elif not src.is_absolute():
                rel = src
            else:
                rel = Path(src.name)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                copy2(src, dst)
            except Exception as exc:
                print(f"⚠ 复制失败 {src} -> {dst}: {exc}")

    write_metadata(output_path, rows)
    print(f"✓ 已写入筛选结果: {output_path}")


if __name__ == "__main__":
    main()

