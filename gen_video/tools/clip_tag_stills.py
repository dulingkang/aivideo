#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 CLIP 模型为素材帧生成场景标签。

依赖：
    pip install open-clip-torch Pillow tqdm numpy

示例：
    python clip_tag_stills.py \
        --metadata ../assets/library/metadata.csv \
        --output ../assets/library/metadata.csv \
        --checkpoint ViT-L-14 \
        --device cuda \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open_clip
from PIL import Image
import torch
from tqdm import tqdm
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "gen_video" / "assets" / "library" / "metadata.csv"
DEFAULT_OUTPUT = DEFAULT_METADATA

# 预定义场景标签，可根据素材特点调整 / 扩展
DEFAULT_PROMPTS = [
    "wide cinematic shot of a vast golden desert landscape under a bright sun",
    "storming sand dune with strong wind and flying sand particles",
    "ancient xianxia style palace or temple surrounded by mystical aura",
    "luxurious interior hall with tall pillars and glowing inscriptions",
    "intense magical battle with cultivators clashing and energy explosions",
    "fiery explosion lighting up the scene with orange sparks",
    "close-up of a serious male cultivator holding a glowing weapon",
    "dark ominous sky with swirling clouds over the desert",
    "calm twilight desert with warm sunset colors and long shadows",
    "glowing mystical artifact floating in the air emitting blue light",
    "vast canyon with steep rocky walls under dramatic lighting",
    "mysterious night scene with stars and luminescent fog",
    "wide aerial shot of blue sky and sand sea from high altitude",
    "battlefield with debris, dust and bright sword trails",
    "close-up of determined female cultivator with magic energy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="利用 CLIP 为素材帧生成场景标签")
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        help="自定义文本提示列表（命令行直接提供）",
    )
    group.add_argument(
        "--prompt-file",
        type=Path,
        help="自定义提示词文件（每行一个提示；支持 .txt/.yaml）",
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
        help="推理设备（默认自动检测 cuda，否则 cpu）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="推理批大小（默认 32）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每张图保留的最有可能标签数量（默认 3）",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="相似度阈值，低于此值的标签将被过滤（默认 0.15）",
    )
    parser.add_argument(
        "--append-column",
        type=str,
        default="clip_tags",
        help="输出 CSV 中新标签列名（默认 clip_tags）",
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


def write_metadata(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_clip_model(
    checkpoint: str,
    pretrained: str,
    device: str,
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        checkpoint,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(checkpoint)
    return model, preprocess, tokenizer


def encode_prompts(
    prompts: Sequence[str],
    model,
    tokenizer,
    device: str,
) -> np.ndarray:
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def batch(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def encode_images(
    image_paths: Sequence[Path],
    preprocess,
    model,
    device: str,
    batch_size: int,
    quiet: bool,
) -> np.ndarray:
    features: List[np.ndarray] = []
    for paths in tqdm(list(batch(image_paths, batch_size)), desc="编码图像", disable=quiet):
        images = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(preprocess(img))
            except Exception:
                images.append(torch.zeros((3, 224, 224)))
        image_tensor = torch.stack(images, dim=0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        features.append(image_features.cpu().numpy())
    if features:
        return np.vstack(features)
    return np.empty((0, model.visual.output_dim))


def assign_tags(
    rows: List[Dict[str, str]],
    metadata_dir: Path,
    model,
    preprocess,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    batch_size: int,
    top_k: int,
    threshold: float,
    quiet: bool,
) -> Dict[str, List[Tuple[str, float]]]:
    prompt_features = encode_prompts(prompts, model, tokenizer, device)
    image_paths = [(metadata_dir / row["frame_path"]).resolve() for row in rows]
    image_features = encode_images(image_paths, preprocess, model, device, batch_size, quiet)

    if image_features.shape[0] != len(rows):
        raise RuntimeError("图像特征数量与 CSV 行数不一致")

    # 相似度矩阵
    similarity = image_features @ prompt_features.T  # [num_images, num_prompts]
    tag_map: Dict[str, List[Tuple[str, float]]] = {}

    for row, sims in zip(rows, similarity):
        idxs = np.argsort(sims)[::-1][:top_k]
        tags: List[Tuple[str, float]] = []
        for idx in idxs:
            score = float(sims[idx])
            if score < threshold:
                continue
            tags.append((prompts[idx], score))
        tag_map[row["frame_path"]] = tags
    return tag_map


def format_tags(tags: Sequence[Tuple[str, float]]) -> str:
    return ";".join(f"{text}({score:.3f})" for text, score in tags)


def merge_clip_tags(
    rows: List[Dict[str, str]],
    tag_map: Dict[str, List[Tuple[str, float]]],
    column: str,
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for row in rows:
        frame_path = row.get("frame_path", "")
        tags = tag_map.get(frame_path, [])
        new_row = dict(row)
        new_row[column] = format_tags(tags)
        merged.append(new_row)
    return merged


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    output_path = args.output.resolve()
    metadata_dir = metadata_path.parent.resolve()

    if args.prompt_file:
        prompt_file = args.prompt_file.resolve()
        if not prompt_file.exists():
            raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
        if prompt_file.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(prompt_file.read_text(encoding="utf-8")) or []
            prompts = [str(item) for item in data]
        else:
            prompts = [
                line.strip()
                for line in prompt_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    elif args.prompts:
        prompts = list(args.prompts)
    else:
        prompts = DEFAULT_PROMPTS

    rows = load_metadata(metadata_path)
    if not rows:
        print("⚠ metadata.csv 为空，未进行处理")
        return

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠ 无可用 CUDA，改用 CPU 推理")
        device = "cpu"

    if not args.quiet:
        print(f"使用 CLIP 模型: {args.checkpoint} ({args.pretrained}) on {device}")
        print(f"提示词数量: {len(prompts)}")

    model, preprocess, tokenizer = load_clip_model(args.checkpoint, args.pretrained, device)
    model.to(device)
    model.eval()

    tag_map = assign_tags(
        rows=rows,
        metadata_dir=metadata_dir,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
        threshold=args.confidence_threshold,
        quiet=args.quiet,
    )

    merged_rows = merge_clip_tags(rows, tag_map, args.append_column)
    write_metadata(output_path, merged_rows)

    if not args.quiet:
        print("CLIP 场景标签生成完成 ✅")
        print(f"已更新文件: {output_path}")


if __name__ == "__main__":
    main()

