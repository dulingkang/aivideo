#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 InsightFace 的人脸相似度筛选脚本。

用途：
- 读取素材元数据 CSV
- 使用 reference 目录中的主角照片构建人脸特征
- 对候选素材逐帧检测人脸并计算与主角的余弦相似度
- 生成筛选标记，可选择复制通过的图片

依赖：
    pip install insightface onnxruntime-gpu pillow numpy opencv-python

示例：
    python filter_stills_face.py \
        --metadata ../assets/library/metadata.csv \
        --reference-dir ../reference_image/韩立 \
        --threshold 0.32 \
        --append-column face_flag \
        --selected-dir ../assets/library/face_selected
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

try:
    from insightface.app import FaceAnalysis  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "缺少 insightface 依赖，请先执行 `pip install insightface onnxruntime-gpu`"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "gen_video" / "assets" / "library" / "metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用人脸识别筛选素材帧")
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
        help="图片相对路径的根目录（默认按元数据路径解析）",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="主角参考图片目录（将使用其中的人脸构建特征）",
    )
    parser.add_argument(
        "--globs",
        nargs="*",
        default=("*.*",),
        help="参考目录下的匹配模式（默认扫描所有文件）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.32,
        help="判定为主角的最小相似度阈值（默认 0.32）",
    )
    parser.add_argument(
        "--append-column",
        type=str,
        default="face_flag",
        help="在 CSV 中写入的标记列（默认 face_flag）",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="face_similarity",
        help="在 CSV 中写入相似度分数的列名（默认 face_similarity）",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(640, 640),
        help="人脸检测输入尺寸（默认 640x640）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="buffalo_l",
        help="InsightFace 模型名称（默认 buffalo_l）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="推理设备（默认 cuda，可选 cpu）",
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        help="可选：复制通过筛选的图片到此目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅计算分数，不写入 CSV 或复制文件",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    return parser.parse_args()


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


def resolve_image_path(row: dict, image_root: Optional[Path]) -> Optional[Path]:
    path_str = row.get("frame_path") or row.get("image_path")
    if not path_str:
        return None
    p = Path(path_str)
    if image_root:
        p = (image_root / p).resolve() if not p.is_absolute() else p
    return p if p.exists() else None


def load_face_app(model: str, device: str, det_size: tuple[int, int]) -> FaceAnalysis:
    ctx_id = 0 if device == "cuda" else -1
    app = FaceAnalysis(name=model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def extract_face_embeddings(app: FaceAnalysis, image: np.ndarray) -> List[np.ndarray]:
    faces = app.get(image)
    embeddings: List[np.ndarray] = []
    for face in faces:
        if face.normed_embedding is None:
            continue
        embeddings.append(face.normed_embedding.astype(np.float32))
    return embeddings


def load_reference_embeddings(
    app: FaceAnalysis,
    ref_dir: Path,
    patterns: Iterable[str],
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    files: List[Path] = []
    for pattern in patterns:
        files.extend(ref_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"参考目录 {ref_dir} 中未找到任何图片")

    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if largest.normed_embedding is None:
            continue
        embeddings.append(largest.normed_embedding.astype(np.float32))

    if not embeddings:
        raise RuntimeError("参考目录内未检测到有效人脸，请确认素材是否清晰")

    return np.stack(embeddings, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-8:
        return -1.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata
    output_path = args.output or metadata_path

    rows = load_metadata(metadata_path)
    if not rows:
        print("⚠ metadata 为空，无需处理")
        return

    image_root = args.image_root.resolve() if args.image_root else None

    app = load_face_app(args.model, args.device, tuple(args.det_size))
    ref_embeddings = load_reference_embeddings(app, args.reference_dir, args.globs)
    ref_center = ref_embeddings.mean(axis=0)
    ref_center /= np.linalg.norm(ref_center)

    valid_indices: List[int] = []
    image_paths: List[Path] = []
    for idx, row in enumerate(rows):
        path = resolve_image_path(row, image_root)
        if path:
            valid_indices.append(idx)
            image_paths.append(path)
        else:
            rows[idx][args.append_column] = "missing"
            rows[idx][args.score_column] = ""

    selected: List[int] = []
    for img_idx, row_idx in enumerate(tqdm(valid_indices, desc="人脸筛选", disable=args.quiet)):
        path = image_paths[img_idx]
        img = cv2.imread(str(path))
        if img is None:
            rows[row_idx][args.append_column] = "load_failed"
            rows[row_idx][args.score_column] = ""
            continue
        faces = app.get(img)
        if not faces:
            rows[row_idx][args.append_column] = "no_face"
            rows[row_idx][args.score_column] = ""
            continue
        sims = [
            cosine_similarity(face.normed_embedding.astype(np.float32), ref_center)
            for face in faces
            if face.normed_embedding is not None
        ]
        score = max(sims) if sims else -1.0
        rows[row_idx][args.score_column] = f"{score:.4f}" if score >= 0 else ""
        keep = score >= args.threshold
        rows[row_idx][args.append_column] = "keep" if keep else "reject"
        if keep:
            selected.append(row_idx)

    kept = len(selected)
    print(f"✓ 人脸筛选完成: 保留 {kept}/{len(valid_indices)} 张 (阈值 {args.threshold})")

    if args.dry_run:
        print("🍀 dry-run 模式，不写入文件")
        return

    if args.selected_dir and kept:
        from shutil import copy2

        dst_root = args.selected_dir
        dst_root.mkdir(parents=True, exist_ok=True)
        for row_idx in selected:
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

