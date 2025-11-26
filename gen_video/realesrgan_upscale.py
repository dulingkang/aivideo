#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN 超分脚本

支持批量图像与视频的超分处理，依赖于 realesrgan==0.3.0。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

import cv2
import numpy as np
import torch
import types

# 全局变量：用于抑制RealESRGANer内部的Tile日志输出
_suppress_tile_output = True

# 兼容 torchvision>=0.17 移除了 functional_tensor 模块的情况
try:
    from torchvision.transforms import functional_tensor as _functional_tensor  # type: ignore
except ImportError:
    try:
        from torchvision.transforms import functional as _functional  # type: ignore
    except ImportError:
        _functional = None
    if _functional is not None:
        functional_tensor_shim = types.ModuleType("torchvision.transforms.functional_tensor")
        functional_tensor_shim.rgb_to_grayscale = _functional.rgb_to_grayscale  # type: ignore[attr-defined]
        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor_shim

# 减少 Real-ESRGAN 和 basicsr 的日志输出（提高速度）
logging.getLogger('basicsr').setLevel(logging.WARNING)
logging.getLogger('realesrgan').setLevel(logging.WARNING)
logging.getLogger('facexlib').setLevel(logging.WARNING)

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    print("未安装 realesrgan 或 basicsr，请先执行 `pip install realesrgan basicsr`")
    sys.exit(1)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".flv"}


def build_model(model_path: Path, scale: int = 4, half: bool = True, tile: int = 0, verbose: bool = False) -> RealESRGANer:
    """构建 RealESRGAN 推理器
    
    Args:
        model_path: 模型路径
        scale: 模型基础缩放倍率
        half: 是否使用半精度（FP16）
        tile: 瓦片大小，0 表示不使用瓦片
        verbose: 是否显示详细日志
    """

    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {model_path}")
    
    # 根据模型文件名自动检测架构参数
    model_name = model_path.name.lower()
    if "anime_6b" in model_name or "6b" in model_name:
        # RealESRGAN_x4plus_anime_6B.pth 使用 6 blocks
        num_block = 6
        num_grow_ch = 32
        if verbose:
            print(f"  检测到 anime_6B 模型，使用架构参数: num_block={num_block}, num_grow_ch={num_grow_ch}")
    elif "x2plus" in model_name or "x2" in model_name:
        # RealESRGAN_x2plus.pth 使用 23 blocks，scale=2
        num_block = 23
        num_grow_ch = 32
        if scale != 2:
            scale = 2  # x2模型强制使用scale=2
        if verbose:
            print(f"  检测到 x2plus 模型，使用架构参数: num_block={num_block}, num_grow_ch={num_grow_ch}, scale={scale}")
    else:
        # 默认 RealESRGAN_x4plus.pth 使用 23 blocks
        num_block = 23
        num_grow_ch = 32
        if verbose:
            print(f"  使用默认架构参数: num_block={num_block}, num_grow_ch={num_grow_ch}")
    
    try:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
            scale=scale,
        )
        
        # 优化：如果 tile=0 且图像较大，自动设置 tile 以提高速度
        # tile 参数可以将大图像分割成小块处理，减少显存占用并可能提高速度
        auto_tile = tile
        if tile == 0:
            # 对于较大的图像（>1024），使用 tile 可以加速
            # 但需要先知道图像尺寸，这里先使用配置的 tile 值
            # 如果用户想要自动优化，可以在调用时传入合适的 tile 值
            pass
        
        # 抑制模型加载时的日志输出
        with contextlib.redirect_stderr(io.StringIO()):
            upscaler = RealESRGANer(
                scale=scale,
                model_path=str(model_path),
                model=model,
                tile=auto_tile,
                tile_pad=10 if auto_tile > 0 else 0,  # 如果 tile=0，不需要 tile_pad
                pre_pad=10,
                half=half and torch.cuda.is_available(),
            )
        return upscaler
    except Exception as e:
        # 如果加载失败，尝试不指定 model 参数，让 RealESRGANer 自动加载
        if verbose:
            print(f"  ⚠ 使用指定架构参数加载失败: {e}")
            print(f"  尝试让 RealESRGANer 自动检测模型架构...")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                upscaler = RealESRGANer(
                    scale=scale,
                    model_path=str(model_path),
                    model=None,  # 让 RealESRGANer 自动加载模型
                    tile=tile,
                    tile_pad=10,
                    pre_pad=10,
                    half=half and torch.cuda.is_available(),
                )
            if verbose:
                print(f"  ✓ 自动加载成功")
            return upscaler
        except Exception as e2:
            raise RuntimeError(f"模型加载失败: {e2}") from e2


def iter_images(directory: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        yield from (p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        yield from (p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def upscale_image(
    upscaler: RealESRGANer,
    src_path: Path,
    dst_path: Path,
    outscale: float,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"✗ 读取失败: {src_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 抑制 Real-ESRGAN 内部的日志输出（包括Tile日志）
    # 使用os.devnull彻底抑制所有输出
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            output, _ = upscaler.enhance(img, outscale=outscale)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dst_path), output)
    print(f"✓ 图像已保存: {dst_path}")


def upscale_image_dir(
    upscaler: RealESRGANer,
    src_dir: Path,
    dst_dir: Path,
    outscale: float,
    recursive: bool = False,
) -> None:
    files = list(iter_images(src_dir, recursive=recursive))
    if not files:
        print(f"目录中未找到图片: {src_dir}")
        return

    for src in files:
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        upscale_image(upscaler, src, dst, outscale)


def upscale_video(
    upscaler: RealESRGANer,
    src_path: Path,
    dst_path: Path,
    outscale: float,
    fps: Optional[float] = None,
    codec: str = "mp4v",
    batch_size: int = 1,
    num_workers: int = 1,
) -> None:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src_path}")

    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)

    sample = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    with contextlib.redirect_stderr(io.StringIO()):
        sample_up, _ = upscaler.enhance(sample, outscale=outscale)
    out_h, out_w = sample_up.shape[:2]

    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {dst_path}")

    frame_idx = 0
    start_time = time.time()
    last_print_time = start_time
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # RealESRGANer 的 enhance 方法已经支持 tile 参数（在 build_model 时设置）
            # 使用 tile 可以加速处理大图像，减少显存占用，提高GPU利用率
            # 抑制 Real-ESRGAN 内部的日志输出（包括Tile日志）
            # 使用os.devnull彻底抑制所有输出
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                try:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    output, _ = upscaler.enhance(rgb, outscale=outscale)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            writer.write(output_bgr)
            frame_idx += 1
            
            # 每 60 帧或每 5 秒显示一次进度（进一步减少输出频率，提高速度）
            current_time = time.time()
            if frame_idx % 60 == 0 or (current_time - last_print_time) >= 5.0 or frame_idx == total_frames:
                progress_pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                elapsed = current_time - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
                print(f"\r处理进度: {frame_idx}/{total_frames} ({progress_pct:.1f}%) | 速度: {fps_actual:.1f} 帧/秒 | 预计剩余: {eta:.0f}秒", end="", flush=True)
                last_print_time = current_time
    finally:
        cap.release()
        writer.release()
        print()
        print(f"✓ 视频已保存: {dst_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Real-ESRGAN 对图像/视频进行超分")
    parser.add_argument("--input", required=True, help="输入文件或目录")
    parser.add_argument("--output", required=True, help="输出文件或目录")
    parser.add_argument(
        "--model-path",
        default="models/realesrgan/RealESRGAN_x4plus.pth",
        help="Real-ESRGAN 权重文件路径",
    )
    parser.add_argument("--scale", type=int, default=4, help="模型基础缩放倍率（默认为 x4）")
    parser.add_argument("--outscale", type=float, default=4.0, help="最终输出缩放倍率")
    parser.add_argument("--tile", type=int, default=0, help="瓦片大小，显存不足时可设置为 64/128 等")
    parser.add_argument("--full-precision", action="store_true", help="禁用 half 精度推理")
    parser.add_argument("--recursive", action="store_true", help="遍历子目录处理图片")
    parser.add_argument("--video-fps", type=float, default=None, help="输出视频帧率（默认沿用源视频）")
    parser.add_argument("--video-codec", default="mp4v", help="OpenCV VideoWriter 编码器 (mp4v/x264/xvid)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="处理视频时的批大小（当前实现逐帧处理，预留参数防止脚本崩溃）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行读取帧的线程数（预留参数，与 VideoComposer 中配置保持一致）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    upscaler = build_model(
        model_path=model_path,
        scale=args.scale,
        half=not args.full_precision,
        tile=args.tile,
    )

    if input_path.is_dir():
        upscale_image_dir(
            upscaler,
            input_path,
            output_path,
            outscale=args.outscale,
            recursive=args.recursive,
        )
    else:
        suffix = input_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            dst = output_path
            if output_path.is_dir() or output_path.suffix == "":
                dst = output_path / input_path.name
            upscale_image(upscaler, input_path, dst, outscale=args.outscale)
        elif suffix in VIDEO_EXTENSIONS:
            dst = output_path
            if output_path.is_dir() or output_path.suffix == "":
                dst = output_path / input_path.name
            upscale_video(
                upscaler,
                input_path,
                dst,
                outscale=args.outscale,
                fps=args.video_fps,
                codec=args.video_codec,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        else:
            raise ValueError(f"不支持的文件类型: {input_path}")


if __name__ == "__main__":
    main()


