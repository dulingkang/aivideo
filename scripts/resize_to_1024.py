#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Tuple

from PIL import Image


def compute_letterbox_size(original_size: Tuple[int, int], target_size: int) -> Tuple[int, int]:
    width, height = original_size
    if width == 0 or height == 0:
        return target_size, target_size
    scale = min(target_size / width, target_size / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_w, new_h


def process_image(src_path: Path, dst_path: Path, target_size: int = 1024, background_color=(0, 0, 0)) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        new_w, new_h = compute_letterbox_size(img.size, target_size)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target_size, target_size), background_color)
        left = (target_size - new_w) // 2
        top = (target_size - new_h) // 2
        canvas.paste(resized, (left, top))
        # Preserve original extension if sensible; default to .png when uncommon
        ext = src_path.suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".png"
            dst_path = dst_path.with_suffix(ext)
        save_kwargs = {}
        if ext in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95
            save_kwargs["optimize"] = True
        elif ext == ".png":
            save_kwargs["optimize"] = True
        canvas.save(dst_path, **save_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize images to 1024x1024 with letterbox padding.")
    parser.add_argument("input_dir", type=str, help="Input directory containing images.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for resized images. Default: '<input_dir>_1024'",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=1024,
        help="Target square size (default: 1024).",
    )
    parser.add_argument(
        "-b",
        "--background",
        type=str,
        default="black",
        help="Background color for padding (name or hex), default: black.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(str(input_dir) + "_1024")

    # Resolve background color
    bg = args.background
    if bg.startswith("#"):
        bg = bg.lstrip("#")
        if len(bg) == 6:
            background_color = tuple(int(bg[i:i+2], 16) for i in (0, 2, 4))
        else:
            raise SystemExit("Invalid hex color. Use format like #000000")
    else:
        # allow basic names; fallback to black
        named = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (127, 127, 127),
            "grey": (127, 127, 127),
        }
        background_color = named.get(bg.lower(), (0, 0, 0))

    supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
    count = 0
    for entry in sorted(input_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in supported_exts:
            continue
        rel_name = entry.name
        dst_path = output_dir / rel_name
        try:
            process_image(entry, dst_path, target_size=args.size, background_color=background_color)
            count += 1
        except Exception as e:
            print(f"[WARN] Failed processing {entry}: {e}")

    print(f"Done. Processed {count} images into: {output_dir}")


if __name__ == "__main__":
    main()


