#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只做超分的脚本
跳过图像生成、视频生成、TTS、字幕等流程，仅对已有视频进行超分处理
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_composer import VideoComposer


def upscale_video_only(
    video_path: str,
    config_path: str = "config.yaml",
    output_path: Optional[str] = None,
) -> str:
    """
    只对视频进行超分处理
    
    Args:
        video_path: 输入视频路径
        config_path: 配置文件路径
        output_path: 输出视频路径（可选，默认在输入文件同目录生成）
    
    Returns:
        输出视频路径
    """
    # 加载配置
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取超分配置
    post_cfg = config.get('composition', {}).get('postprocess', {})
    if not post_cfg.get('enabled', False):
        print("⚠ 警告: 配置中 postprocess.enabled 为 false，将强制启用超分")
        post_cfg['enabled'] = True
    
    # 初始化 VideoComposer（只需要加载配置，不需要其他组件）
    composer = VideoComposer(config_path)
    
    # 确定输出路径
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入视频不存在: {video_path}")
    
    if output_path is None:
        suffix = post_cfg.get('suffix', '_upx2')
        output_path = str(input_path.with_name(input_path.stem + suffix + input_path.suffix))
    
    print("=" * 60)
    print("视频超分处理")
    print("=" * 60)
    print(f"输入: {video_path}")
    print(f"输出: {output_path}")
    print(f"模型: {post_cfg.get('model_path', 'N/A')}")
    print(f"缩放: {post_cfg.get('outscale', 2.0)}x")
    print("=" * 60)
    
    # 执行超分
    result_path = composer.postprocess_with_realesrgan(video_path, post_cfg)
    
    # 如果指定了输出路径且与结果路径不同，重命名
    if output_path != result_path and Path(result_path).exists():
        Path(result_path).rename(output_path)
        result_path = output_path
    
    print(f"\n✓ 超分完成: {result_path}")
    return result_path


def main():
    parser = argparse.ArgumentParser(description="只对视频进行超分处理")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", help="输出视频路径（可选，默认在输入文件同目录生成）")
    parser.add_argument("--config", "-c", default="config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        upscale_video_only(
            video_path=args.input,
            config_path=args.config,
            output_path=args.output,
        )
    except Exception as e:
        print(f"✗ 超分失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

