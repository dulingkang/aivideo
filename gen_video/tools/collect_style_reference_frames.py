#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从processed目录收集风格参考图像
用于训练风格LoRA或IP-Adapter多参考图像

从已有的关键帧中选择代表性图像，确保风格多样性和一致性
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
from collections import defaultdict
import random


def find_keyframes(base_dir: Path) -> List[Path]:
    """查找所有关键帧图像"""
    keyframes = []
    
    # 1. 从processed/keyframes/目录查找
    global_keyframes_dir = base_dir / "processed" / "keyframes"
    if global_keyframes_dir.exists():
        keyframes.extend(global_keyframes_dir.glob("*.jpg"))
        keyframes.extend(global_keyframes_dir.glob("*.png"))
    
    # 2. 从各个episode的keyframes目录查找
    processed_dir = base_dir / "processed"
    if processed_dir.exists():
        for episode_dir in processed_dir.iterdir():
            if episode_dir.is_dir() and episode_dir.name.startswith("episode_"):
                keyframes_dir = episode_dir / "keyframes"
                if keyframes_dir.exists():
                    keyframes.extend(keyframes_dir.glob("*.jpg"))
                    keyframes.extend(keyframes_dir.glob("*.png"))
    
    return keyframes


def group_by_episode(keyframes: List[Path]) -> Dict[str, List[Path]]:
    """按episode分组"""
    grouped = defaultdict(list)
    
    for kf in keyframes:
        # 从文件名提取episode编号
        # 格式：episode_XXX_clean-Scene-XXX_start.jpg
        name = kf.stem
        if "episode_" in name:
            parts = name.split("_")
            if len(parts) >= 2:
                episode = f"episode_{parts[1]}"
                grouped[episode].append(kf)
    
    return grouped


def select_diverse_frames(
    keyframes: List[Path],
    num_frames: int = 30,
    method: str = "diverse"
) -> List[Path]:
    """
    选择多样化的关键帧
    
    Args:
        keyframes: 所有关键帧列表
        num_frames: 需要选择的帧数
        method: 选择方法
            - "diverse": 确保多样性（不同场景、不同位置）
            - "random": 随机选择
            - "balanced": 平衡选择（每个episode平均分配）
    """
    if len(keyframes) <= num_frames:
        return keyframes
    
    if method == "random":
        return random.sample(keyframes, num_frames)
    
    elif method == "diverse":
        # 先按episode分组，确保从多个episode选择
        grouped = group_by_episode(keyframes)
        
        # 计算每个episode应该选择的帧数（平衡分配）
        num_episodes = len(grouped)
        frames_per_episode = max(1, num_frames // num_episodes)
        remaining_frames = num_frames - (frames_per_episode * num_episodes)
        
        selected = []
        
        # 从每个episode中选择多样化的帧
        for episode, episode_frames in grouped.items():
            if len(selected) >= num_frames:
                break
            
            # 当前episode需要选择的帧数
            current_need = frames_per_episode
            if remaining_frames > 0:
                current_need += 1
                remaining_frames -= 1
            
            # 按场景分组
            scene_groups = defaultdict(list)
            for kf in episode_frames:
                name = kf.stem
                if "Scene-" in name:
                    scene_num = name.split("Scene-")[1].split("_")[0]
                    scene_groups[scene_num].append(kf)
                else:
                    scene_groups["unknown"].append(kf)
            
            # 从每个场景中选择1张（优先middle帧）
            episode_selected = []
            for scene_frames in scene_groups.values():
                if len(episode_selected) >= current_need:
                    break
                # 优先选择middle帧
                middle_frames = [f for f in scene_frames if "_middle" in f.stem]
                start_frames = [f for f in scene_frames if "_start" in f.stem]
                
                if middle_frames:
                    episode_selected.extend(middle_frames[:1])
                elif start_frames:
                    episode_selected.extend(start_frames[:1])
                else:
                    episode_selected.extend(scene_frames[:1])
            
            # 如果还不够，随机补充
            if len(episode_selected) < current_need:
                remaining_episode = [f for f in episode_frames if f not in episode_selected]
                episode_selected.extend(random.sample(remaining_episode, min(current_need - len(episode_selected), len(remaining_episode))))
            
            selected.extend(episode_selected[:current_need])
        
        # 如果还不够，随机补充
        if len(selected) < num_frames:
            remaining = [f for f in keyframes if f not in selected]
            selected.extend(random.sample(remaining, min(num_frames - len(selected), len(remaining))))
        
        return selected[:num_frames]
    
    elif method == "balanced":
        # 按episode分组，平均分配
        grouped = group_by_episode(keyframes)
        selected = []
        frames_per_episode = max(1, num_frames // len(grouped))
        
        for episode_frames in grouped.values():
            selected.extend(random.sample(episode_frames, min(frames_per_episode, len(episode_frames))))
        
        # 如果还不够，随机补充
        if len(selected) < num_frames:
            remaining = [f for f in keyframes if f not in selected]
            selected.extend(random.sample(remaining, num_frames - len(selected)))
        
        return selected[:num_frames]
    
    else:
        return random.sample(keyframes, num_frames)


def copy_frames(
    selected_frames: List[Path],
    output_dir: Path,
    rename: bool = True
) -> List[Path]:
    """复制选中的帧到输出目录"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    for i, frame in enumerate(selected_frames, 1):
        if rename:
            # 重命名为 style_ref_001.jpg, style_ref_002.jpg ...
            new_name = f"style_ref_{i:03d}.jpg"
            dest = output_dir / new_name
        else:
            # 保持原名
            dest = output_dir / frame.name
        
        shutil.copy2(frame, dest)
        copied_files.append(dest)
        print(f"  ✓ 复制: {frame.name} -> {dest.name}")
    
    return copied_files


def create_metadata(
    selected_frames: List[Path],
    output_dir: Path,
    metadata_file: str = "metadata.json"
) -> None:
    """创建元数据文件"""
    metadata = {
        "total_frames": len(selected_frames),
        "source": "processed/keyframes",
        "frames": []
    }
    
    for i, frame in enumerate(selected_frames, 1):
        frame_info = {
            "index": i,
            "original_path": str(frame),
            "original_name": frame.name,
            "new_name": f"style_ref_{i:03d}.jpg" if i <= len(selected_frames) else frame.name,
            "episode": None,
            "scene": None,
            "position": None
        }
        
        # 解析文件名信息
        name = frame.stem
        if "episode_" in name:
            parts = name.split("_")
            if len(parts) >= 2:
                frame_info["episode"] = f"episode_{parts[1]}"
        
        if "Scene-" in name:
            scene_part = name.split("Scene-")[1]
            frame_info["scene"] = scene_part.split("_")[0]
        
        if "_start" in name:
            frame_info["position"] = "start"
        elif "_middle" in name:
            frame_info["position"] = "middle"
        elif "_end" in name:
            frame_info["position"] = "end"
        
        metadata["frames"].append(frame_info)
    
    metadata_path = output_dir / metadata_file
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 元数据已保存: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="从processed目录收集风格参考图像")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="项目根目录（默认：当前目录）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gen_video/reference_materials/style_frames",
        help="输出目录（默认：gen_video/reference_materials/style_frames）"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="需要选择的帧数（默认：30）"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="diverse",
        choices=["diverse", "random", "balanced"],
        help="选择方法：diverse（多样性，推荐）、random（随机）、balanced（平衡）"
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        default=True,
        help="重命名文件为 style_ref_XXX.jpg（默认：True）"
    )
    parser.add_argument(
        "--no-rename",
        dest="rename",
        action="store_false",
        help="保持原文件名"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可重复性）"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
    
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output).resolve()
    
    print(f"从processed目录收集风格参考图像")
    print(f"  基础目录: {base_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  选择方法: {args.method}")
    print(f"  目标帧数: {args.num_frames}")
    print()
    
    # 1. 查找所有关键帧
    print("查找关键帧...")
    keyframes = find_keyframes(base_dir)
    print(f"  ✓ 找到 {len(keyframes)} 张关键帧")
    
    if len(keyframes) == 0:
        print("  ✗ 未找到关键帧，请检查processed目录")
        return
    
    # 2. 按episode分组统计
    grouped = group_by_episode(keyframes)
    print(f"  ✓ 覆盖 {len(grouped)} 个episode")
    for episode, frames in sorted(grouped.items()):
        print(f"    - {episode}: {len(frames)} 张")
    
    print()
    
    # 3. 选择多样化的帧
    print(f"选择 {args.num_frames} 张代表性帧（方法：{args.method}）...")
    selected_frames = select_diverse_frames(
        keyframes,
        num_frames=args.num_frames,
        method=args.method
    )
    print(f"  ✓ 已选择 {len(selected_frames)} 张帧")
    
    # 统计选择的帧的分布
    selected_grouped = group_by_episode(selected_frames)
    print(f"  ✓ 分布：")
    for episode, frames in sorted(selected_grouped.items()):
        print(f"    - {episode}: {len(frames)} 张")
    
    print()
    
    # 4. 复制到输出目录
    print(f"复制到输出目录: {output_dir}")
    copied_files = copy_frames(selected_frames, output_dir, rename=args.rename)
    print(f"  ✓ 已复制 {len(copied_files)} 张帧")
    
    print()
    
    # 5. 创建元数据
    print("创建元数据...")
    create_metadata(selected_frames, output_dir)
    
    print()
    print("=" * 60)
    print("✓ 完成！")
    print(f"  输出目录: {output_dir}")
    print(f"  帧数: {len(copied_files)}")
    print(f"  元数据: {output_dir / 'metadata.json'}")
    print()
    print("下一步：")
    print("  1. 检查选中的帧是否具有代表性")
    print("  2. 训练风格LoRA:")
    print(f"     python scripts/train_style_lora_sdxl.py \\")
    print(f"         --instance_data_dir {output_dir} \\")
    print(f"         --output_dir gen_video/models/lora/anime_style/ \\")
    print(f"         --instance_prompt 'anime style, xianxia animation style'")
    print("=" * 60)


if __name__ == "__main__":
    main()

