#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ComfyUI AnimateDiff 多个场景
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from gen_video.comfyui_integration import test_comfyui_connection
from gen_video.comfyui_animatediff_api import ComfyUIAnimateDiffAPI


def load_json_script(json_path: Path) -> List[Dict[str, Any]]:
    """加载 JSON 脚本并提取场景"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get('scenes', [])
    return scenes


def find_scene_image(scene_num: int, base_dir: Path) -> Path:
    """查找场景对应的图像"""
    # 尝试多个可能的路径
    possible_paths = [
        base_dir / f"scene_{scene_num:03d}.png",
        base_dir / f"scene_{scene_num:03d}.jpg",
        base_dir / f"scene_{scene_num}.png",
        base_dir / f"scene_{scene_num}.jpg",
        base_dir / f"scene_{scene_num:03d}" / "image.png",
    ]
    
    # 也尝试在 outputs 目录中查找
    outputs_base = Path("/vepfs-dev/shawn/vid/fanren/gen_video/outputs")
    possible_output_paths = [
        outputs_base / "output" / "images" / f"scene_{scene_num:03d}.png",
        outputs_base / "images" / "lingjie_ep9_full" / f"scene_{scene_num:03d}.png",
        outputs_base / "images" / "lingjie_ep5_full" / f"scene_{scene_num:03d}.png",
        outputs_base / "images" / "lingjie_ep2_full" / f"scene_{scene_num:03d}.png",
    ]
    
    all_paths = possible_paths + possible_output_paths
    
    for path in all_paths:
        if path.exists():
            return path
    
    return None


def build_prompt_from_scene(scene: Dict[str, Any]) -> tuple:
    """从场景数据构建 prompt"""
    # 基础风格
    base_style = "anime style, xianxia fantasy, cinematic lighting"
    
    # 场景描述
    description = scene.get('description', '')
    visual = scene.get('visual', {})
    composition = visual.get('composition', '')
    
    # 构建完整 prompt
    prompt_parts = [base_style]
    
    if description:
        prompt_parts.append(description)
    
    if composition:
        prompt_parts.append(composition)
    
    full_prompt = ", ".join(filter(None, prompt_parts))
    
    # 负面 prompt
    negative_prompt = "blurry, low quality, noise, deformed, distorted, bad anatomy, text, watermark"
    
    return full_prompt, negative_prompt


def test_multiple_scenes(
    json_path: str,
    max_scenes: int = 5,
    start_scene: int = 1,
    output_dir: str = None,
):
    """测试多个场景"""
    print("=" * 60)
    print("ComfyUI AnimateDiff 多场景测试")
    print("=" * 60)
    
    # 检查连接
    print("\n[1] 检查 ComfyUI 连接...")
    if not test_comfyui_connection():
        print("✗ ComfyUI 服务器未运行")
        print("  请先启动: bash gen_video/启动ComfyUI服务器.sh")
        return False
    
    print("✓ ComfyUI 服务器连接成功")
    
    # 加载 JSON 脚本
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"✗ JSON 文件不存在: {json_path}")
        return False
    
    print(f"\n[2] 加载 JSON 脚本: {json_path}")
    scenes = load_json_script(json_path)
    print(f"✓ 找到 {len(scenes)} 个场景")
    
    # 确定测试范围
    end_scene = min(start_scene + max_scenes - 1, len(scenes))
    test_scenes = scenes[start_scene-1:end_scene]
    
    print(f"\n[3] 测试场景范围: {start_scene} - {end_scene} (共 {len(test_scenes)} 个场景)")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path("/vepfs-dev/shawn/vid/fanren/gen_video/outputs/comfyui_test")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 API 客户端
    api = ComfyUIAnimateDiffAPI()
    
    # 测试每个场景
    results = []
    for idx, scene in enumerate(test_scenes, start=start_scene):
        scene_num = idx
        print(f"\n{'='*60}")
        print(f"测试场景 {scene_num}/{len(scenes)}")
        print(f"{'='*60}")
        
        # 查找场景图像
        print(f"\n[场景 {scene_num}] 查找场景图像...")
        scene_image = find_scene_image(scene_num, json_path.parent)
        
        if scene_image is None:
            print(f"  ⚠ 未找到场景 {scene_num} 的图像，跳过")
            results.append({
                "scene_num": scene_num,
                "status": "skipped",
                "reason": "image_not_found"
            })
            continue
        
        print(f"  ✓ 找到图像: {scene_image}")
        
        # 构建 prompt
        prompt, negative_prompt = build_prompt_from_scene(scene)
        print(f"  [Prompt] {prompt[:80]}...")
        
        # 生成视频
        try:
            print(f"  [场景 {scene_num}] 开始生成视频...")
            frames = api.generate_video_from_image(
                image_path=str(scene_image),
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=16,
                width=1280,  # 进一步提升分辨率以改善清晰度（SDXL推荐分辨率）
                height=1280,  # 进一步提升分辨率以改善清晰度（SDXL推荐分辨率）
                output_dir=str(output_dir / f"scene_{scene_num:03d}"),
                timeout=900,  # 15分钟超时（Refiner需要更多时间）
            )
            
            print(f"  ✓ 场景 {scene_num} 生成成功: {len(frames)} 帧")
            results.append({
                "scene_num": scene_num,
                "status": "success",
                "num_frames": len(frames),
                "output_dir": str(output_dir / f"scene_{scene_num:03d}")
            })
            
        except Exception as e:
            print(f"  ✗ 场景 {scene_num} 生成失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "scene_num": scene_num,
                "status": "error",
                "error": str(e)
            })
    
    # 打印总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = sum(1 for r in results if r.get("status") == "error")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    
    print(f"总场景数: {len(test_scenes)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"跳过: {skipped_count}")
    print(f"\n输出目录: {output_dir}")
    
    # 打印详细结果
    print(f"\n详细结果:")
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗" if r["status"] == "error" else "⚠"
        print(f"  {status_icon} 场景 {r['scene_num']}: {r['status']}")
        if r["status"] == "success":
            print(f"      - 帧数: {r.get('num_frames', 'N/A')}")
            print(f"      - 输出: {r.get('output_dir', 'N/A')}")
        elif r["status"] == "error":
            print(f"      - 错误: {r.get('error', 'N/A')}")
    
    return success_count > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 ComfyUI AnimateDiff 多个场景")
    parser.add_argument(
        "--script",
        type=str,
        default="lingjie/1.json",
        help="JSON 脚本路径"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=5,
        help="最大测试场景数"
    )
    parser.add_argument(
        "--start-scene",
        type=int,
        default=1,
        help="起始场景编号（从1开始）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: outputs/comfyui_test）"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = Path(__file__).parent.parent / script_path
    
    test_multiple_scenes(
        json_path=str(script_path),
        max_scenes=args.max_scenes,
        start_scene=args.start_scene,
        output_dir=args.output_dir,
    )

