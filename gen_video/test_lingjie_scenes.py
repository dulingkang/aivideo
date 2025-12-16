#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试凡人修仙灵界第一章的场景
使用 Flux 生成图片，HunyuanVideo 生成视频，云卷仙音女声配音
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import AIVideoPipeline


def test_lingjie_scenes(
    episode_json: str = "lingjie/episode/1.json",
    scene_ids: Optional[List[int]] = None,
    output_name: str = "lingjie_test_scenes"
):
    """
    测试灵界第一章的指定场景
    
    Args:
        episode_json: 章节JSON文件路径（相对于项目根目录）
        scene_ids: 要测试的场景ID列表，如果为None则测试前几个场景
        output_name: 输出文件名
    """
    print("=" * 60)
    print("测试凡人修仙灵界第一章场景")
    print("=" * 60)
    
    # 加载章节JSON（相对于项目根目录 fanren）
    # project_root 是 gen_video 目录，project_root.parent 是 fanren 目录
    if Path(episode_json).is_absolute():
        episode_path = Path(episode_json)
    else:
        # 如果是相对路径，先尝试相对于 project_root.parent (fanren)
        episode_path = project_root.parent / episode_json
        # 如果路径中有 ..，需要解析
        if ".." in episode_json:
            episode_path = (project_root.parent / episode_json).resolve()
    
    if not episode_path.exists():
        print(f"❌ 文件不存在: {episode_path}")
        print(f"   项目根目录: {project_root.parent}")
        print(f"   尝试的路径: {episode_path}")
        return
    
    with open(episode_path, 'r', encoding='utf-8') as f:
        episode_data = json.load(f)
    
    all_scenes = episode_data.get('scenes', [])
    print(f"章节包含 {len(all_scenes)} 个场景")
    
    # 选择要测试的场景
    if scene_ids is None:
        # 默认测试前4个场景（id 0-3）
        scene_ids = [0, 1, 2, 3]
    
    selected_scenes = [s for s in all_scenes if s.get('id') in scene_ids]
    
    if not selected_scenes:
        print(f"❌ 没有找到场景ID: {scene_ids}")
        return
    
    print(f"\n选择测试场景: {[s['id'] for s in selected_scenes]}")
    for scene in selected_scenes:
        print(f"  场景 {scene['id']}: {scene.get('narration', '无旁白')[:30]}...")
    
    # 创建测试用的场景JSON（只包含选中的场景）
    test_data = {
        "episode": episode_data.get("episode", 1),
        "title": episode_data.get("title", "Test"),
        "scenes": selected_scenes
    }
    
    # 保存临时JSON文件
    temp_json = project_root / "temp" / f"{output_name}.json"
    temp_json.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n临时场景文件已保存: {temp_json}")
    
    # 初始化流水线
    print("\n初始化AI视频生成流水线...")
    print("  - 图像生成: Flux")
    print("  - 视频生成: HunyuanVideo")
    print("  - 配音: 云卷仙音（CosyVoice）")
    
    pipeline = AIVideoPipeline(
        config_path=project_root / "config.yaml",
        load_image=True,
        load_video=True,
        load_tts=True,
        load_subtitle=True,
        load_composer=True,
    )
    
    # 生成视频
    print("\n开始生成视频...")
    try:
        pipeline.process_script(
            script_path=str(temp_json),
            output_name=output_name
        )
        print("\n" + "=" * 60)
        print("✅ 测试完成！")
        print("=" * 60)
        output_dir = Path(pipeline.paths['output_dir']) / output_name
        final_video = output_dir / f"{output_name}_final.mp4"
        if final_video.exists():
            print(f"最终视频: {final_video}")
        else:
            print(f"输出目录: {output_dir}")
    except Exception as e:
        print(f"\n❌ 生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试凡人修仙灵界第一章场景")
    parser.add_argument(
        "--episode",
        type=str,
        default="lingjie/episode/1.json",
        help="章节JSON文件路径（相对于项目根目录 fanren）"
    )
    parser.add_argument(
        "--scenes",
        type=int,
        nargs="+",
        default=None,
        help="要测试的场景ID列表（例如: --scenes 0 1 2 3）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lingjie_test_scenes",
        help="输出文件名"
    )
    
    args = parser.parse_args()
    
    test_lingjie_scenes(
        episode_json=args.episode,
        scene_ids=args.scenes,
        output_name=args.output
    )

