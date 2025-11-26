#!/usr/bin/env python3
"""
完整工作流：narration文本 -> 检索场景 -> 匹配时长 -> 生成视频列表
"""

import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
import sys

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from search_scenes import load_index, load_scene_metadata, hybrid_search, build_keyword_index
from sentence_transformers import SentenceTransformer
import faiss

def load_narration(narration_file: Path) -> List[Dict]:
    """加载narration文件"""
    with open(narration_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持多种格式
    if 'narration_parts' in data:
        return data['narration_parts']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"无法识别的narration格式: {narration_file}")

def search_all_narrations(
    narration_parts: List[Dict],
    index_path: Path,
    metadata_path: Path,
    scene_metadata_files: List[Path],
    output_dir: Path,
    top_k: int = 5
) -> Dict[str, List[Dict]]:
    """为所有narration段落检索场景"""
    # 加载索引和模型
    index, index_metadata = load_index(index_path, metadata_path)
    
    print("加载CLIP模型...")
    clip_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 加载场景metadata
    scenes = load_scene_metadata(scene_metadata_files)
    print(f"  共 {len(scenes)} 个场景")
    
    # 建立关键词索引
    keyword_index = build_keyword_index(scenes)
    
    # 为每个narration段落检索
    search_results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n为 {len(narration_parts)} 个narration段落检索场景...")
    
    for i, part in enumerate(narration_parts, 1):
        text = part.get('text', '').strip()
        if not text:
            continue
        
        print(f"\n[{i}/{len(narration_parts)}] 检索: {text[:50]}...")
        
        # 执行混合检索
        results = hybrid_search(
            text,
            index,
            index_metadata,
            scenes,
            keyword_index,
            clip_model,
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=top_k
        )
        
        # 转换为输出格式
        output_results = []
        for scene_id, score, scene_data in results:
            output_results.append({
                "scene_id": scene_id,
                "score": float(score),
                "scene_data": scene_data
            })
        
        search_results[f"narration_{i}"] = output_results
        
        # 保存单个检索结果
        result_file = output_dir / f"narration_{i}_search.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "narration_index": i,
                "narration_text": text,
                "query": text,
                "method": "hybrid",
                "total_results": len(output_results),
                "results": output_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 找到 {len(output_results)} 个相关场景，已保存: {result_file.name}")
    
    # 保存所有检索结果
    all_results_file = output_dir / "all_search_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_narrations": len(narration_parts),
            "search_results": search_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 所有检索结果已保存: {all_results_file}")
    
    return search_results

def main():
    parser = argparse.ArgumentParser(description='narration文本 -> 检索场景完整工作流')
    parser.add_argument('--narration', '-n', required=True,
                       help='narration JSON文件')
    parser.add_argument('--index', required=True,
                       help='FAISS索引路径')
    parser.add_argument('--metadata', required=True,
                       help='索引metadata路径')
    parser.add_argument('--scenes', '-s', required=True, nargs='+',
                       help='场景metadata JSON文件（可多个）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录（保存检索结果）')
    parser.add_argument('--top-k', type=int, default=5,
                       help='每个narration检索的top-k结果（默认: 5）')
    
    args = parser.parse_args()
    
    narration_file = Path(args.narration)
    if not narration_file.exists():
        print(f"错误: narration文件不存在: {narration_file}")
        return 1
    
    # 加载narration
    narration_parts = load_narration(narration_file)
    print(f"加载narration: {len(narration_parts)} 个段落")
    
    # 检索场景
    search_results = search_all_narrations(
        narration_parts,
        Path(args.index),
        Path(args.metadata),
        [Path(f) for f in args.scenes],
        Path(args.output),
        args.top_k
    )
    
    print(f"\n✓ 工作流完成！")
    print(f"  下一步: 使用 match_scenes_to_narration.py 匹配视频时长")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

