#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景参考图像选择器

根据场景描述，从processed目录中的keyframes中检索最相关的参考图像。
使用FAISS索引和混合检索（向量+关键词）来匹配场景。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠ 警告: sentence-transformers 未安装，将只使用关键词检索")


def load_index(index_path: Path, metadata_path: Path) -> Tuple[faiss.Index, Dict]:
    """加载FAISS索引和元数据"""
    print(f"加载索引: {index_path}")
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"  索引维度: {index.d}, 向量数: {index.ntotal}")
    return index, metadata


def load_scene_metadata(metadata_files: List[Path]) -> Dict[str, Dict]:
    """加载所有场景metadata"""
    all_scenes = {}
    
    for metadata_file in metadata_files:
        if not metadata_file.exists():
            print(f"⚠ 跳过不存在的文件: {metadata_file}")
            continue
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            scenes = json.load(f)
        
        for scene_id, scene_data in scenes.items():
            all_scenes[scene_id] = scene_data
    
    return all_scenes


def build_keyword_index(scenes: Dict[str, Dict]) -> Dict[str, List[str]]:
    """构建关键词索引（TF-IDF风格）"""
    keyword_index = {}
    
    for scene_id, scene_data in scenes.items():
        # 组合所有文本字段
        text_parts = []
        if scene_data.get("text"):
            text_parts.append(scene_data["text"])
        if scene_data.get("visual_caption"):
            text_parts.append(scene_data["visual_caption"])
        if scene_data.get("subtitle_text"):
            text_parts.append(scene_data["subtitle_text"])
        
        combined_text = " ".join(text_parts).lower()
        
        # 提取关键词（简单分词）
        keywords = combined_text.split()
        
        for keyword in keywords:
            if len(keyword) > 1:  # 忽略单字符
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(scene_id)
    
    return keyword_index


def keyword_search(query: str, keyword_index: Dict[str, List[str]], scenes: Dict,
                   top_k: int = 10) -> List[Tuple[str, float]]:
    """关键词检索"""
    query_lower = query.lower()
    query_keywords = query_lower.split()
    
    scene_scores = {}
    
    for keyword in query_keywords:
        if keyword in keyword_index:
            for scene_id in keyword_index[keyword]:
                scene_scores[scene_id] = scene_scores.get(scene_id, 0) + 1
    
    # 归一化分数
    if scene_scores:
        max_score = max(scene_scores.values())
        if max_score > 0:
            scene_scores = {k: v / max_score for k, v in scene_scores.items()}
    
    sorted_results = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def vector_search(query: str, index: faiss.Index, metadata: Dict,
                 clip_model: SentenceTransformer, top_k: int = 10) -> List[Tuple[int, float]]:
    """向量检索（基于embedding相似度）"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return []
    
    # 将查询文本编码为向量
    query_embedding = clip_model.encode([query], convert_to_numpy=True).astype('float32')
    
    # 搜索
    distances, indices = index.search(query_embedding, top_k)
    
    # 转换为列表
    results = [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]
    return results


def hybrid_search(query: str, index: faiss.Index, metadata: Dict, scenes: Dict,
                 keyword_index: Dict[str, List[str]], clip_model: Optional[SentenceTransformer],
                 vector_weight: float = 0.7, keyword_weight: float = 0.3,
                 top_k: int = 10) -> List[Tuple[str, float, Dict]]:
    """混合检索：结合向量检索和关键词检索"""
    scene_scores = {}
    
    # 1. 向量检索
    if SENTENCE_TRANSFORMERS_AVAILABLE and clip_model is not None:
        vector_results = vector_search(query, index, metadata, clip_model, top_k=top_k * 2)
        
        if vector_results:
            max_distance = max(d for _, d in vector_results) if vector_results else 1.0
            min_distance = min(d for _, d in vector_results) if vector_results else 0.0
            distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
            
            for idx_pos, distance in vector_results:
                scene_id = metadata["id_mapping"].get(str(idx_pos), "")
                if scene_id:
                    normalized_score = 1.0 - ((distance - min_distance) / distance_range) if distance_range > 0 else 1.0
                    scene_scores[scene_id] = scene_scores.get(scene_id, 0.0) + normalized_score * vector_weight
    
    # 2. 关键词检索
    keyword_results = keyword_search(query, keyword_index, scenes, top_k=top_k * 2)
    
    if keyword_results:
        max_keyword_score = max(s for _, s in keyword_results) if keyword_results else 1.0
        for scene_id, score in keyword_results:
            normalized_score = score / max_keyword_score if max_keyword_score > 0 else score
            scene_scores[scene_id] = scene_scores.get(scene_id, 0.0) + normalized_score * keyword_weight
    
    # 3. 排序并返回
    sorted_results = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
    results = [(scene_id, final_score, scenes.get(scene_id, {})) 
               for scene_id, final_score in sorted_results[:top_k] if scene_id in scenes]
    
    return results


def find_keyframe_path(scene_id: str, base_dir: Path) -> Optional[Path]:
    """根据scene_id查找对应的keyframe图像路径"""
    # scene_id格式: "171_scene_001" 或 "episode_171_scene_001"
    parts = scene_id.split('_')
    
    if len(parts) >= 2:
        episode_num = parts[0]
        scene_num = parts[-1] if len(parts) > 2 else parts[1]
        
        # 尝试不同的路径格式
        possible_paths = [
            base_dir / f"episode_{episode_num}" / "keyframes" / f"episode_{episode_num}_clean-Scene-{scene_num.zfill(3)}_middle.jpg",
            base_dir / f"episode_{episode_num}" / "keyframes" / f"episode_{episode_num}_clean-Scene-{scene_num.zfill(3)}_start.jpg",
            base_dir / f"episode_{episode_num}" / "keyframes" / f"scene_{scene_num.zfill(3)}_middle.jpg",
            base_dir / f"episode_{episode_num}" / "keyframes" / f"scene_{scene_num.zfill(3)}_start.jpg",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
    
    return None


def build_scene_query(scene: Dict[str, Any]) -> str:
    """根据场景数据构建查询文本"""
    query_parts = []
    
    # 1. 场景描述
    if scene.get("description"):
        query_parts.append(scene["description"])
    
    # 2. visual.composition
    if scene.get("visual", {}).get("composition"):
        query_parts.append(scene["visual"]["composition"])
    
    # 3. visual.environment
    if scene.get("visual", {}).get("environment"):
        query_parts.append(scene["visual"]["environment"])
    
    # 4. visual.character_pose
    if scene.get("visual", {}).get("character_pose"):
        query_parts.append(scene["visual"]["character_pose"])
    
    # 5. 原始prompt（如果存在）
    if scene.get("prompt"):
        query_parts.append(scene["prompt"])
    
    return " ".join(query_parts)


def select_reference_images(
    scene: Dict[str, Any],
    index_path: Path,
    metadata_path: Path,
    scene_metadata_files: List[Path],
    keyframes_base_dir: Path,
    top_k: int = 3,
    method: str = "hybrid"
) -> List[Tuple[Path, float, Dict]]:
    """
    为场景选择参考图像
    
    Returns:
        [(keyframe_path, score, scene_data), ...] 按分数降序排列
    """
    # 构建查询文本
    query = build_scene_query(scene)
    
    if not query:
        print("  ⚠ 场景没有足够的描述信息，无法检索")
        return []
    
    print(f"  🔍 场景查询: {query[:100]}...")
    
    # 加载索引
    index, index_metadata = load_index(index_path, metadata_path)
    
    # 加载场景metadata
    all_scenes = load_scene_metadata(scene_metadata_files)
    
    # 加载CLIP模型（如果需要）
    clip_model = None
    if method in ['vector', 'hybrid'] and SENTENCE_TRANSFORMERS_AVAILABLE:
        print("  加载CLIP模型...")
        clip_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 构建关键词索引（如果需要）
    keyword_index = None
    if method in ['keyword', 'hybrid']:
        keyword_index = build_keyword_index(all_scenes)
    
    # 执行检索
    if method == 'vector':
        vector_results = vector_search(query, index, index_metadata, clip_model, top_k=top_k)
        results = []
        for idx_pos, distance in vector_results:
            scene_id = index_metadata["id_mapping"].get(str(idx_pos), "")
            if scene_id in all_scenes:
                keyframe_path = find_keyframe_path(scene_id, keyframes_base_dir)
                if keyframe_path:
                    score = 1.0 / (1.0 + distance)
                    results.append((keyframe_path, score, all_scenes[scene_id]))
    
    elif method == 'keyword':
        keyword_results = keyword_search(query, keyword_index, all_scenes, top_k=top_k)
        results = []
        for scene_id, score in keyword_results:
            keyframe_path = find_keyframe_path(scene_id, keyframes_base_dir)
            if keyframe_path:
                results.append((keyframe_path, score, all_scenes[scene_id]))
    
    else:  # hybrid
        hybrid_results = hybrid_search(
            query, index, index_metadata, all_scenes, keyword_index, clip_model,
            vector_weight=0.7, keyword_weight=0.3, top_k=top_k
        )
        results = []
        for scene_id, score, scene_data in hybrid_results:
            keyframe_path = find_keyframe_path(scene_id, keyframes_base_dir)
            if keyframe_path:
                results.append((keyframe_path, score, scene_data))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='为场景选择参考图像')
    parser.add_argument('--scene', required=True, help='场景JSON文件或JSON字符串')
    parser.add_argument('--index', default='processed/global_index.faiss', help='FAISS索引路径')
    parser.add_argument('--metadata', default='processed/index_metadata.json', help='索引metadata路径')
    parser.add_argument('--scenes', nargs='+', default=['processed/episode_*/scene_metadata.json'],
                       help='场景metadata JSON文件（支持glob模式）')
    parser.add_argument('--keyframes-base', default='processed', help='keyframes基础目录')
    parser.add_argument('--top-k', type=int, default=3, help='返回top k个参考图像')
    parser.add_argument('--method', choices=['vector', 'keyword', 'hybrid'], 
                       default='hybrid', help='检索方法')
    parser.add_argument('--output', help='输出JSON文件路径（可选）')
    
    args = parser.parse_args()
    
    # 解析场景数据
    scene_path = Path(args.scene)
    if scene_path.exists():
        with open(scene_path, 'r', encoding='utf-8') as f:
            scene = json.load(f)
    else:
        # 尝试作为JSON字符串解析
        scene = json.loads(args.scene)
    
    # 展开glob模式
    from glob import glob
    scene_metadata_files = []
    for pattern in args.scenes:
        scene_metadata_files.extend([Path(f) for f in glob(pattern)])
    
    # 选择参考图像
    results = select_reference_images(
        scene,
        Path(args.index),
        Path(args.metadata),
        scene_metadata_files,
        Path(args.keyframes_base),
        top_k=args.top_k,
        method=args.method
    )
    
    # 输出结果
    print(f"\n找到 {len(results)} 个参考图像:\n")
    output_data = []
    for i, (keyframe_path, score, scene_data) in enumerate(results, 1):
        print(f"{i}. {keyframe_path.name}")
        print(f"   相似度: {score:.3f}")
        print(f"   场景ID: {scene_data.get('scene_id', 'unknown')}")
        print(f"   描述: {scene_data.get('text', '')[:50]}...")
        print()
        
        output_data.append({
            "keyframe_path": str(keyframe_path),
            "score": score,
            "scene_data": scene_data
        })
    
    # 保存到文件（如果指定）
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

