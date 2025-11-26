#!/usr/bin/env python3
"""
场景检索脚本：支持向量检索和关键词检索的混合检索
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: faiss-cpu 或 faiss-gpu 未安装")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 未安装")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("警告: jieba 未安装，关键词检索将使用简单匹配")

def load_index(index_path: Path, metadata_path: Path) -> Tuple[faiss.Index, Dict]:
    """加载FAISS索引和metadata"""
    if not FAISS_AVAILABLE:
        raise ImportError("需要安装: pip install faiss-cpu 或 faiss-gpu")
    
    if not index_path.exists():
        raise FileNotFoundError(f"索引文件不存在: {index_path}")
    
    index = faiss.read_index(str(index_path))
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return index, metadata

def load_scene_metadata(metadata_files: List[Path]) -> Dict:
    """加载所有场景的完整metadata"""
    all_scenes = {}
    
    for metadata_file in metadata_files:
        if not metadata_file.exists():
            print(f"警告: 文件不存在，跳过: {metadata_file}")
            continue
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            episode_data = json.load(f)
        
        # 提取集号
        episode_id = metadata_file.parent.name.split('_')[-1] if '_' in metadata_file.parent.name else 'unknown'
        
        for scene_id, scene_data in episode_data.items():
            global_id = f"{episode_id}_{scene_id}"
            all_scenes[global_id] = {
                **scene_data,
                "episode_id": episode_id,
                "scene_id": scene_id,
                "global_id": global_id
            }
    
    return all_scenes

def tokenize_chinese(text: str) -> List[str]:
    """中文分词"""
    if JIEBA_AVAILABLE:
        return list(jieba.cut(text))
    else:
        # 简单分词：按字符分割（不推荐，但作为fallback）
        return list(text)

def build_keyword_index(scenes: Dict, use_subtitle: bool = True) -> Dict[str, List[str]]:
    """
    建立关键词倒排索引（用于关键词检索）
    
    Args:
        scenes: 场景数据字典
        use_subtitle: 是否使用字幕文本（默认True，用于开头/结尾场景查找时设为False，只用场景描述）
    
    Returns:
        keyword -> [scene_id1, scene_id2, ...] 的映射
    """
    keyword_index = {}
    
    for scene_id, scene_data in scenes.items():
        # 提取文本字段
        if use_subtitle:
            # 使用所有文本字段（包括字幕）
            text_fields = [
                scene_data.get("text", ""),
                scene_data.get("subtitle_text", ""),
                scene_data.get("visual_caption", ""),
            ]
        else:
            # 只使用场景描述（不包括字幕）
            # 检查 text 是否是字幕：如果 text == subtitle_text，则 text 是字幕，不使用
            text_value = scene_data.get("text", "")
            subtitle_value = scene_data.get("subtitle_text", "")
            
            text_fields = [
                scene_data.get("caption", ""),  # 场景描述
                scene_data.get("visual_caption", ""),  # 视觉描述（英文）
            ]
            # 只有当 text 不等于 subtitle_text 时，才使用 text（说明 text 是场景描述）
            if text_value and text_value != subtitle_value:
                text_fields.append(text_value)
        
        # 合并并分词
        combined_text = " ".join(filter(None, text_fields))
        if not combined_text:
            continue
        
        # 分词并去重
        tokens = set(tokenize_chinese(combined_text.lower()))
        
        # 建立倒排索引
        for token in tokens:
            if len(token.strip()) > 0:  # 过滤空字符串
                if token not in keyword_index:
                    keyword_index[token] = []
                keyword_index[token].append(scene_id)
    
    return keyword_index

def keyword_search(query: str, keyword_index: Dict[str, List[str]], scenes: Dict, 
                   top_k: int = 10, use_subtitle: bool = True) -> List[Tuple[str, float]]:
    """
    关键词检索（基于TF-IDF权重）
    
    Args:
        query: 查询文本
        keyword_index: 关键词索引
        scenes: 场景数据字典
        top_k: 返回top k个结果
        use_subtitle: 是否使用字幕文本（默认True，用于开头/结尾场景查找时设为False，只用场景描述）
    
    Returns:
        [(scene_id, score), ...] 按分数降序排列
    """
    # 查询分词
    query_tokens = set(tokenize_chinese(query.lower()))
    
    # 计算每个场景的TF-IDF分数
    scene_scores = {}
    
    # 计算IDF（逆文档频率）
    total_scenes = len(scenes)
    token_idf = {}
    for token in query_tokens:
        if token in keyword_index:
            # IDF = log(总文档数 / 包含该词的文档数)
            doc_freq = len(set(keyword_index[token]))
            token_idf[token] = np.log(total_scenes / (doc_freq + 1))
        else:
            token_idf[token] = 0
    
    # 计算每个场景的TF-IDF分数
    for scene_id, scene_data in scenes.items():
        if use_subtitle:
            # 使用所有文本字段（包括字幕）
            text_fields = [
                scene_data.get("text", ""),
                scene_data.get("subtitle_text", ""),
                scene_data.get("visual_caption", ""),
            ]
        else:
            # 只使用场景描述（不包括字幕）
            # 检查 text 是否是字幕：如果 text == subtitle_text，则 text 是字幕，不使用
            text_value = scene_data.get("text", "")
            subtitle_value = scene_data.get("subtitle_text", "")
            
            text_fields = [
                scene_data.get("caption", ""),  # 场景描述
                scene_data.get("visual_caption", ""),  # 视觉描述（英文）
            ]
            # 只有当 text 不等于 subtitle_text 时，才使用 text（说明 text 是场景描述）
            if text_value and text_value != subtitle_value:
                text_fields.append(text_value)
        combined_text = " ".join(filter(None, text_fields)).lower()
        
        if not combined_text:
            continue
        
        scene_tokens = tokenize_chinese(combined_text)
        token_tf = {}
        
        # 计算TF（词频）
        for token in scene_tokens:
            token_tf[token] = token_tf.get(token, 0) + 1
        
        # 归一化TF（除以文档总词数）
        total_tokens = len(scene_tokens)
        if total_tokens > 0:
            for token in token_tf:
                token_tf[token] = token_tf[token] / total_tokens
        
        # 计算TF-IDF分数
        score = 0.0
        for token in query_tokens:
            if token in token_tf and token in token_idf:
                score += token_tf[token] * token_idf[token]
        
        if score > 0:
            scene_scores[scene_id] = score
    
    # 按分数排序
    sorted_results = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

def vector_search(query: str, index: faiss.Index, metadata: Dict, 
                 clip_model: SentenceTransformer, top_k: int = 10) -> List[Tuple[int, float]]:
    """
    向量检索（基于embedding相似度）
    
    Returns:
        [(index_position, distance), ...] 按距离升序排列（距离越小越相似）
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("需要安装: pip install sentence-transformers")
    
    # 将查询文本编码为向量
    query_embedding = clip_model.encode([query], convert_to_numpy=True).astype('float32')
    
    # 搜索
    distances, indices = index.search(query_embedding, top_k)
    
    # 转换为列表
    results = [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]
    return results

def hybrid_search(query: str, index: faiss.Index, metadata: Dict, scenes: Dict,
                 keyword_index: Dict[str, List[str]], clip_model: SentenceTransformer,
                 vector_weight: float = 0.7, keyword_weight: float = 0.3,
                 top_k: int = 10, use_subtitle: bool = True) -> List[Tuple[str, float, Dict]]:
    """
    混合检索：结合向量检索和关键词检索
    
    Args:
        query: 查询文本
        vector_weight: 向量检索权重（0-1）
        keyword_weight: 关键词检索权重（0-1）
        top_k: 返回top k个结果
    
    Returns:
        [(scene_id, final_score, scene_data), ...] 按分数降序排列
    """
    # 1. 向量检索
    vector_results = vector_search(query, index, metadata, clip_model, top_k=top_k * 2)
    
    # 2. 关键词检索
    keyword_results = keyword_search(query, keyword_index, scenes, top_k=top_k * 2, use_subtitle=use_subtitle)
    
    # 3. 归一化分数并合并
    scene_scores = {}
    
    # 归一化向量检索分数（距离越小分数越高，转换为0-1范围）
    if vector_results:
        max_distance = max(d for _, d in vector_results) if vector_results else 1.0
        min_distance = min(d for _, d in vector_results) if vector_results else 0.0
        distance_range = max_distance - min_distance if max_distance > min_distance else 1.0
        
        for idx_pos, distance in vector_results:
            scene_id = metadata["id_mapping"].get(str(idx_pos), "")
            if scene_id:
                # 将距离转换为相似度分数（0-1）
                normalized_score = 1.0 - ((distance - min_distance) / distance_range) if distance_range > 0 else 1.0
                scene_scores[scene_id] = scene_scores.get(scene_id, 0.0) + normalized_score * vector_weight
    
    # 归一化关键词检索分数
    if keyword_results:
        max_score = max(s for _, s in keyword_results) if keyword_results else 1.0
        min_score = min(s for _, s in keyword_results) if keyword_results else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        for scene_id, score in keyword_results:
            normalized_score = (score - min_score) / score_range if score_range > 0 else score
            scene_scores[scene_id] = scene_scores.get(scene_id, 0.0) + normalized_score * keyword_weight
    
    # 4. 按最终分数排序
    sorted_results = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 5. 返回结果（包含完整场景数据）
    results = []
    for scene_id, final_score in sorted_results[:top_k]:
        if scene_id in scenes:
            results.append((scene_id, final_score, scenes[scene_id]))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='场景检索：支持向量检索和关键词检索')
    parser.add_argument('--query', '-q', required=True, help='查询文本')
    parser.add_argument('--index', required=True, help='FAISS索引路径')
    parser.add_argument('--metadata', required=True, help='索引metadata路径')
    parser.add_argument('--scenes', '-s', required=True, nargs='+',
                       help='场景metadata JSON文件（可多个）')
    parser.add_argument('--top-k', type=int, default=10, help='返回top k个结果')
    parser.add_argument('--method', choices=['vector', 'keyword', 'hybrid'], 
                       default='hybrid', help='检索方法')
    parser.add_argument('--vector-weight', type=float, default=0.7,
                       help='混合检索中向量检索的权重（0-1）')
    parser.add_argument('--keyword-weight', type=float, default=0.3,
                       help='混合检索中关键词检索的权重（0-1）')
    parser.add_argument('--output', '-o', help='输出JSON文件路径（可选）')
    
    args = parser.parse_args()
    
    # 加载索引
    print(f"加载索引: {args.index}")
    index, index_metadata = load_index(Path(args.index), Path(args.metadata))
    
    # 加载场景metadata
    print(f"加载场景metadata...")
    metadata_files = [Path(f) for f in args.scenes]
    scenes = load_scene_metadata(metadata_files)
    print(f"  共 {len(scenes)} 个场景")
    
    # 加载CLIP模型（用于向量检索）
    if args.method in ['vector', 'hybrid']:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("向量检索需要安装: pip install sentence-transformers")
        print("加载CLIP模型...")
        clip_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 建立关键词索引（用于关键词检索）
    keyword_index = None
    if args.method in ['keyword', 'hybrid']:
        print("建立关键词索引...")
        keyword_index = build_keyword_index(scenes)
        print(f"  索引了 {len(keyword_index)} 个关键词")
    
    # 执行检索
    print(f"\n查询: {args.query}")
    print(f"方法: {args.method}")
    
    if args.method == 'vector':
        results = vector_search(args.query, index, index_metadata, clip_model, top_k=args.top_k)
        # 转换为场景ID和完整数据
        output_results = []
        for idx_pos, distance in results:
            scene_id = index_metadata["id_mapping"].get(str(idx_pos), "")
            if scene_id in scenes:
                output_results.append({
                    "scene_id": scene_id,
                    "score": 1.0 / (1.0 + distance),  # 转换为相似度分数
                    "distance": distance,
                    "scene_data": scenes[scene_id]
                })
    
    elif args.method == 'keyword':
        results = keyword_search(args.query, keyword_index, scenes, top_k=args.top_k, use_subtitle=True)
        output_results = []
        for scene_id, score in results:
            output_results.append({
                "scene_id": scene_id,
                "score": score,
                "scene_data": scenes[scene_id]
            })
    
    else:  # hybrid
        results = hybrid_search(
            args.query, index, index_metadata, scenes, keyword_index, clip_model,
            vector_weight=args.vector_weight, keyword_weight=args.keyword_weight,
            top_k=args.top_k, use_subtitle=True  # 默认使用字幕
        )
        output_results = []
        for scene_id, final_score, scene_data in results:
            output_results.append({
                "scene_id": scene_id,
                "score": final_score,
                "scene_data": scene_data
            })
    
    # 输出结果
    print(f"\n找到 {len(output_results)} 个相关场景:\n")
    for i, result in enumerate(output_results, 1):
        scene_data = result["scene_data"]
        print(f"{i}. [{result['scene_id']}] 分数: {result['score']:.4f}")
        print(f"   集数: {scene_data.get('episode_id', 'unknown')}")
        print(f"   描述: {scene_data.get('text', '无描述')[:100]}...")
        if scene_data.get('subtitle_text'):
            print(f"   字幕: {scene_data.get('subtitle_text', '')[:100]}...")
        print()
    
    # 保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                "query": args.query,
                "method": args.method,
                "total_results": len(output_results),
                "results": output_results
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存: {args.output}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

