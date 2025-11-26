#!/usr/bin/env python3
"""
建立跨集检索索引：使用FAISS建立全局场景索引
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: faiss-cpu 或 faiss-gpu 未安装")

def load_scene_metadata(metadata_files: List[Path]) -> Dict:
    """加载所有集的场景metadata"""
    all_scenes = {}
    
    for metadata_file in metadata_files:
        if not metadata_file.exists():
            print(f"警告: 文件不存在，跳过: {metadata_file}")
            continue
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {metadata_file}")
            print(f"   错误位置: 第 {e.lineno} 行, 第 {e.colno} 列")
            print(f"   错误信息: {e.msg}")
            # 显示错误附近的上下文
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    start_line = max(0, e.lineno - 3)
                    end_line = min(len(lines), e.lineno + 2)
                    print(f"   上下文:")
                    for i in range(start_line, end_line):
                        marker = ">>> " if i == e.lineno - 1 else "    "
                        print(f"   {marker}{i+1:5d}: {lines[i].rstrip()}")
            except:
                pass
            continue
        except Exception as e:
            print(f"❌ 读取文件失败: {metadata_file}")
            print(f"   错误: {e}")
            continue
        
        # 提取集号
        episode_id = metadata_file.parent.name.split('_')[-1] if '_' in metadata_file.parent.name else 'unknown'
        
        for scene_id, scene_data in episode_data.items():
            # 生成全局唯一ID
            global_id = f"{episode_id}_{scene_id}"
            all_scenes[global_id] = {
                **scene_data,
                "episode_id": episode_id,
                "scene_id": scene_id,
                "global_id": global_id
            }
    
    return all_scenes

def build_faiss_index(scenes: Dict, output_index_path: Path, output_metadata_path: Path):
    """建立FAISS索引"""
    if not FAISS_AVAILABLE:
        raise ImportError("需要安装: pip install faiss-cpu 或 faiss-gpu")
    
    # 收集所有embeddings
    embeddings = []
    scene_ids = []
    skipped_scenes = []
    
    # 先统计所有embedding的长度，找出最常见的长度
    embedding_lengths = {}
    for scene_id, scene_data in scenes.items():
        if scene_data.get("embedding") is not None:
            length = len(scene_data["embedding"])
            embedding_lengths[length] = embedding_lengths.get(length, 0) + 1
    
    if not embedding_lengths:
        print("错误: 没有找到有效的embedding")
        return False
    
    # 找出最常见的embedding长度（期望维度）
    expected_dimension = max(embedding_lengths.items(), key=lambda x: x[1])[0]
    print(f"期望的embedding维度: {expected_dimension}")
    if len(embedding_lengths) > 1:
        print(f"警告: 发现不同长度的embedding: {embedding_lengths}")
        print(f"  将跳过长度不等于 {expected_dimension} 的场景")
    
    # 收集有效的embeddings
    for scene_id, scene_data in scenes.items():
        embedding = scene_data.get("embedding")
        if embedding is not None:
            if len(embedding) == expected_dimension:
                embeddings.append(embedding)
                scene_ids.append(scene_id)
            else:
                skipped_scenes.append((scene_id, len(embedding)))
    
    if len(embeddings) == 0:
        print("错误: 没有找到有效的embedding")
        return False
    
    if skipped_scenes:
        print(f"警告: 跳过了 {len(skipped_scenes)} 个embedding长度不正确的场景:")
        for scene_id, length in skipped_scenes[:10]:  # 只显示前10个
            print(f"  {scene_id}: 长度 {length} (期望 {expected_dimension})")
        if len(skipped_scenes) > 10:
            print(f"  ... 还有 {len(skipped_scenes) - 10} 个场景被跳过")
    
    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]
    
    print(f"建立FAISS索引: {len(embeddings)} 个场景, 维度 {dimension}")
    
    # 创建FAISS索引（使用L2距离）
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 保存索引
    faiss.write_index(index, str(output_index_path))
    print(f"✓ 索引已保存: {output_index_path}")
    
    # 保存metadata映射（索引位置 -> scene_id）
    id_mapping = {str(i): scene_ids[i] for i in range(len(scene_ids))}
    
    # 保存完整的场景数据（用于检索时快速访问）
    scenes_data = {scene_ids[i]: scenes[scene_ids[i]] for i in range(len(scene_ids))}
    
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "id_mapping": id_mapping,  # 索引位置 -> scene_id
            "scenes": scenes_data,    # 完整场景数据（可选，如果文件太大可以移除）
            "dimension": dimension,
            "total_scenes": len(scene_ids)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Metadata已保存: {output_metadata_path}")
    
    return True

def expand_glob_patterns(patterns: List[str]) -> List[Path]:
    """展开通配符模式，返回文件路径列表"""
    files = []
    for pattern in patterns:
        # 使用glob展开通配符
        matched = glob.glob(pattern, recursive=True)
        if matched:
            files.extend([Path(f) for f in matched])
        else:
            # 如果没有匹配，尝试作为普通路径
            p = Path(pattern)
            if p.exists():
                files.append(p)
            else:
                print(f"警告: 模式未匹配到文件: {pattern}")
    # 去重并排序
    return sorted(set(files))

def main():
    parser = argparse.ArgumentParser(description='建立跨集场景检索索引')
    parser.add_argument('--input', '-i', required=True, nargs='+',
                       help='场景metadata JSON文件（可多个，支持通配符如 *.json）')
    parser.add_argument('--index', required=True, help='输出FAISS索引路径')
    parser.add_argument('--metadata', required=True, help='输出metadata映射路径')
    
    args = parser.parse_args()
    
    # 展开通配符
    metadata_files = expand_glob_patterns(args.input)
    
    if not metadata_files:
        print("错误: 没有找到任何匹配的文件")
        return 1
    
    print(f"加载 {len(metadata_files)} 个metadata文件...")
    scenes = load_scene_metadata(metadata_files)
    print(f"  共 {len(scenes)} 个场景")
    
    output_index = Path(args.index)
    output_metadata = Path(args.metadata)
    
    if build_faiss_index(scenes, output_index, output_metadata):
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())

