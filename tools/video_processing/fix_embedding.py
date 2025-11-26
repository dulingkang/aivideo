#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复场景 metadata 中不完整的 embedding
"""

import json
import sys
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("错误: 需要安装 sentence-transformers")
    print("请运行: pip install sentence-transformers")
    sys.exit(1)


def fix_scene_embedding(metadata_file: Path, scene_id: str, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
    """
    修复指定场景的 embedding
    
    Args:
        metadata_file: metadata JSON 文件路径
        scene_id: 场景ID（如 "scene_024"）
        model_name: embedding 模型名称
    """
    print(f"加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"读取文件: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if scene_id not in data:
        print(f"错误: 场景 {scene_id} 不存在")
        return False
    
    scene = data[scene_id]
    current_embedding = scene.get("embedding")
    current_length = len(current_embedding) if current_embedding else 0
    
    print(f"当前 embedding 长度: {current_length}")
    
    # 获取文本描述（优先使用 text，否则使用 visual_caption + subtitle_text）
    text = scene.get("text")
    if not text:
        visual_caption = scene.get("visual_caption", "")
        subtitle_text = scene.get("subtitle_text", "")
        text = f"{visual_caption} {subtitle_text}".strip()
    
    if not text:
        print(f"错误: 场景 {scene_id} 没有可用的文本描述")
        return False
    
    print(f"使用文本: {text[:100]}...")
    
    # 生成新的 embedding
    print("生成新的 embedding...")
    new_embedding = model.encode(text, convert_to_numpy=True).tolist()
    new_length = len(new_embedding)
    
    print(f"新 embedding 长度: {new_length}")
    
    # 更新 embedding
    scene["embedding"] = new_embedding
    
    # 保存文件
    print(f"保存文件: {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 修复完成: {scene_id} (长度: {current_length} → {new_length})")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='修复场景 metadata 中不完整的 embedding')
    parser.add_argument('--input', '-i', required=True, help='metadata JSON 文件路径')
    parser.add_argument('--scene', '-s', required=True, help='场景ID（如 scene_024）')
    parser.add_argument('--model', '-m', default='paraphrase-multilingual-MiniLM-L12-v2',
                       help='embedding 模型名称（默认: paraphrase-multilingual-MiniLM-L12-v2）')
    
    args = parser.parse_args()
    
    metadata_file = Path(args.input)
    if not metadata_file.exists():
        print(f"错误: 文件不存在: {metadata_file}")
        return 1
    
    if fix_scene_embedding(metadata_file, args.scene, args.model):
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

