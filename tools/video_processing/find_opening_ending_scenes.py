#!/usr/bin/env python3
"""
查找适合作为开头和结尾的视频场景
从已有场景中检索符合语境的场景，优先使用原版视频
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from search_scenes import load_index, load_scene_metadata, hybrid_search, build_keyword_index
from sentence_transformers import SentenceTransformer
import faiss

# 开头场景推荐查询词
OPENING_QUERIES = [
    "标题画面 片头 开场",
    "全景 远景 星空 云海",
    "淡入 开始 欢迎",
    "平静 优雅 空镜头",
    "仙域 山川 河流"
]

# 结尾场景推荐查询词
ENDING_QUERIES = [
    "淡出 结束 片尾",
    "日落 夜幕 平静",
    "远去 背影 告别",
    "再见 结尾",
    "空镜头 总结"
]

def search_opening_scenes(
    index_path: Path,
    metadata_path: Path,
    scene_metadata_files: List[Path],
    top_k: int = 10
) -> List[Dict]:
    """检索适合作为开头的场景"""
    print("\n" + "="*60)
    print("检索开头场景...")
    print("="*60)
    
    # 加载索引
    index, index_metadata = load_index(index_path, metadata_path)
    print("加载CLIP模型...")
    clip_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 加载场景metadata
    scenes = load_scene_metadata(scene_metadata_files)
    print(f"  共 {len(scenes)} 个场景")
    
    # 建立关键词索引（只使用场景描述，不使用字幕）
    print("建立关键词索引（只使用场景描述，不使用字幕）...")
    keyword_index = build_keyword_index(scenes, use_subtitle=False)
    
    # 合并所有查询，检索
    all_results = {}
    
    for query in OPENING_QUERIES:
        print(f"\n查询: {query}")
        results = hybrid_search(
            query,
            index,
            index_metadata,
            scenes,
            keyword_index,
            clip_model,
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=top_k,
            use_subtitle=False  # 只使用场景描述，不使用字幕
        )
        
        for scene_id, score, scene_data in results:
            # 只保留有场景描述的场景（不使用字幕）
            # 检查是否有真正的场景描述（caption 或 visual_caption）
            has_description = bool(
                scene_data.get('caption') or 
                scene_data.get('visual_caption')
            )
            # 如果 text 字段不等于 subtitle_text，说明 text 是场景描述
            text_value = scene_data.get('text', '')
            subtitle_value = scene_data.get('subtitle_text', '')
            # 如果 text 不等于字幕，说明 text 可能是场景描述
            if text_value and text_value != subtitle_value:
                has_description = True
            
            # 如果没有场景描述（只有字幕），跳过
            if not has_description:
                continue
            
            if scene_id not in all_results or all_results[scene_id]['score'] < score:
                all_results[scene_id] = {
                    'scene_id': scene_id,
                    'score': float(score),
                    'scene_data': scene_data,
                    'query': query
                }
    
    # 按分数排序
    sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
    
    print(f"\n找到 {len(sorted_results)} 个候选开头场景（去重后，只使用场景描述）")
    
    return sorted_results[:top_k * 2]  # 返回更多候选

def search_ending_scenes(
    index_path: Path,
    metadata_path: Path,
    scene_metadata_files: List[Path],
    top_k: int = 10
) -> List[Dict]:
    """检索适合作为结尾的场景"""
    print("\n" + "="*60)
    print("检索结尾场景...")
    print("="*60)
    
    # 加载索引
    index, index_metadata = load_index(index_path, metadata_path)
    print("加载CLIP模型...")
    clip_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 加载场景metadata
    scenes = load_scene_metadata(scene_metadata_files)
    print(f"  共 {len(scenes)} 个场景")
    
    # 建立关键词索引（只使用场景描述，不使用字幕）
    print("建立关键词索引（只使用场景描述，不使用字幕）...")
    keyword_index = build_keyword_index(scenes, use_subtitle=False)
    
    # 合并所有查询，检索
    all_results = {}
    
    for query in ENDING_QUERIES:
        print(f"\n查询: {query}")
        results = hybrid_search(
            query,
            index,
            index_metadata,
            scenes,
            keyword_index,
            clip_model,
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=top_k,
            use_subtitle=False  # 只使用场景描述，不使用字幕
        )
        
        for scene_id, score, scene_data in results:
            # 只保留有场景描述的场景（不使用字幕）
            # 检查是否有真正的场景描述（caption 或 visual_caption）
            has_description = bool(
                scene_data.get('caption') or 
                scene_data.get('visual_caption')
            )
            # 如果 text 字段不等于 subtitle_text，说明 text 是场景描述
            text_value = scene_data.get('text', '')
            subtitle_value = scene_data.get('subtitle_text', '')
            # 如果 text 不等于字幕，说明 text 可能是场景描述
            if text_value and text_value != subtitle_value:
                has_description = True
            
            # 如果没有场景描述（只有字幕），跳过
            if not has_description:
                continue
            
            if scene_id not in all_results or all_results[scene_id]['score'] < score:
                all_results[scene_id] = {
                    'scene_id': scene_id,
                    'score': float(score),
                    'scene_data': scene_data,
                    'query': query
                }
    
    # 按分数排序
    sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
    
    print(f"\n找到 {len(sorted_results)} 个候选结尾场景（去重后，只使用场景描述）")
    
    return sorted_results[:top_k * 2]  # 返回更多候选

def get_scene_position_info(scene_id: str, scene_data: Dict, all_scenes: Dict) -> Dict:
    """获取场景位置信息（是否是开头/结尾场景）"""
    episode_id = scene_data.get('episode_id', '')
    
    # 获取该集的所有场景
    episode_scenes = {
        sid: sdata for sid, sdata in all_scenes.items()
        if sdata.get('episode_id') == episode_id
    }
    
    if not episode_scenes:
        return {}
    
    # 提取场景编号
    scene_numbers = []
    for sid in episode_scenes.keys():
        if '_scene_' in sid:
            try:
                num = int(sid.split('_scene_')[1])
                scene_numbers.append((sid, num))
            except:
                pass
    
    if not scene_numbers:
        return {}
    
    scene_numbers.sort(key=lambda x: x[1])
    total_scenes = len(scene_numbers)
    
    # 找到当前场景的位置
    current_num = None
    for sid, num in scene_numbers:
        if sid == scene_id:
            current_num = num
            break
    
    if current_num is None:
        return {}
    
    position = current_num / total_scenes if total_scenes > 0 else 0.5
    is_opening = position < 0.1  # 前10%
    is_ending = position > 0.9   # 后10%
    
    return {
        'position_percent': position * 100,
        'scene_number': current_num,
        'total_scenes': total_scenes,
        'is_opening_area': is_opening,
        'is_ending_area': is_ending
    }

def format_output(results: List[Dict], scene_type: str, all_scenes: Dict) -> None:
    """格式化输出结果"""
    print(f"\n{'='*60}")
    print(f"推荐的{scene_type}场景（Top 10）:")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results[:10], 1):
        scene_id = result['scene_id']
        scene_data = result['scene_data']
        score = result['score']
        query = result.get('query', '')
        
        episode_id = scene_data.get('episode_id', 'N/A')
        # 优先使用caption，如果没有则使用visual_caption，最后使用subtitle_text
        description = scene_data.get('caption') or scene_data.get('visual_caption') or scene_data.get('description') or 'N/A'
        if description != 'N/A':
            description = description[:100]
        subtitle = scene_data.get('subtitle_text', scene_data.get('subtitle', ''))[:80]
        
        # 获取位置信息
        pos_info = get_scene_position_info(scene_id, scene_data, all_scenes)
        
        # 构建视频文件路径（用于验证）
        from pathlib import Path
        scene_num = scene_id.split('_')[-1] if '_' in scene_id else scene_id.split('_')[-1] if '_scene_' in scene_id else 'unknown'
        if '_' in scene_id and scene_id.split('_')[0].isdigit():
            ep_id, _ = scene_id.split('_', 1)
        else:
            ep_id = episode_id
        video_path = Path(f"processed/episode_{ep_id}/scenes/episode_{ep_id}_clean-Scene-{scene_num}.mp4")
        
        print(f"[{i}] 分数: {score:.4f}")
        print(f"    场景ID: {scene_id}")
        print(f"    集数: {episode_id}")
        if pos_info:
            print(f"    位置: 第{pos_info['scene_number']}个场景（共{pos_info['total_scenes']}个，{pos_info['position_percent']:.1f}%）")
            if pos_info['is_opening_area']:
                print(f"    ⭐ 位于开头区域（前10%）")
            if pos_info['is_ending_area']:
                print(f"    ⭐ 位于结尾区域（后10%）")
        print(f"    视频文件: {video_path.name} {'✅' if video_path.exists() else '❌'}")
        if description != 'N/A':
            print(f"    描述: {description}")
        if subtitle:
            print(f"    字幕: {subtitle}")
        print(f"    匹配查询: {query}")
        print()

def main():
    parser = argparse.ArgumentParser(description='查找适合作为开头和结尾的视频场景')
    parser.add_argument('--index', required=True,
                       help='FAISS索引路径')
    parser.add_argument('--metadata', required=True,
                       help='索引metadata路径')
    parser.add_argument('--scenes', '-s', required=True, nargs='+',
                       help='场景metadata JSON文件（可多个）')
    parser.add_argument('--output', '-o',
                       help='输出JSON文件（可选，保存推荐结果）')
    parser.add_argument('--opening-only', action='store_true',
                       help='只检索开头场景')
    parser.add_argument('--ending-only', action='store_true',
                       help='只检索结尾场景')
    parser.add_argument('--top-k', type=int, default=10,
                       help='每个查询返回的top-k结果（默认: 10）')
    
    args = parser.parse_args()
    
    # 加载所有场景数据（用于位置分析）
    all_scenes = load_scene_metadata([Path(f) for f in args.scenes])
    
    opening_results = []
    ending_results = []
    
    # 检索开头场景
    if not args.ending_only:
        opening_results = search_opening_scenes(
            Path(args.index),
            Path(args.metadata),
            [Path(f) for f in args.scenes],
            args.top_k
        )
        format_output(opening_results, "开头", all_scenes)
    
    # 检索结尾场景
    if not args.opening_only:
        ending_results = search_ending_scenes(
            Path(args.index),
            Path(args.metadata),
            [Path(f) for f in args.scenes],
            args.top_k
        )
        format_output(ending_results, "结尾", all_scenes)
    
    # 保存结果到JSON
    if args.output:
        output_data = {}
        if opening_results:
            output_data['opening_scenes'] = opening_results[:10]
        if ending_results:
            output_data['ending_scenes'] = ending_results[:10]
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 推荐结果已保存: {args.output}")
    
    # 给出使用建议
    print(f"\n{'='*60}")
    print("使用建议:")
    print(f"{'='*60}")
    print("1. 从推荐列表中选择最符合语境的场景")
    print("2. 优先选择位于开头/结尾区域的场景（标注⭐）")
    print("3. 查看场景描述和字幕，确认是否符合要求")
    print("4. 在narration工作流中指定选定的场景ID")
    print("5. 建议选择1个通用开头场景和1个通用结尾场景，所有集数复用")
    print()

if __name__ == '__main__':
    sys.exit(main())
