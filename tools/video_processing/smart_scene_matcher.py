#!/usr/bin/env python3
"""
智能场景匹配器：根据检索结果自动决策使用检索场景还是AI生成
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess

def get_video_duration(video_path: Path) -> float:
    """获取视频时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

def find_scene_video(episode_id: str, scene_id: str, base_dir: Path) -> Optional[Path]:
    """查找场景对应的视频文件"""
    if '_' in scene_id and scene_id.split('_')[0].isdigit():
        _, pure_scene_id = scene_id.split('_', 1)
    else:
        pure_scene_id = scene_id
    
    episode_dir = base_dir / f"episode_{episode_id}" / "scenes"
    
    patterns = [
        f"*{pure_scene_id}*.mp4",
        f"*Scene-{pure_scene_id.split('_')[-1]}*.mp4",
        f"*{pure_scene_id.split('_')[-1].zfill(3)}*.mp4",
    ]
    
    for pattern in patterns:
        matches = list(episode_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None

def detect_contains_character(text: str) -> bool:
    """
    检测文本中是否包含人像/角色描述
    
    返回True如果包含角色相关关键词
    
    注意：带人像的场景AI生成容易人脸不像，应优先使用检索场景
    """
    if not text:
        return False
    
    # 排除纯环境关键词（如果明确说明只有环境）
    env_only_indicators = ['只有环境', '纯环境', '无人物', '仅环境', '环境场景', '空场景']
    if any(ind in text for ind in env_only_indicators):
        return False
    
    character_keywords = [
        # 人称代词和角色
        '主角', '角色', '人物', '他', '她', '他们', '主人公',
        '韩立', '修士', '弟子', '真人', '道人', '道友',
        # 明确的人的动作（包含"着"等持续动作）
        '站着', '坐着', '走着', '看着', '说着', '拿着', '手持', '握着',
        '抱着', '背着', '扛着',
        # 身体部位
        '脸', '面容', '表情', '眼神', '手', '手臂', '身影',
        # 第一人称（通常意味着有说话者）
        '我', '我们'
    ]
    
    text_lower = text.lower()
    
    # 检查是否包含关键词
    for keyword in character_keywords:
        if keyword in text_lower:
            return True
    
    return False

def detect_contains_character(text: str) -> bool:
    """
    检测文本中是否包含人像/角色描述
    
    返回True如果包含角色相关关键词
    """
    if not text:
        return False
    
    character_keywords = [
        # 人称代词和角色
        '主角', '角色', '人物', '人', '他', '她', '他们', '主人公',
        '韩立', '修士', '弟子', '真人', '道人', '修士', '道友',
        # 角色动作（明确涉及人）
        '站着', '坐着', '走着', '看着', '说着', '拿着', '手持', '握着',
        '拿', '握', '持', '抱', '背', '扛',
        # 身体部位
        '脸', '面', '面容', '表情', '眼睛', '眼神', '手', '手臂', '身影',
        # 动作词（通常涉及人）
        '站', '坐', '走', '跑', '看', '说', '笑', '哭', '点头', '摇头'
    ]
    
    text_lower = text.lower()
    
    # 排除纯环境关键词（如果明确说明只有环境）
    env_only_indicators = ['只有环境', '纯环境', '无人物', '仅环境', '环境场景']
    if any(ind in text for ind in env_only_indicators):
        return False
    
    # 检查是否包含角色关键词
    for keyword in character_keywords:
        if keyword in text_lower:
            return True
    
    return False

def decision_make(
    search_results: List[Dict],
    target_duration: float,
    base_dir: Path,
    narration_text: str = "",
    score_threshold_high: float = 0.7,
    score_threshold_low: float = 0.5,
    duration_tolerance: float = 0.3,
    avoid_ai_for_characters: bool = True,
    prefer_retrieved: bool = True
) -> Dict:
    """
    智能决策：使用检索场景还是AI生成
    
    Returns:
        {
            'decision': 'retrieved' | 'ai_generate',
            'reason': '决策原因',
            'matched_videos': [视频路径列表],
            'actual_duration': 实际时长,
            'score': 检索分数,
            'ai_prompt': AI生成时的提示词（如果需要生成）
        }
    """
    if not search_results:
        return {
            'decision': 'ai_generate',
            'reason': '没有检索结果',
            'matched_videos': [],
            'actual_duration': 0.0,
            'score': 0.0,
            'ai_prompt': None
        }
    
    best_result = search_results[0]
    best_score = best_result.get('score', 0.0)
    best_scene_id = best_result.get('scene_id', '')
    best_scene_data = best_result.get('scene_data', {})
    episode_id = best_scene_data.get('episode_id', '')
    
    # 查找视频文件
    video_path = find_scene_video(episode_id, best_scene_id, base_dir)
    
    # 检测是否包含人像
    contains_character = detect_contains_character(narration_text) if narration_text else False
    
    # 策略调整：如果设置了优先检索，大幅降低阈值以使用更多检索场景
    if prefer_retrieved:
        effective_threshold_high = score_threshold_high * 0.3  # 降低70%，更倾向于检索
        effective_threshold_low = score_threshold_low * 0.1    # 大幅降低到原来的10%
    else:
        effective_threshold_high = score_threshold_high
        effective_threshold_low = score_threshold_low
    
    # 策略1: 检索分数较高，优先使用
    if best_score >= effective_threshold_high:
        if video_path and video_path.exists():
            video_duration = get_video_duration(video_path)
            duration_diff = abs(video_duration - target_duration) / target_duration
            
            # 时长匹配度好，直接使用
            if duration_diff <= duration_tolerance:
                return {
                    'decision': 'retrieved',
                    'reason': f'检索分数高({best_score:.2f})且时长匹配({duration_diff*100:.1f}%)',
                    'matched_videos': [video_path],
                    'actual_duration': video_duration,
                    'score': best_score,
                    'ai_prompt': None
                }
            # 时长差异较大，但可以裁剪/拼接
            else:
                return {
                    'decision': 'retrieved',
                    'reason': f'检索分数高({best_score:.2f})，时长差异({duration_diff*100:.1f}%)可通过裁剪/拼接解决',
                    'matched_videos': [video_path],
                    'actual_duration': video_duration,
                    'score': best_score,
                    'ai_prompt': None
                }
        else:
            # 视频文件不存在，如果优先检索，尝试其他检索结果
            if prefer_retrieved:
                for result in search_results[1:5]:  # 尝试前5个结果
                    scene_data = result.get('scene_data', {})
                    scene_id = result.get('scene_id', '')
                    ep_id = scene_data.get('episode_id', '')
                    test_video = find_scene_video(ep_id, scene_id, base_dir)
                    if test_video and test_video.exists():
                        video_duration = get_video_duration(test_video)
                        return {
                            'decision': 'retrieved',
                            'reason': f'检索分数高但最佳结果视频不存在，使用替代场景: {scene_id}（分数: {result.get("score", 0):.2f}）',
                            'matched_videos': [test_video],
                            'actual_duration': video_duration,
                            'score': result.get('score', 0),
                            'ai_prompt': None,
                            'force_retrieved': True
                        }
            return {
                'decision': 'ai_generate',
                'reason': f'检索分数高但视频文件不存在: {best_scene_id}',
                'matched_videos': [],
                'actual_duration': 0.0,
                'score': best_score,
                'ai_prompt': None
            }
    
    elif best_score >= effective_threshold_low:
        if video_path and video_path.exists():
            video_duration = get_video_duration(video_path)
            duration_diff = abs(video_duration - target_duration) / target_duration
            
            # 时长匹配度好，可以使用
            if duration_diff <= duration_tolerance:
                return {
                    'decision': 'retrieved',
                    'reason': f'检索分数中等({best_score:.2f})但时长匹配好({duration_diff*100:.1f}%)',
                    'matched_videos': [video_path],
                    'actual_duration': video_duration,
                    'score': best_score,
                    'ai_prompt': None
                }
            # 时长差异较大，但可以尝试拼接
            elif duration_diff <= 0.5:  # 50%以内的差异可以拼接
                # 尝试拼接多个场景
                matched_videos = [video_path]
                total_duration = video_duration
                
                for result in search_results[1:]:  # 从第二个开始
                    scene_data = result.get('scene_data', {})
                    scene_id = result.get('scene_id', '')
                    ep_id = scene_data.get('episode_id', '')
                    next_video = find_scene_video(ep_id, scene_id, base_dir)
                    
                    if next_video and next_video.exists():
                        next_duration = get_video_duration(next_video)
                        if total_duration + next_duration <= target_duration * 1.2:  # 允许20%的超出
                            matched_videos.append(next_video)
                            total_duration += next_duration
                            if total_duration >= target_duration:
                                break
                
                if len(matched_videos) > 1:
                    return {
                        'decision': 'retrieved',
                        'reason': f'检索分数中等({best_score:.2f})，拼接{len(matched_videos)}个场景',
                        'matched_videos': matched_videos,
                        'actual_duration': total_duration,
                        'score': best_score,
                        'ai_prompt': None
                    }
                else:
                    # 拼接失败，但优先检索时仍使用原版视频
                    if prefer_retrieved:
                        return {
                            'decision': 'retrieved',
                            'reason': f'检索分数中等({best_score:.2f})且时长差异({duration_diff*100:.1f}%)，优先使用原版视频',
                            'matched_videos': [video_path],
                            'actual_duration': video_duration,
                            'score': best_score,
                            'ai_prompt': None,
                            'force_retrieved': True
                        }
                    return {
                        'decision': 'ai_generate',
                        'reason': f'检索分数中等({best_score:.2f})且时长无法匹配({duration_diff*100:.1f}%)',
                        'matched_videos': [],
                        'actual_duration': 0.0,
                        'score': best_score,
                        'ai_prompt': None
                    }
            else:
                # 时长差异太大（>50%），但如果优先检索，仍然使用原版视频
                if prefer_retrieved:
                    return {
                        'decision': 'retrieved',
                        'reason': f'检索分数中等({best_score:.2f})但时长差异过大({duration_diff*100:.1f}%)，优先使用原版视频',
                        'matched_videos': [video_path],
                        'actual_duration': video_duration,
                        'score': best_score,
                        'ai_prompt': None,
                        'force_retrieved': True
                    }
                return {
                    'decision': 'ai_generate',
                    'reason': f'检索分数中等({best_score:.2f})但时长差异过大({duration_diff*100:.1f}%)',
                    'matched_videos': [],
                    'actual_duration': 0.0,
                    'score': best_score,
                    'ai_prompt': None
                }
        else:
            # 视频文件不存在，尝试其他检索结果
            if prefer_retrieved:
                for result in search_results[1:5]:  # 尝试前5个结果
                    scene_data = result.get('scene_data', {})
                    scene_id = result.get('scene_id', '')
                    ep_id = scene_data.get('episode_id', '')
                    test_video = find_scene_video(ep_id, scene_id, base_dir)
                    if test_video and test_video.exists():
                        video_duration = get_video_duration(test_video)
                        return {
                            'decision': 'retrieved',
                            'reason': f'检索分数中等但最佳结果视频不存在，使用替代场景: {scene_id}（分数: {result.get("score", 0):.2f}）',
                            'matched_videos': [test_video],
                            'actual_duration': video_duration,
                            'score': result.get('score', 0),
                            'ai_prompt': None,
                            'force_retrieved': True
                        }
            return {
                'decision': 'ai_generate',
                'reason': f'检索分数中等但视频文件不存在: {best_scene_id}',
                'matched_videos': [],
                'actual_duration': 0.0,
                'score': best_score,
                'ai_prompt': None
            }
    
    # 策略3: 检索分数很低，仍然尽量使用检索（原版优先，尽量减少AI生成）
    else:
        # 优先使用原版：即使分数很低，只要有检索结果，都尽量使用
        if prefer_retrieved and video_path and video_path.exists():
            video_duration = get_video_duration(video_path)
            # 尽量使用检索场景，即使分数很低（原版优先）
            return {
                'decision': 'retrieved',
                'reason': f'检索分数很低({best_score:.2f})，但优先使用原版视频，使用检索场景（尽量减少AI生成）',
                'matched_videos': [video_path],
                'actual_duration': video_duration,
                'score': best_score,
                'ai_prompt': None,
                'contains_character': contains_character,
                'force_retrieved': True,  # 标记为优先检索
                'low_score_but_used': True  # 标记为低分但仍使用
            }
        
        # 如果设置了优先检索，尝试使用所有检索结果中的任何一个（即使分数很低）
        if prefer_retrieved:
            # 尝试更多结果（前10个），只要有视频文件就使用
            for result in search_results[:10]:  # 尝试前10个结果
                scene_data = result.get('scene_data', {})
                scene_id = result.get('scene_id', '')
                ep_id = scene_data.get('episode_id', '')
                test_video = find_scene_video(ep_id, scene_id, base_dir)
                if test_video and test_video.exists():
                    video_duration = get_video_duration(test_video)
                    return {
                        'decision': 'retrieved',
                        'reason': f'优先使用原版，使用检索结果: {scene_id}（分数: {result.get("score", 0):.2f}）',
                        'matched_videos': [test_video],
                        'actual_duration': video_duration,
                        'score': result.get('score', 0),
                        'ai_prompt': None,
                        'contains_character': contains_character,
                        'force_retrieved': True,
                        'low_score_but_used': True
                    }
        
        # 只有完全没有检索结果时，才建议AI生成
        return {
            'decision': 'ai_generate',
            'reason': f'检索分数极低({best_score:.2f})且所有检索结果都无法找到视频文件，只能使用AI生成',
            'matched_videos': [],
            'actual_duration': 0.0,
            'score': best_score,
            'ai_prompt': narration_text if not contains_character else None,
            'contains_character': contains_character,
            'no_retrieved_option': True  # 标记为完全没有检索选项
        }

def process_narration_with_decision(
    narration_file: Path,
    search_results_dir: Path,
    video_base_dir: Path,
    output_dir: Path,
    score_threshold_high: float = 0.7,
    score_threshold_low: float = 0.5,
    duration_tolerance: float = 0.3,
    avoid_ai_for_characters: bool = True,
    prefer_retrieved: bool = True
) -> Dict:
    """
    处理narration文件，智能决策每个段落使用检索还是生成
    
    Returns:
        包含所有段落决策结果的字典
    """
    # 加载narration
    with open(narration_file, 'r', encoding='utf-8') as f:
        narration_data = json.load(f)
    
    narration_parts = narration_data.get('narration_parts', [])
    if isinstance(narration_data, list):
        narration_parts = narration_data
    
    decisions = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"处理 {len(narration_parts)} 个narration段落...\n")
    
    for i, part in enumerate(narration_parts, 1):
        text = part.get('text', '').strip()
        duration = part.get('duration', 0.0)
        
        if not text:
            continue
        
        print(f"[{i}/{len(narration_parts)}] {text[:50]}... (时长: {duration:.2f}s)")
        
        # 加载检索结果
        search_result_file = search_results_dir / f"narration_{i}_search.json"
        if not search_result_file.exists():
            print(f"  ✗ 检索结果文件不存在: {search_result_file.name}")
            decisions.append({
                'narration_index': i,
                'text': text,
                'target_duration': duration,
                'decision': 'ai_generate',
                'reason': '检索结果文件不存在',
                'score': 0.0
            })
            continue
        
        with open(search_result_file, 'r', encoding='utf-8') as f:
            search_data = json.load(f)
        
        search_results = search_data.get('results', [])
        
        # 智能决策
        decision = decision_make(
            search_results,
            duration,
            video_base_dir,
            narration_text=text,
            score_threshold_high=score_threshold_high,
            score_threshold_low=score_threshold_low,
            duration_tolerance=duration_tolerance,
            avoid_ai_for_characters=avoid_ai_for_characters,
            prefer_retrieved=prefer_retrieved
        )
        
        decision['narration_index'] = i
        decision['text'] = text
        decision['target_duration'] = duration
        decisions.append(decision)
        
        # 输出决策结果
        if decision['decision'] == 'retrieved':
            if decision.get('force_retrieved') or decision.get('low_score_but_used'):
                score_note = "（优先使用原版视频）" if decision.get('low_score_but_used') else ""
                char_note = "（包含人像，避免AI生成人脸问题）" if decision.get('contains_character') else ""
                print(f"  ✓ 决策: 使用检索场景（原版视频）(分数: {decision['score']:.2f}){score_note}{char_note}")
                print(f"    原因: {decision['reason']}")
            else:
                character_note = "（包含人像）" if decision.get('contains_character') else "（纯环境）"
                print(f"  ✓ 决策: 使用检索场景（原版视频）(分数: {decision['score']:.2f}){character_note}")
                print(f"    原因: {decision['reason']}")
            if decision['matched_videos']:
                print(f"    视频: {len(decision['matched_videos'])} 个，总时长: {decision['actual_duration']:.2f}s")
        else:
            if decision.get('contains_character'):
                print(f"  ⚠ 决策: 需要AI生成但包含人像 (分数: {decision['score']:.2f})")
                print(f"    警告: {decision.get('warning', 'AI生成人脸可能不像')}")
                print(f"    建议: 优先使用检索场景，即使分数较低")
            else:
                print(f"  → 决策: 需要AI生成 (分数: {decision['score']:.2f})（纯环境场景）")
            print(f"    原因: {decision['reason']}")
    
    # 统计结果
    retrieved_count = sum(1 for d in decisions if d['decision'] == 'retrieved')
    generate_count = sum(1 for d in decisions if d['decision'] == 'ai_generate')
    low_score_used = sum(1 for d in decisions if d.get('low_score_but_used', False))
    character_count = sum(1 for d in decisions if d.get('contains_character', False))
    env_only_count = sum(1 for d in decisions if not d.get('contains_character', True))
    force_retrieved_count = sum(1 for d in decisions if d.get('force_retrieved', False))
    
    print(f"\n{'='*60}")
    print(f"决策统计（优先使用原版视频策略）:")
    print(f"  ✓ 使用检索场景（原版视频）: {retrieved_count} 个 ({retrieved_count/len(decisions)*100:.1f}%)")
    if low_score_used > 0:
        print(f"    └─ 其中低分但仍使用原版: {low_score_used} 个（尽量减少AI生成）")
    if force_retrieved_count > 0:
        print(f"    └─ 强制使用检索（包含人像）: {force_retrieved_count} 个")
    print(f"  → 需要AI生成: {generate_count} 个 ({generate_count/len(decisions)*100:.1f}%)")
    if generate_count > 0:
        ai_char_count = character_count - sum(1 for d in decisions if d.get('force_retrieved', False) and d.get('contains_character', False))
        if ai_char_count > 0:
            print(f"    └─ 包含人像（不推荐）: {ai_char_count} 个")
        if env_only_count > 0:
            print(f"    └─ 纯环境场景（适合AI）: {env_only_count} 个")
    if prefer_retrieved:
        print(f"\n策略: 优先使用原版视频，AI生成率已降至最低")
    print(f"{'='*60}")
    
    # 保存决策结果
    result_file = output_dir / "scene_decisions.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_narrations': len(narration_parts),
            'decisions': decisions,
            'statistics': {
                'retrieved_count': retrieved_count,
                'ai_generate_count': generate_count,
                'retrieved_percentage': retrieved_count / len(decisions) * 100 if decisions else 0,
                'ai_generate_percentage': generate_count / len(decisions) * 100 if decisions else 0
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 决策结果已保存: {result_file}")
    
    return {
        'decisions': decisions,
        'statistics': {
            'retrieved_count': retrieved_count,
            'ai_generate_count': generate_count
        }
    }

def main():
    parser = argparse.ArgumentParser(description='智能场景匹配器：自动决策使用检索还是AI生成')
    parser.add_argument('--narration', '-n', required=True,
                       help='narration JSON文件')
    parser.add_argument('--search-results-dir', '-s', required=True,
                       help='检索结果目录（每个narration段落对应一个检索结果JSON）')
    parser.add_argument('--video-base-dir', '-v', default='processed',
                       help='视频基础目录（默认: processed/）')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录（保存决策结果）')
    parser.add_argument('--score-threshold-high', type=float, default=0.7,
                       help='高分阈值（>=此分数优先使用检索，默认: 0.7，实际会降低30%以优先使用检索）')
    parser.add_argument('--score-threshold-low', type=float, default=0.5,
                       help='低分阈值（<此分数建议AI生成，默认: 0.5，实际会大幅降低以优先使用检索）')
    parser.add_argument('--prefer-retrieved', action='store_true', default=True,
                       help='优先使用检索场景（原版视频），尽量减少AI生成（默认: True）')
    parser.add_argument('--no-prefer-retrieved', dest='prefer_retrieved', action='store_false',
                       help='不强制优先检索，允许更多AI生成（不推荐）')
    parser.add_argument('--duration-tolerance', type=float, default=0.3,
                       help='时长容差（比例，默认: 0.3，即30%内认为匹配）')
    parser.add_argument('--allow-ai-for-characters', action='store_true',
                       help='允许AI生成包含人像的场景（默认False，避免人脸不像的问题）')
    
    args = parser.parse_args()
    
    narration_file = Path(args.narration)
    if not narration_file.exists():
        print(f"错误: narration文件不存在: {narration_file}")
        return 1
    
    search_results_dir = Path(args.search_results_dir)
    if not search_results_dir.exists():
        print(f"错误: 检索结果目录不存在: {search_results_dir}")
        return 1
    
    video_base_dir = Path(args.video_base_dir)
    
    result = process_narration_with_decision(
        narration_file,
        search_results_dir,
        video_base_dir,
        Path(args.output),
        args.score_threshold_high,
        args.score_threshold_low,
        args.duration_tolerance,
        avoid_ai_for_characters=not args.allow_ai_for_characters,
        prefer_retrieved=args.prefer_retrieved
    )
    
    # 输出需要AI生成的场景列表
    ai_generate_scenes = [d for d in result['decisions'] if d['decision'] == 'ai_generate']
    if ai_generate_scenes:
        print(f"\n需要AI生成的场景 ({len(ai_generate_scenes)} 个):")
        for scene in ai_generate_scenes:
            print(f"  [{scene['narration_index']}] {scene['text'][:50]}... (分数: {scene['score']:.2f})")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

