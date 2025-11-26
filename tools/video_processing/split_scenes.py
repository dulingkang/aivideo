#!/usr/bin/env python3
"""
镜头自动切分脚本：使用PySceneDetect进行镜头级切分
"""

import os
import sys
import subprocess
import argparse
import json
import re
from pathlib import Path

def split_video_scenes(input_path, output_dir, method='content', threshold=30.0):
    """
    使用PySceneDetect切分视频为镜头片段
    
    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        method: 检测方法 ('content' 或 'threshold')
        threshold: 检测阈值
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PySceneDetect命令
    # 注意：-t 参数属于 detect-content，不是 split-video
    cmd = [
        'scenedetect',
        '-i', str(input_path),
        '-o', str(output_dir),
    ]
    
    if method == 'content':
        cmd.extend(['detect-content', '-t', str(threshold)])
    else:
        cmd.extend(['detect-threshold', '-t', str(threshold)])
    
    cmd.append('split-video')
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    
    print(f"✓ 切分完成，输出目录: {output_dir}")
    
    # 统计切分结果
    scene_files = sorted(output_dir.glob('*.mp4'))
    print(f"  共切分出 {len(scene_files)} 个镜头片段")
    
    # 生成场景时间列表（用于字幕对齐）
    scene_list = []
    for i, scene_file in enumerate(scene_files):
        # 从文件名提取场景信息（例如：episode_171_clean-Scene-007.mp4）
        scene_match = re.search(r'Scene-(\d+)', scene_file.name)
        if scene_match:
            scene_num = int(scene_match.group(1))
            scene_id = f"scene_{scene_num:03d}"
        else:
            scene_id = f"scene_{i:03d}"
        
        # 获取视频时长
        duration_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(scene_file)
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        duration = float(duration_result.stdout.strip()) if duration_result.returncode == 0 else 0.0
        
        # 计算开始时间（累加前面所有场景的时长）
        start_time = sum(s.get("duration", 0) for s in scene_list)
        end_time = start_time + duration
        
        scene_list.append({
            "scene_id": scene_id,
            "file": scene_file.name,
            "start": start_time,
            "end": end_time,
            "duration": duration
        })
    
    # 保存场景列表JSON
    scene_list_json = output_dir.parent / "scene_list.json"
    with open(scene_list_json, 'w', encoding='utf-8') as f:
        json.dump({
            "total_scenes": len(scene_list),
            "scenes": scene_list
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 场景列表已保存: {scene_list_json}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='自动切分视频为镜头片段')
    parser.add_argument('--input', '-i', required=True, help='输入视频路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录')
    parser.add_argument('--method', '-m', default='content', choices=['content', 'threshold'],
                       help='检测方法: content(内容变化) 或 threshold(阈值)')
    parser.add_argument('--threshold', '-t', type=float, default=30.0,
                       help='检测阈值 (content方法使用)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return 1
    
    if split_video_scenes(input_path, args.output, args.method, args.threshold):
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())

