#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视频质量分析器
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.video_quality_analyzer import VideoQualityAnalyzer


def test_analyze_video(video_path: str):
    """测试视频质量分析"""
    print("=" * 60)
    print("视频质量分析测试")
    print("=" * 60)
    
    analyzer = VideoQualityAnalyzer()
    
    try:
        results = analyzer.analyze(video_path)
        report = analyzer.generate_report(results)
        print(report)
        
        # 保存结果到JSON
        import json
        output_path = Path(video_path).parent / f"{Path(video_path).stem}_quality.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 分析结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 使用默认测试视频
        test_video = "outputs/test_novel/novel_video.mp4"
        if Path(test_video).exists():
            test_analyze_video(test_video)
        else:
            print("用法: python test_video_quality.py <video_path>")
            print(f"或确保测试视频存在: {test_video}")
    else:
        test_analyze_video(sys.argv[1])

