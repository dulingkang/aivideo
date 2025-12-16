#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说推文视频生成测试脚本
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from generate_novel_video import NovelVideoGenerator


def test_novel_video():
    """测试小说推文视频生成"""
    print("=" * 60)
    print("小说推文视频生成测试")
    print("=" * 60)
    
    # 创建生成器
    generator = NovelVideoGenerator()
    
    # 测试提示词
    test_prompt = "一个美丽的山谷，有瀑布和彩虹，阳光透过云层洒下，远处有雪山，近处有绿树"
    
    # 生成视频
    result = generator.generate(
        prompt=test_prompt,
        output_dir=project_root / "outputs" / "test_novel",
        width=1280,
        height=768,
        num_frames=120,
        fps=24,
    )
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"图像: {result['image']}")
    print(f"视频: {result['video']}")


if __name__ == "__main__":
    test_novel_video()

