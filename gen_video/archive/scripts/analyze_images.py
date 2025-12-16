#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分析脚本
分析生成的图像，对比prompt和实际内容，找出优化点
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_analyzer import ImageAnalyzer


def main():
    parser = argparse.ArgumentParser(description="分析生成的图像，对比prompt和实际内容")
    parser.add_argument("json_file", help="场景JSON文件路径")
    parser.add_argument("--image-dir", default="outputs/images", help="图像目录（默认: outputs/images）")
    parser.add_argument("--output", default="image_analysis_report.json", help="输出文件路径（默认: image_analysis_report.json）")
    parser.add_argument("--report", default="image_analysis_report.txt", help="报告文件路径（默认: image_analysis_report.txt）")
    
    args = parser.parse_args()
    
    # 加载场景数据
    print(f"加载场景数据: {args.json_file}")
    with open(args.json_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    scenes = script_data.get('scenes', [])
    print(f"找到 {len(scenes)} 个场景")
    
    # 初始化分析器
    print("\n初始化图像分析器...")
    analyzer = ImageAnalyzer()
    
    # 批量分析
    print(f"\n开始批量分析图像...")
    results = analyzer.analyze_batch(scenes, args.image_dir, args.output)
    
    # 生成报告
    print(f"\n生成分析报告...")
    report = analyzer.generate_report(results)
    
    # 保存报告
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ 分析完成")
    print(f"  - 分析结果: {args.output}")
    print(f"  - 分析报告: {args.report}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("分析摘要")
    print("=" * 80)
    
    total_mismatches = sum(len(r.get("comparison", {}).get("mismatches", [])) for r in results)
    total_missing = sum(len(r.get("comparison", {}).get("missing", [])) for r in results)
    total_suggestions = sum(len(r.get("suggestions", [])) for r in results)
    
    print(f"总场景数: {len(results)}")
    print(f"不匹配项: {total_mismatches}")
    print(f"缺失项: {total_missing}")
    print(f"优化建议: {total_suggestions}")
    
    if total_mismatches > 0 or total_missing > 0:
        print("\n⚠ 发现需要优化的场景，请查看详细报告")


if __name__ == "__main__":
    main()

