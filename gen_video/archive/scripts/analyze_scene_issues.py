#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析scene_007和scene_012的问题
"""

import json
import sys
from pathlib import Path

def analyze_scene_issues():
    """分析scene_007和scene_012的问题"""
    
    # 读取分析结果
    analysis_file = Path(__file__).parent / "analysis_results.json"
    if not analysis_file.exists():
        print(f"❌ 未找到分析结果文件: {analysis_file}")
        return
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # 查找scene_007 (scene_id=6) 和 scene_012 (scene_id=11)
    scene_007 = None
    scene_012 = None
    
    for item in analysis_data:
        if item.get('scene_id') == 6:
            scene_007 = item
        elif item.get('scene_id') == 11:
            scene_012 = item
    
    print("=" * 80)
    print("场景问题分析报告")
    print("=" * 80)
    
    # 分析scene_007
    if scene_007:
        print("\n【场景 7 (scene_007) 分析】")
        print("-" * 80)
        prompt_analysis = scene_007.get('prompt_analysis', {})
        expected = prompt_analysis.get('expected', {})
        prompt_text = prompt_analysis.get('prompt_text', '')
        
        print(f"期望的场景描述:")
        print(f"  - character_pose: {expected.get('character_pose', 'N/A')}")
        print(f"  - composition: {expected.get('composition', 'N/A')}")
        print(f"  - camera: {expected.get('camera', 'N/A')}")
        print(f"  - action: {expected.get('action', 'N/A')}")
        print(f"\n实际使用的prompt:")
        print(f"  {prompt_text}")
        
        print(f"\n问题分析:")
        print(f"  ⚠️ 用户反馈：出现了坦克（不正常）")
        print(f"  🔍 可能原因：")
        print(f"    1. prompt中包含'gravel'（沙砾），可能被模型误解")
        print(f"    2. 'wide shot'可能触发了某些军事场景的联想")
        print(f"    3. prompt中缺少明确的排除项（如'no vehicles', 'no tanks'）")
        print(f"    4. 模型训练数据中'gravel'和'wide shot'的组合可能关联到军事场景")
        
        suggestions = scene_007.get('suggestions', [])
        if suggestions:
            print(f"\n优化建议:")
            for sug in suggestions:
                print(f"  - {sug}")
    else:
        print("\n❌ 未找到scene_007的分析数据")
    
    # 分析scene_012
    if scene_012:
        print("\n【场景 12 (scene_012) 分析】")
        print("-" * 80)
        prompt_analysis = scene_012.get('prompt_analysis', {})
        expected = prompt_analysis.get('expected', {})
        prompt_text = prompt_analysis.get('prompt_text', '')
        
        print(f"期望的场景描述:")
        print(f"  - character_pose: {expected.get('character_pose', 'N/A')}")
        print(f"  - composition: {expected.get('composition', 'N/A')}")
        print(f"  - camera: {expected.get('camera', 'N/A')}")
        print(f"  - action: {expected.get('action', 'N/A')}")
        print(f"\n实际使用的prompt:")
        print(f"  {prompt_text}")
        
        print(f"\n问题分析:")
        print(f"  ⚠️ 用户反馈：出现了10个一样的人（不正常）")
        print(f"  🔍 可能原因：")
        print(f"    1. 单人约束（single person）权重不够高或位置不对")
        print(f"    2. prompt被截断，单人约束被移除")
        print(f"    3. 模型理解错误，将某些描述理解为'多个人'")
        print(f"    4. 场景描述中可能包含暗示多人的词汇")
        
        suggestions = scene_012.get('suggestions', [])
        if suggestions:
            print(f"\n优化建议:")
            for sug in suggestions:
                print(f"  - {sug}")
    else:
        print("\n❌ 未找到scene_012的分析数据")
        print("   尝试查找scene_id=11...")
        # 尝试查找scene_id=11
        for item in analysis_data:
            if item.get('scene_id') == 11:
                scene_012 = item
                break
    
    print("\n" + "=" * 80)
    print("建议的修复方案")
    print("=" * 80)
    
    print("\n【针对scene_007（坦克问题）】")
    print("1. 在prompt中添加明确的排除项：")
    print("   - 'no vehicles, no tanks, no military equipment, no weapons'")
    print("2. 强化场景描述，明确是修仙场景：")
    print("   - 'xianxia fantasy, cultivation world, no modern elements'")
    print("3. 检查'gravel'是否被误解，考虑使用更明确的描述：")
    print("   - 'gray-green sand ground' 替代 'gravel'")
    
    print("\n【针对scene_012（多人问题）】")
    print("1. 提高单人约束的权重和优先级：")
    print("   - 将单人约束放在prompt最前面，权重提高到2.5")
    print("   - 添加多个单人约束变体：'single person', 'only one character', 'lone figure'")
    print("2. 在negative prompt中添加：")
    print("   - 'multiple people, crowd, group, many characters'")
    print("3. 检查prompt是否被截断，确保单人约束在77 tokens限制内")
    print("4. 如果使用InstantID，检查参考图像是否包含多个人")

if __name__ == "__main__":
    analyze_scene_issues()


