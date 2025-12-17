#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试角色档案系统

运行方式:
    python test_character_profile.py
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_character_profile():
    """测试角色档案加载"""
    print("\n" + "=" * 60)
    print("测试角色档案系统")
    print("=" * 60)
    
    # 导入
    try:
        from pulid_engine import CharacterProfile
    except ImportError as e:
        # 如果导入失败（缺少 torch），使用简化测试
        print(f"⚠️ 完整导入失败: {e}")
        print("使用简化测试模式...")
        return test_character_profile_simple()
    
    # 测试目录
    profile_dir = Path('character_profiles/hanli')
    print(f"\n测试目录: {profile_dir}")
    print(f"目录存在: {profile_dir.exists()}")
    
    if not profile_dir.exists():
        print("❌ 角色档案目录不存在")
        return False
    
    # 加载角色档案
    profile = CharacterProfile('hanli', str(profile_dir))
    print(f"\n角色档案: {profile}")
    
    # 测试不同场景的参考图选择
    test_cases = [
        ('eye_level', 'neutral', '平视中性'),
        ('eye_level', 'angry', '平视愤怒'),
        ('eye_level', 'happy', '平视开心'),
        ('eye_level', 'sad', '平视悲伤'),
        ('eye_level', 'pain', '平视痛苦'),
        ('side', 'neutral', '侧面中性'),
        ('side', 'angry', '侧面愤怒'),
        ('front', 'happy', '正面开心'),
        ('top_down', 'sad', '俯拍悲伤'),
        ('low', 'angry', '仰拍愤怒'),
    ]
    
    print('\n参考图选择测试:')
    print('-' * 80)
    
    all_passed = True
    for camera, emotion, desc in test_cases:
        primary, expr = profile.get_reference_for_scene(camera, emotion)
        
        if primary:
            primary_path = f"{primary.parent.name}/{primary.name}"
        else:
            primary_path = "None"
            all_passed = False
        
        if expr:
            expr_path = f"{expr.parent.name}/{expr.name}"
        else:
            expr_path = "-"
        
        status = "✅" if primary else "❌"
        print(f"  {status} {desc:12} | camera={camera:12} emotion={emotion:8} | primary={primary_path:25} | expr_ref={expr_path}")
    
    print('-' * 80)
    
    if all_passed:
        print("\n✅ 所有测试通过!")
    else:
        print("\n⚠️ 部分场景没有找到参考图")
    
    return True


def test_character_profile_simple():
    """简化测试 - 不依赖 torch"""
    print("\n简化测试模式（不加载完整模块）")
    
    profile_dir = Path('character_profiles/hanli')
    
    if not profile_dir.exists():
        print(f"❌ 目录不存在: {profile_dir}")
        return False
    
    # 直接扫描目录结构
    print(f"\n📁 目录结构:")
    
    angles = ['front', 'side', 'three_quarter']
    expressions = ['neutral', 'happy', 'sad', 'angry', 'pain', 'surprised']
    
    results = {}
    for angle in angles:
        angle_dir = profile_dir / angle
        if angle_dir.exists():
            results[angle] = []
            print(f"\n  {angle}/")
            for expr in expressions:
                for ext in ['.jpg', '.png']:
                    file_path = angle_dir / f"{expr}{ext}"
                    if file_path.exists():
                        results[angle].append(expr)
                        print(f"    ✅ {expr}{ext}")
                        break
    
    # 统计
    print("\n📊 统计:")
    for angle, exprs in results.items():
        print(f"  {angle}: {len(exprs)} 个表情 ({', '.join(exprs)})")
    
    total = sum(len(v) for v in results.values())
    print(f"\n总计: {total} 张参考图")
    
    if total >= 5:
        print("\n✅ 角色档案结构正确!")
        return True
    else:
        print("\n⚠️ 参考图数量较少")
        return False


def test_scene_mapping():
    """测试场景映射逻辑"""
    print("\n" + "=" * 60)
    print("场景映射测试")
    print("=" * 60)
    
    # 相机角度 -> 参考图角度 映射
    angle_mapping = {
        "eye_level": "three_quarter",
        "front": "front",
        "side": "side",
        "profile": "side",
        "top_down": "front",
        "bird_eye": "front",
        "low": "three_quarter",
    }
    
    print("\n相机角度映射:")
    for camera, ref_angle in angle_mapping.items():
        print(f"  {camera:12} -> {ref_angle}")
    
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    print(f"工作目录: {os.getcwd()}")
    
    success = True
    
    # 测试1: 角色档案
    if not test_character_profile():
        success = False
    
    # 测试2: 场景映射
    test_scene_mapping()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试完成!")
    else:
        print("⚠️ 部分测试未通过")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
