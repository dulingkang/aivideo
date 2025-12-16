#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的视频生成Pipeline
测试Deforum相机运动、SVD参数生成、场景类型分类等功能
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from deforum_camera_motion import DeforumCameraMotion
from svd_parameter_generator import SVDParameterGenerator
from scene_type_classifier import SceneTypeClassifier
from scene_motion_analyzer import SceneMotionAnalyzer


def test_deforum_camera_motion():
    """测试Deforum相机运动模块"""
    print("=" * 60)
    print("测试1: Deforum相机运动模块")
    print("=" * 60)
    
    deforum = DeforumCameraMotion()
    
    # 测试场景1：远景场景（应该zoom in）
    scene1 = {
        "visual": {
            "composition": "wide shot, distant view, 远景",
            "motion": {"type": "static"}
        },
        "camera": "wide shot"
    }
    
    params1 = deforum.generate_motion_params_from_scene(scene1)
    print(f"\n场景1（远景）:")
    print(f"  Zoom: {params1['zoom']['start']:.2f} → {params1['zoom']['end']:.2f}")
    print(f"  Pan X: {params1['pan_x']['start']:.2f} → {params1['pan_x']['end']:.2f}")
    assert params1['zoom']['end'] > params1['zoom']['start'], "远景应该zoom in"
    print("  ✓ 测试通过：远景场景正确设置zoom in")
    
    # 测试场景2：中景场景（应该pan）
    scene2 = {
        "visual": {
            "composition": "medium shot, mid view, 中景",
            "motion": {"type": "pan", "direction": "left_to_right"}
        }
    }
    
    params2 = deforum.generate_motion_params_from_scene(scene2)
    print(f"\n场景2（中景+pan）:")
    print(f"  Zoom: {params2['zoom']['start']:.2f} → {params2['zoom']['end']:.2f}")
    print(f"  Pan X: {params2['pan_x']['start']:.2f} → {params2['pan_x']['end']:.2f}")
    assert params2['pan_x']['end'] != params2['pan_x']['start'], "中景pan应该有平移"
    print("  ✓ 测试通过：中景场景正确设置pan")
    
    # 测试场景3：特写场景（应该轻微zoom）
    scene3 = {
        "visual": {
            "composition": "close-up, 特写",
            "motion": {"type": "static"}
        }
    }
    
    params3 = deforum.generate_motion_params_from_scene(scene3)
    print(f"\n场景3（特写）:")
    print(f"  Zoom: {params3['zoom']['start']:.2f} → {params3['zoom']['end']:.2f}")
    assert 1.0 <= params3['zoom']['end'] <= 1.1, "特写应该轻微zoom"
    print("  ✓ 测试通过：特写场景正确设置轻微zoom")
    
    print("\n✓ Deforum相机运动模块测试全部通过！\n")


def test_svd_parameter_generator():
    """测试SVD参数生成模块"""
    print("=" * 60)
    print("测试2: SVD参数生成模块")
    print("=" * 60)
    
    generator = SVDParameterGenerator()
    analyzer = SceneMotionAnalyzer()
    
    # 测试场景1：人物特写
    scene1 = {
        "visual": {
            "composition": "Han Li close-up, 韩立特写",
        },
        "characters": [{"name": "hanli"}]
    }
    
    analysis1 = analyzer.analyze(scene1)
    params1 = generator.generate_params(scene1, analysis1)
    print(f"\n场景1（人物特写）:")
    print(f"  motion_bucket_id: {params1['motion_bucket_id']}")
    print(f"  noise_aug_strength: {params1['noise_aug_strength']}")
    print(f"  num_inference_steps: {params1['num_inference_steps']}")
    assert params1['motion_bucket_id'] <= 1.6, "人物特写应该较低motion_bucket_id"
    print("  ✓ 测试通过：人物特写参数正确")
    
    # 测试场景2：物体展开
    scene2 = {
        "visual": {
            "composition": "golden scroll unfurling, 金色卷轴展开",
            "fx": "scroll-unfurling effect"
        },
        "description": "卷轴缓缓展开"
    }
    
    analysis2 = analyzer.analyze(scene2)
    params2 = generator.generate_params(scene2, analysis2)
    print(f"\n场景2（物体展开）:")
    print(f"  motion_bucket_id: {params2['motion_bucket_id']}")
    print(f"  noise_aug_strength: {params2['noise_aug_strength']}")
    print(f"  decode_chunk_size: {params2['decode_chunk_size']}")
    assert params2['motion_bucket_id'] >= 1.6, "物体展开应该使用中等运动参数（1.6）"
    assert params2['motion_bucket_id'] <= 1.8, "物体展开不应该过高（避免过快运动）"
    print("  ✓ 测试通过：物体展开参数正确（使用中等运动参数1.6）")
    
    # 测试场景3：镜头运动（pan）
    scene3 = {
        "visual": {
            "motion": {"type": "pan", "direction": "left_to_right"}
        },
        "camera": "pan shot"
    }
    
    analysis3 = analyzer.analyze(scene3)
    params3 = generator.generate_params(scene3, analysis3)
    print(f"\n场景3（镜头pan）:")
    print(f"  motion_bucket_id: {params3['motion_bucket_id']}")
    print(f"  noise_aug_strength: {params3['noise_aug_strength']}")
    assert params3['motion_bucket_id'] >= 1.6, "镜头pan应该使用中等运动参数（1.6）"
    assert params3['motion_bucket_id'] <= 1.8, "镜头pan不应该过高（避免过快运动）"
    print("  ✓ 测试通过：镜头pan参数正确（使用中等运动参数1.6）")
    
    print("\n✓ SVD参数生成模块测试全部通过！\n")


def test_scene_type_classifier():
    """测试场景类型分类模块"""
    print("=" * 60)
    print("测试3: 场景类型分类模块")
    print("=" * 60)
    
    classifier = SceneTypeClassifier()
    
    # 测试场景1：人物场景
    scene1 = {
        "description": "Han Li standing in desert, 韩立站在沙漠中",
        "characters": [{"name": "hanli"}],
        "visual": {
            "composition": "Han Li, medium shot, 中景"
        }
    }
    
    result1 = classifier.classify(scene1)
    print(f"\n场景1（人物场景）:")
    print(f"  primary_type: {result1['primary_type']}")
    print(f"  camera_type: {result1['camera_type']}")
    print(f"  has_character: {result1['has_character']}")
    print(f"  recommended_strategy: {result1['recommended_strategy']}")
    assert result1['primary_type'] == "character_scene", "应该是人物场景"
    assert result1['has_character'] == True, "应该有角色"
    assert result1['recommended_strategy'] == "svd", "人物场景应该推荐SVD"
    print("  ✓ 测试通过：人物场景分类正确")
    
    # 测试场景2：环境场景
    scene2 = {
        "description": "desert landscape, 沙漠风景",
        "visual": {
            "composition": "wide desert view, 远景沙漠"
        }
    }
    
    result2 = classifier.classify(scene2)
    print(f"\n场景2（环境场景）:")
    print(f"  primary_type: {result2['primary_type']}")
    print(f"  has_character: {result2['has_character']}")
    print(f"  recommended_strategy: {result2['recommended_strategy']}")
    assert result2['primary_type'] == "environment_scene", "应该是环境场景"
    assert result2['has_character'] == False, "应该没有角色"
    print("  ✓ 测试通过：环境场景分类正确")
    
    # 测试场景3：物体焦点场景
    scene3 = {
        "description": "golden scroll unfurling, 金色卷轴展开",
        "visual": {
            "composition": "golden scroll, 金色卷轴",
            "fx": "scroll-unfurling"
        }
    }
    
    result3 = classifier.classify(scene3)
    print(f"\n场景3（物体焦点）:")
    print(f"  primary_type: {result3['primary_type']}")
    print(f"  has_object_motion: {result3['has_object_motion']}")
    print(f"  recommended_strategy: {result3['recommended_strategy']}")
    assert result3['primary_type'] == "object_focus", "应该是物体焦点场景"
    assert result3['has_object_motion'] == True, "应该有物体运动"
    assert result3['recommended_strategy'] == "svd", "物体运动应该推荐SVD"
    print("  ✓ 测试通过：物体焦点场景分类正确")
    
    # 测试场景4：静态场景
    scene4 = {
        "description": "still, motionless, 静止不动",
        "visual": {
            "composition": "static view",
            "motion": {"type": "static"}
        }
    }
    
    result4 = classifier.classify(scene4)
    print(f"\n场景4（静态场景）:")
    print(f"  primary_type: {result4['primary_type']}")
    print(f"  recommended_strategy: {result4['recommended_strategy']}")
    assert result4['primary_type'] == "static_scene", "应该是静态场景"
    assert result4['recommended_strategy'] == "deforum", "静态场景应该推荐Deforum"
    print("  ✓ 测试通过：静态场景分类正确")
    
    print("\n✓ 场景类型分类模块测试全部通过！\n")


def test_integration():
    """测试模块集成"""
    print("=" * 60)
    print("测试4: 模块集成测试")
    print("=" * 60)
    
    # 创建一个完整的场景
    scene = {
        "description": "Han Li lying on gray-green sand, motionless, 韩立躺在青灰色沙地上一动不动",
        "characters": [{"name": "hanli"}],
        "camera": "wide shot",
        "visual": {
            "composition": "Han Li lying on sand, wide shot, 韩立躺在沙地上，远景",
            "motion": {"type": "static"}
        }
    }
    
    # 1. 场景类型分类
    classifier = SceneTypeClassifier()
    classification = classifier.classify(scene)
    print(f"\n场景类型分类:")
    print(f"  primary_type: {classification['primary_type']}")
    print(f"  recommended_strategy: {classification['recommended_strategy']}")
    
    # 2. 运动分析
    analyzer = SceneMotionAnalyzer()
    analysis = analyzer.analyze(scene)
    print(f"\n运动分析:")
    print(f"  use_svd: {analysis['use_svd']}")
    print(f"  motion_intensity: {analysis['motion_intensity']}")
    
    # 3. SVD参数生成（如果需要）
    if analysis['use_svd']:
        generator = SVDParameterGenerator()
        svd_params = generator.generate_params(scene, analysis)
        print(f"\nSVD参数:")
        print(f"  motion_bucket_id: {svd_params['motion_bucket_id']}")
        print(f"  noise_aug_strength: {svd_params['noise_aug_strength']}")
    else:
        # 4. Deforum参数生成（如果使用Deforum）
        deforum = DeforumCameraMotion()
        deforum_params = deforum.generate_motion_params_from_scene(scene)
        print(f"\nDeforum参数:")
        print(f"  Zoom: {deforum_params['zoom']['start']:.2f} → {deforum_params['zoom']['end']:.2f}")
        print(f"  Pan X: {deforum_params['pan_x']['start']:.2f} → {deforum_params['pan_x']['end']:.2f}")
    
    print("\n✓ 模块集成测试通过！\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("优化后的视频生成Pipeline测试")
    print("=" * 60 + "\n")
    
    try:
        test_deforum_camera_motion()
        test_svd_parameter_generator()
        test_scene_type_classifier()
        test_integration()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        print("\n优化模块已成功集成，可以开始使用！\n")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

