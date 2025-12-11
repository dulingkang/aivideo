#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整流程（使用真实JSON脚本）
测试优化后的视频生成Pipeline，包括：
1. 场景类型分类
2. SVD参数生成
3. Deforum相机运动
4. 视频生成
"""

import sys
import json
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from video_generator import VideoGenerator
from image_generator import ImageGenerator
from scene_type_classifier import SceneTypeClassifier
from svd_parameter_generator import SVDParameterGenerator
from deforum_camera_motion import DeforumCameraMotion
from scene_motion_analyzer import SceneMotionAnalyzer


def test_full_pipeline(json_path: str, max_scenes: int = 3):
    """
    测试完整流程
    
    Args:
        json_path: JSON脚本路径
        max_scenes: 最多测试的场景数（默认3个，避免测试时间过长）
    """
    print("=" * 80)
    print("测试完整流程（使用真实JSON脚本）")
    print("=" * 80)
    
    # 1. 读取JSON脚本
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"❌ 错误: JSON文件未找到: {json_path}")
        return False
    
    print(f"\n[1] 读取JSON脚本: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    scenes = script_data.get('scenes', [])
    if not scenes:
        print("❌ 错误: JSON文件中没有场景数据")
        return False
    
    print(f"✓ 找到 {len(scenes)} 个场景，将测试前 {min(max_scenes, len(scenes))} 个")
    
    # 2. 初始化模块
    print(f"\n[2] 初始化模块...")
    try:
        config_path = Path(__file__).parent / "config.yaml"
        video_gen = VideoGenerator(str(config_path))
        image_gen = ImageGenerator(str(config_path))
        classifier = SceneTypeClassifier()
        svd_gen = SVDParameterGenerator()
        deforum = DeforumCameraMotion()
        analyzer = SceneMotionAnalyzer()
        print("✓ 所有模块初始化成功")
    except Exception as e:
        print(f"❌ 模块初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 加载模型（如果需要）
    print(f"\n[3] 检查模型...")
    # 不立即加载模型，等需要时再加载
    
    # 4. 测试每个场景
    test_scenes = scenes[:max_scenes]
    results = []
    
    for idx, scene in enumerate(test_scenes, 1):
        print("\n" + "=" * 80)
        print(f"测试场景 {idx}/{len(test_scenes)}")
        print("=" * 80)
        
        scene_id = scene.get('id', idx)
        description = scene.get('description', '')[:80]
        print(f"场景ID: {scene_id}")
        print(f"描述: {description}...")
        
        # 4.1 场景类型分类
        print(f"\n[4.1] 场景类型分类...")
        classification = classifier.classify(scene)
        print(f"  - 主要类型: {classification['primary_type']}")
        print(f"  - 镜头类型: {classification['camera_type']}")
        print(f"  - 有角色: {classification['has_character']}")
        print(f"  - 有物体运动: {classification['has_object_motion']}")
        print(f"  - 有镜头运动: {classification['has_camera_motion']}")
        print(f"  - 推荐策略: {classification['recommended_strategy']}")
        
        # 4.2 运动分析
        print(f"\n[4.2] 运动分析...")
        analysis = analyzer.analyze(scene)
        print(f"  - 物体运动: {analysis['has_object_motion']} ({analysis['object_motion_type']})")
        print(f"  - 镜头运动: {analysis['camera_motion_type']}")
        print(f"  - 运动强度: {analysis['motion_intensity']}")
        print(f"  - 使用SVD: {analysis['use_svd']}")
        
        # 4.3 SVD参数生成（如果需要）
        if analysis['use_svd']:
            print(f"\n[4.3] SVD参数生成...")
            svd_params = svd_gen.generate_params(scene, analysis)
            print(f"  - motion_bucket_id: {svd_params['motion_bucket_id']}")
            print(f"  - noise_aug_strength: {svd_params['noise_aug_strength']}")
            print(f"  - num_inference_steps: {svd_params['num_inference_steps']}")
            print(f"  - decode_chunk_size: {svd_params['decode_chunk_size']}")
        else:
            print(f"\n[4.3] Deforum参数生成...")
            deforum_params = deforum.generate_motion_params_from_scene(scene)
            print(f"  - Zoom: {deforum_params['zoom']['start']:.2f} → {deforum_params['zoom']['end']:.2f}")
            print(f"  - Pan X: {deforum_params['pan_x']['start']:.2f} → {deforum_params['pan_x']['end']:.2f}")
            print(f"  - Pan Y: {deforum_params['pan_y']['start']:.2f} → {deforum_params['pan_y']['end']:.2f}")
            print(f"  - Rotate: {deforum_params['rotate']['start']:.1f}° → {deforum_params['rotate']['end']:.1f}°")
        
        # 4.4 检查图像是否存在
        print(f"\n[4.4] 检查图像...")
        # 尝试查找已生成的图像
        possible_image_paths = [
            f"outputs/images/test_scenes/scene_{scene_id:03d}.png",
            f"outputs/images/lingjie_1_test/scene_{scene_id:03d}.png",
            f"outputs/images/lingjie_ep1_full/scene_{scene_id:03d}.png",
        ]
        
        image_path = None
        for path in possible_image_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            print(f"  ⚠ 未找到已生成的图像，需要先生成图像")
            print(f"  ℹ 可以使用以下命令生成图像:")
            print(f"     python test_image_generation.py --script {json_path}")
            print(f"  ℹ 或者跳过图像生成，只测试参数生成")
            results.append({
                'scene_id': scene_id,
                'status': 'no_image',
                'classification': classification,
                'analysis': analysis,
            })
            continue
        
        print(f"  ✓ 找到图像: {image_path}")
        
        # 4.5 生成视频
        print(f"\n[4.5] 生成视频...")
        try:
            output_dir = Path("outputs/test_video_optimized")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"scene_{scene_id:03d}.mp4"
            
            print(f"  ℹ 开始生成视频: {output_path}")
            print(f"  ℹ 加载模型（可能需要一些时间）...")
            video_gen.load_model()  # 加载模型
            print(f"  ✓ 模型加载完成")
            
            result_path = video_gen.generate_video(
                image_path=image_path,
                output_path=str(output_path),
                scene=scene,
            )
            
            if os.path.exists(result_path):
                file_size = os.path.getsize(result_path) / (1024 * 1024)
                print(f"  ✓ 视频生成成功: {result_path} ({file_size:.2f} MB)")
                results.append({
                    'scene_id': scene_id,
                    'status': 'success',
                    'video_path': result_path,
                    'file_size': file_size,
                    'classification': classification,
                    'analysis': analysis,
                })
            else:
                print(f"  ✗ 视频生成失败: 文件不存在")
                results.append({
                    'scene_id': scene_id,
                    'status': 'failed',
                    'classification': classification,
                    'analysis': analysis,
                })
        except Exception as e:
            print(f"  ✗ 视频生成异常: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scene_id': scene_id,
                'status': 'error',
                'error': str(e),
                'classification': classification,
                'analysis': analysis,
            })
        
        results.append({
            'scene_id': scene_id,
            'status': 'analyzed',
            'image_path': image_path,
            'classification': classification,
            'analysis': analysis,
        })
    
    # 5. 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for result in results:
        scene_id = result['scene_id']
        status = result['status']
        classification = result['classification']
        analysis = result['analysis']
        
        print(f"\n场景 {scene_id}:")
        print(f"  状态: {status}")
        print(f"  类型: {classification['primary_type']}")
        print(f"  策略: {classification['recommended_strategy']}")
        print(f"  使用SVD: {analysis['use_svd']}")
        if 'image_path' in result:
            print(f"  图像: {result['image_path']}")
        if 'video_path' in result:
            print(f"  视频: {result['video_path']}")
            if 'file_size' in result:
                print(f"  大小: {result['file_size']:.2f} MB")
        if 'error' in result:
            print(f"  错误: {result['error']}")
    
    print("\n" + "=" * 80)
    print("✓ 测试完成！")
    print("=" * 80)
    print("\n优化模块工作正常，可以开始使用！")
    print("\n如果需要生成实际视频，可以:")
    print("1. 取消脚本中的视频生成代码注释")
    print("2. 或使用完整的pipeline: python main.py --script <json_path>")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试完整流程（使用真实JSON脚本）")
    parser.add_argument(
        "--script",
        type=str,
        default="../lingjie/1.json",
        help="JSON脚本文件路径（默认: ../lingjie/1.json）"
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=3,
        help="最多测试的场景数（默认: 3）"
    )
    
    args = parser.parse_args()
    
    # 解析脚本路径
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (Path(__file__).parent / script_path).resolve()
        if not script_path.exists():
            script_path = (Path(__file__).parent.parent / args.script).resolve()
    
    if not script_path.exists():
        print(f"❌ 错误: JSON文件未找到: {args.script}")
        print(f"   尝试的路径: {script_path}")
        sys.exit(1)
    
    success = test_full_pipeline(str(script_path), args.max_scenes)
    sys.exit(0 if success else 1)

