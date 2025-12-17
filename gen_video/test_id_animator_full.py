#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ID-Animator 完整测试脚本
测试视频身份保持功能（M6 里程碑）

测试内容:
1. 人脸嵌入提取
2. AnimateDiff Pipeline 加载
3. 视频生成（基础版）
4. 身份一致性验证

Author: AI Video Team
Date: 2025-12-18
Project: M6 - 视频身份保持
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_face_embedding():
    """测试 1: 人脸嵌入提取"""
    print("\n" + "=" * 60)
    print("🧪 测试 1: 人脸嵌入提取")
    print("=" * 60)
    
    from id_animator_engine import IDAnimatorEngine
    
    # 创建引擎
    engine = IDAnimatorEngine({
        "model_dir": "models",
        "id_strength": 0.7,
    })
    
    # 查找参考图
    ref_candidates = [
        "reference_image/hanli_mid.jpg",
        "reference_image/hanli/front_neutral.jpg",
        "character_profiles/hanli/references/front_neutral.png",
    ]
    
    ref_path = None
    for r in ref_candidates:
        if Path(r).exists():
            ref_path = r
            break
    
    if not ref_path:
        print("❌ 未找到参考图")
        print("   检查的路径:")
        for r in ref_candidates:
            print(f"     - {r}: {'存在' if Path(r).exists() else '不存在'}")
        return False, None
    
    print(f"📁 参考图: {ref_path}")
    
    # 加载人脸分析器
    print("\n加载人脸分析器...")
    engine._load_face_analyzer()
    
    if engine.face_analyzer is None:
        print("❌ 人脸分析器加载失败")
        return False, None
    
    print("✅ 人脸分析器加载成功")
    
    # 提取嵌入
    print("\n提取人脸嵌入...")
    embedding = engine.extract_face_embedding(ref_path)
    
    if embedding is None:
        print("❌ 人脸嵌入提取失败")
        return False, None
    
    print(f"✅ 人脸嵌入提取成功")
    print(f"   嵌入维度: {embedding.shape}")
    print(f"   嵌入范围: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    # 清理
    engine.unload()
    
    return True, ref_path


def test_animatediff_pipeline():
    """测试 2: AnimateDiff Pipeline 加载"""
    print("\n" + "=" * 60)
    print("🧪 测试 2: AnimateDiff Pipeline 加载")
    print("=" * 60)
    
    import torch
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    from id_animator_engine import IDAnimatorEngine
    
    # 创建引擎
    engine = IDAnimatorEngine({
        "model_dir": "models",
        "id_strength": 0.7,
        "num_frames": 16,
        "fps": 8,
    })
    
    print("\n加载 AnimateDiff Pipeline...")
    start_time = time.time()
    
    try:
        engine._load_animatediff_pipeline()
        load_time = time.time() - start_time
        print(f"✅ Pipeline 加载成功 ({load_time:.1f}s)")
        
        # 显示 Pipeline 信息
        if engine.pipeline is not None:
            print(f"\nPipeline 信息:")
            print(f"   类型: {type(engine.pipeline).__name__}")
            print(f"   设备: {engine.pipeline.device if hasattr(engine.pipeline, 'device') else 'N/A'}")
            print(f"   步数: {engine.num_inference_steps}")
        
        # 清理
        engine.unload()
        return True
        
    except Exception as e:
        print(f"❌ Pipeline 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_generation(ref_path: str = None):
    """测试 3: 视频生成"""
    print("\n" + "=" * 60)
    print("🧪 测试 3: 视频生成")
    print("=" * 60)
    
    if ref_path is None:
        # 查找参考图
        ref_candidates = [
            "reference_image/hanli_mid.jpg",
            "reference_image/hanli/front_neutral.jpg",
        ]
        for r in ref_candidates:
            if Path(r).exists():
                ref_path = r
                break
        
        if ref_path is None:
            print("❌ 未找到参考图")
            return False, None
    
    print(f"📁 参考图: {ref_path}")
    
    from id_animator_engine import IDAnimatorEngine
    
    # 创建引擎
    engine = IDAnimatorEngine({
        "model_dir": "models",
        "id_strength": 0.7,
        "num_frames": 16,  # 短视频测试
        "fps": 8,
        "num_inference_steps": 20,  # 较少步数加速测试
        "guidance_scale": 7.0,
    })
    
    # 输出目录
    output_dir = Path("outputs/id_animator_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"test_video_{timestamp}.mp4"
    
    # 测试 prompt
    prompt = "A Chinese man in traditional Chinese clothes walking slowly, ancient Chinese style, high quality, detailed face"
    
    print(f"\n生成测试视频...")
    print(f"   Prompt: {prompt[:60]}...")
    print(f"   帧数: {engine.num_frames}")
    print(f"   步数: {engine.num_inference_steps}")
    print(f"   输出: {output_path}")
    
    start_time = time.time()
    
    try:
        frames = engine.generate_video(
            prompt=prompt,
            reference_image=ref_path,
            output_path=str(output_path),
            seed=42,  # 固定种子便于复现
        )
        
        gen_time = time.time() - start_time
        print(f"✅ 视频生成成功 ({gen_time:.1f}s)")
        print(f"   生成帧数: {len(frames)}")
        print(f"   输出文件: {output_path}")
        
        # 检查文件大小
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"   文件大小: {size_mb:.2f} MB")
        
        engine.unload()
        return True, str(output_path)
        
    except Exception as e:
        print(f"❌ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        engine.unload()
        return False, None


def test_identity_verification(video_path: str, ref_path: str):
    """测试 4: 身份一致性验证"""
    print("\n" + "=" * 60)
    print("🧪 测试 4: 身份一致性验证")
    print("=" * 60)
    
    if not video_path or not Path(video_path).exists():
        print("❌ 视频文件不存在")
        return False
    
    if not ref_path or not Path(ref_path).exists():
        print("❌ 参考图不存在")
        return False
    
    print(f"📁 视频: {video_path}")
    print(f"📁 参考图: {ref_path}")
    
    try:
        from utils.video_identity_analyzer import VideoIdentityAnalyzer
        
        # 创建分析器
        analyzer = VideoIdentityAnalyzer()
        
        print("\n分析视频身份一致性...")
        report = analyzer.analyze_video(
            video_path=video_path,
            reference_image=ref_path,
            sample_interval=2,  # 每 2 帧采样
        )
        
        # 打印结果
        print("\n" + analyzer.format_report(report))
        
        # 保存报告
        output_dir = Path("outputs/id_animator_test")
        report_path = output_dir / f"identity_report_{Path(video_path).stem}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        print(f"\n📄 报告已保存: {report_path}")
        
        analyzer.unload()
        
        return report.overall_passed
        
    except ImportError as e:
        print(f"⚠️ 无法导入视频分析器: {e}")
        return None
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🚀 ID-Animator 完整测试")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 测试 1: 人脸嵌入提取
    success, ref_path = test_face_embedding()
    results["face_embedding"] = success
    
    if not success:
        print("\n⚠️ 人脸嵌入测试失败，停止后续测试")
        return results
    
    # 测试 2: AnimateDiff Pipeline 加载
    success = test_animatediff_pipeline()
    results["animatediff_pipeline"] = success
    
    if not success:
        print("\n⚠️ Pipeline 加载失败，停止视频生成测试")
        return results
    
    # 测试 3: 视频生成
    success, video_path = test_video_generation(ref_path)
    results["video_generation"] = success
    
    if success and video_path:
        # 测试 4: 身份一致性验证
        success = test_identity_verification(video_path, ref_path)
        results["identity_verification"] = success
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else ("⚠️ 跳过" if passed is None else "❌ 失败")
        print(f"   {test_name}: {status}")
    
    total_passed = sum(1 for v in results.values() if v is True)
    total_tests = len(results)
    print(f"\n✅ 通过: {total_passed}/{total_tests}")
    
    return results


def quick_test():
    """快速测试（仅测试人脸嵌入）"""
    print("\n" + "=" * 60)
    print("⚡ 快速测试模式")
    print("=" * 60)
    
    success, ref_path = test_face_embedding()
    
    if success:
        print("\n✅ 快速测试通过！")
        print("   ID-Animator 基础功能正常")
    else:
        print("\n❌ 快速测试失败")
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ID-Animator 测试脚本")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--skip-video", action="store_true", help="跳过视频生成测试")
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        results = run_all_tests()
        
        # 返回退出码
        all_passed = all(v is True or v is None for v in results.values())
        sys.exit(0 if all_passed else 1)
