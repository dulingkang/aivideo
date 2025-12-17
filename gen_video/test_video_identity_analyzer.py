#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频身份分析器测试脚本
测试 VideoIdentityAnalyzer 的功能
"""

import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_video_identity_analyzer():
    """测试视频身份分析器"""
    print("=" * 60)
    print("📹 视频身份分析器测试")
    print("=" * 60)
    
    # 导入模块
    try:
        from utils.video_identity_analyzer import VideoIdentityAnalyzer, VideoIdentityReport
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    
    # 查找可用的视频和参考图
    video_candidates = [
        "outputs/test_hunyuanvideo/test_video.mp4",
        "outputs/test_hunyuanvideo/test_video_v2.mp4",
        "outputs/test_pipeline_single/test_video.mp4",
        "outputs/test_novel/novel_video.mp4",
    ]
    
    ref_candidates = [
        "reference_image/hanli_mid.jpg",
        "reference_image/hanli/front_neutral.jpg",
        "outputs/reference_strength_tuning/shot_close_strength_60.png",
    ]
    
    # 找到存在的视频和参考图
    video_path = None
    ref_path = None
    
    for v in video_candidates:
        if Path(v).exists():
            video_path = v
            break
    
    for r in ref_candidates:
        if Path(r).exists():
            ref_path = r
            break
    
    if not video_path:
        print("⚠️ 未找到可用的测试视频")
        print("   检查的路径:")
        for v in video_candidates:
            print(f"     - {v}: {'存在' if Path(v).exists() else '不存在'}")
        return False
    
    if not ref_path:
        print("⚠️ 未找到可用的参考图")
        print("   检查的路径:")
        for r in ref_candidates:
            print(f"     - {r}: {'存在' if Path(r).exists() else '不存在'}")
        return False
    
    print(f"\n📁 测试视频: {video_path}")
    print(f"📁 参考图: {ref_path}")
    
    # 创建分析器
    print("\n初始化分析器...")
    analyzer = VideoIdentityAnalyzer()
    
    # 分析视频
    print("\n开始分析视频（每 5 帧采样一次）...")
    try:
        report = analyzer.analyze_video(
            video_path=video_path,
            reference_image=ref_path,
            sample_interval=5,  # 每 5 帧采样
            max_frames=50  # 最多分析 50 帧
        )
        
        # 打印报告
        print("\n" + analyzer.format_report(report))
        
        # 记录到日志
        analyzer.log_report(report)
        
        # 保存 JSON 报告
        output_dir = Path("outputs/identity_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"identity_report_{Path(video_path).stem}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        print(f"\n📄 报告已保存: {report_path}")
        
        # 打印关键指标
        print("\n📊 关键指标:")
        print(f"   平均相似度: {report.avg_similarity:.3f} (目标 ≥0.65)")
        print(f"   最低相似度: {report.min_similarity:.3f} (目标 ≥0.50)")
        print(f"   漂移帧比例: {report.drift_ratio*100:.1f}% (目标 ≤10%)")
        print(f"   人脸检测率: {report.face_detected_ratio*100:.1f}% (目标 ≥80%)")
        
        success = report.overall_passed
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        # 清理
        analyzer.unload()
    
    return success


def analyze_multiple_videos():
    """分析多个视频"""
    print("\n" + "=" * 60)
    print("📹 批量视频身份分析")
    print("=" * 60)
    
    from utils.video_identity_analyzer import VideoIdentityAnalyzer
    
    # 查找所有视频
    video_dir = Path("outputs")
    videos = list(video_dir.rglob("*.mp4"))
    
    if not videos:
        print("⚠️ 未找到视频文件")
        return
    
    print(f"找到 {len(videos)} 个视频文件")
    
    # 参考图
    ref_candidates = [
        "reference_image/hanli_mid.jpg",
        "reference_image/hanli/front_neutral.jpg",
    ]
    
    ref_path = None
    for r in ref_candidates:
        if Path(r).exists():
            ref_path = r
            break
    
    if not ref_path:
        print("⚠️ 未找到参考图")
        return
    
    print(f"使用参考图: {ref_path}")
    
    # 分析所有视频
    analyzer = VideoIdentityAnalyzer()
    results = []
    
    for video_path in videos[:5]:  # 最多分析 5 个
        print(f"\n分析: {video_path.name}...")
        try:
            report = analyzer.analyze_video(
                video_path=str(video_path),
                reference_image=ref_path,
                sample_interval=10,
                max_frames=30
            )
            results.append({
                "video": video_path.name,
                "avg_similarity": report.avg_similarity,
                "drift_ratio": report.drift_ratio,
                "passed": report.overall_passed
            })
            status = "✅" if report.overall_passed else "❌"
            print(f"   {status} 平均相似度: {report.avg_similarity:.3f}, 漂移: {report.drift_ratio*100:.1f}%")
        except Exception as e:
            print(f"   ❌ 分析失败: {e}")
    
    analyzer.unload()
    
    # 汇总
    print("\n" + "=" * 60)
    print("📊 汇总结果")
    print("=" * 60)
    
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"通过率: {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")
        
        avg_sim = sum(r["avg_similarity"] for r in results) / len(results)
        print(f"平均相似度: {avg_sim:.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("视频身份分析器测试")
    print("=" * 60)
    
    # 测试 1：单视频分析
    success = test_video_identity_analyzer()
    
    # 测试 2：批量分析（可选）
    # analyze_multiple_videos()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试通过!")
    else:
        print("⚠️ 测试未通过，请检查日志")
    print("=" * 60)
