#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频质量分析器
用于自动评估生成的视频质量，包括：
- 色彩分析（饱和度、亮度、对比度）
- 帧间一致性（闪烁检测）
- 清晰度评估
- 运动流畅度
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoQualityAnalyzer:
    """视频质量分析器"""
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        分析视频质量
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            质量分析结果字典
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 读取视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        frames = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
                
                # 限制分析帧数（避免内存过大）
                if frame_count >= 120:
                    break
        finally:
            cap.release()
        
        if len(frames) == 0:
            raise ValueError("视频中没有帧")
        
        # 转换为numpy数组
        frames_array = np.array(frames)
        
        # 执行各项分析
        results = {
            "video_path": video_path,
            "frame_count": len(frames),
            "color_analysis": self._analyze_color(frames_array),
            "consistency_analysis": self._analyze_consistency(frames_array),
            "sharpness_analysis": self._analyze_sharpness(frames_array),
            "motion_analysis": self._analyze_motion(frames_array),
            "overall_score": 0.0
        }
        
        # 计算总体评分
        results["overall_score"] = self._calculate_overall_score(results)
        
        return results
    
    def _analyze_color(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        分析色彩质量
        
        Args:
            frames: 帧数组 (N, H, W, 3)
            
        Returns:
            色彩分析结果
        """
        # 转换为HSV色彩空间
        hsv_frames = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_frames.append(hsv)
        hsv_frames = np.array(hsv_frames)
        
        # 计算平均饱和度
        saturation = hsv_frames[:, :, :, 1].mean()
        saturation_std = hsv_frames[:, :, :, 1].std()
        
        # 计算平均亮度
        brightness = hsv_frames[:, :, :, 2].mean()
        brightness_std = hsv_frames[:, :, :, 2].std()
        
        # 计算RGB通道统计
        rgb_mean = frames.mean(axis=(0, 1, 2))
        rgb_std = frames.std(axis=(0, 1, 2))
        
        # 评估色彩质量
        # 饱和度应该在合理范围内（50-150，避免过浓或过淡）
        saturation_score = 1.0
        if saturation > 150:
            saturation_score = max(0.0, 1.0 - (saturation - 150) / 100)  # 过浓扣分
        elif saturation < 50:
            saturation_score = max(0.0, saturation / 50)  # 过淡扣分
        
        # 亮度应该在合理范围内（80-200，避免过暗或过亮）
        brightness_score = 1.0
        if brightness > 200:
            brightness_score = max(0.0, 1.0 - (brightness - 200) / 55)  # 过亮扣分
        elif brightness < 80:
            brightness_score = max(0.0, brightness / 80)  # 过暗扣分
        
        return {
            "saturation": {
                "mean": float(saturation),
                "std": float(saturation_std),
                "score": float(saturation_score),
                "status": "正常" if 50 <= saturation <= 150 else ("过浓" if saturation > 150 else "过淡")
            },
            "brightness": {
                "mean": float(brightness),
                "std": float(brightness_std),
                "score": float(brightness_score),
                "status": "正常" if 80 <= brightness <= 200 else ("过亮" if brightness > 200 else "过暗")
            },
            "rgb_mean": [float(x) for x in rgb_mean],
            "rgb_std": [float(x) for x in rgb_std],
            "color_score": float((saturation_score + brightness_score) / 2)
        }
    
    def _analyze_consistency(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        分析帧间一致性（闪烁检测）
        
        Args:
            frames: 帧数组 (N, H, W, 3)
            
        Returns:
            一致性分析结果
        """
        if len(frames) < 2:
            return {"flicker_score": 1.0, "status": "帧数不足"}
        
        # 计算相邻帧之间的差异
        diffs = []
        for i in range(len(frames) - 1):
            # 转换为灰度图
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # 计算差异
            diff = cv2.absdiff(gray1, gray2)
            diff_mean = diff.mean()
            diffs.append(diff_mean)
        
        diffs = np.array(diffs)
        diff_mean = diffs.mean()
        diff_std = diffs.std()
        
        # 评估一致性
        # 差异均值应该较小（< 10），差异标准差也应该较小（< 5）
        consistency_score = 1.0
        if diff_mean > 20:
            consistency_score = max(0.0, 1.0 - (diff_mean - 20) / 80)  # 差异过大扣分
        if diff_std > 10:
            consistency_score = max(0.0, consistency_score - (diff_std - 10) / 50)  # 差异不稳定扣分
        
        return {
            "flicker_score": float(consistency_score),
            "diff_mean": float(diff_mean),
            "diff_std": float(diff_std),
            "status": "正常" if consistency_score > 0.8 else ("轻微闪烁" if consistency_score > 0.5 else "严重闪烁")
        }
    
    def _analyze_sharpness(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        分析清晰度
        
        Args:
            frames: 帧数组 (N, H, W, 3)
            
        Returns:
            清晰度分析结果
        """
        sharpness_scores = []
        
        for frame in frames:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 使用Laplacian算子计算清晰度
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_scores.append(sharpness)
        
        sharpness_scores = np.array(sharpness_scores)
        sharpness_mean = sharpness_scores.mean()
        sharpness_std = sharpness_scores.std()
        
        # 评估清晰度（Laplacian方差 > 100 通常认为是清晰的）
        sharpness_score = min(1.0, sharpness_mean / 100.0)
        
        return {
            "mean": float(sharpness_mean),
            "std": float(sharpness_std),
            "score": float(sharpness_score),
            "status": "清晰" if sharpness_mean > 100 else ("一般" if sharpness_mean > 50 else "模糊")
        }
    
    def _analyze_motion(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        分析运动流畅度
        
        Args:
            frames: 帧数组 (N, H, W, 3)
            
        Returns:
            运动分析结果
        """
        if len(frames) < 2:
            return {"motion_score": 1.0, "status": "帧数不足"}
        
        # 使用光流法分析运动
        motion_magnitudes = []
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # 计算运动幅度
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_magnitudes.append(magnitude.mean())
            
            prev_gray = curr_gray
        
        motion_magnitudes = np.array(motion_magnitudes)
        motion_mean = motion_magnitudes.mean()
        motion_std = motion_magnitudes.std()
        
        # 评估运动流畅度
        # 运动应该平滑（标准差较小）
        motion_score = 1.0
        if motion_std > 5:
            motion_score = max(0.0, 1.0 - (motion_std - 5) / 20)  # 运动不稳定扣分
        
        return {
            "motion_score": float(motion_score),
            "motion_mean": float(motion_mean),
            "motion_std": float(motion_std),
            "status": "流畅" if motion_score > 0.8 else ("一般" if motion_score > 0.5 else "不流畅")
        }
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        计算总体评分
        
        Args:
            results: 分析结果字典
            
        Returns:
            总体评分（0-100）
        """
        color_score = results["color_analysis"]["color_score"]
        consistency_score = results["consistency_analysis"]["flicker_score"]
        sharpness_score = results["sharpness_analysis"]["score"]
        motion_score = results["motion_analysis"]["motion_score"]
        
        # 加权平均
        overall = (
            color_score * 0.3 +
            consistency_score * 0.3 +
            sharpness_score * 0.2 +
            motion_score * 0.2
        ) * 100
        
        return float(overall)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成质量分析报告
        
        Args:
            results: 分析结果字典
            
        Returns:
            报告字符串
        """
        report = []
        report.append("=" * 60)
        report.append("视频质量分析报告")
        report.append("=" * 60)
        report.append(f"\n视频路径: {results['video_path']}")
        report.append(f"帧数: {results['frame_count']}")
        report.append(f"\n总体评分: {results['overall_score']:.1f}/100")
        
        # 色彩分析
        color = results["color_analysis"]
        report.append("\n" + "-" * 60)
        report.append("色彩分析")
        report.append("-" * 60)
        report.append(f"  饱和度: {color['saturation']['mean']:.1f} ({color['saturation']['status']})")
        report.append(f"  亮度: {color['brightness']['mean']:.1f} ({color['brightness']['status']})")
        report.append(f"  色彩评分: {color['color_score']*100:.1f}/100")
        
        # 一致性分析
        consistency = results["consistency_analysis"]
        report.append("\n" + "-" * 60)
        report.append("一致性分析（闪烁检测）")
        report.append("-" * 60)
        report.append(f"  帧间差异均值: {consistency['diff_mean']:.2f}")
        report.append(f"  帧间差异标准差: {consistency['diff_std']:.2f}")
        report.append(f"  一致性评分: {consistency['flicker_score']*100:.1f}/100 ({consistency['status']})")
        
        # 清晰度分析
        sharpness = results["sharpness_analysis"]
        report.append("\n" + "-" * 60)
        report.append("清晰度分析")
        report.append("-" * 60)
        report.append(f"  Laplacian方差: {sharpness['mean']:.1f}")
        report.append(f"  清晰度评分: {sharpness['score']*100:.1f}/100 ({sharpness['status']})")
        
        # 运动分析
        motion = results["motion_analysis"]
        report.append("\n" + "-" * 60)
        report.append("运动流畅度分析")
        report.append("-" * 60)
        report.append(f"  运动幅度均值: {motion['motion_mean']:.2f}")
        report.append(f"  运动幅度标准差: {motion['motion_std']:.2f}")
        report.append(f"  流畅度评分: {motion['motion_score']*100:.1f}/100 ({motion['status']})")
        
        # 建议
        report.append("\n" + "-" * 60)
        report.append("优化建议")
        report.append("-" * 60)
        
        suggestions = []
        if color['saturation']['mean'] > 150:
            suggestions.append("色彩过浓：建议降低饱和度（saturation_factor: 0.7-0.75）")
        elif color['saturation']['mean'] < 50:
            suggestions.append("色彩过淡：建议提高饱和度（saturation_factor: 0.9-1.0）")
        
        if color['brightness']['mean'] < 80:
            suggestions.append("画面过暗：建议增加亮度（brightness_factor: 1.2-1.3）")
        elif color['brightness']['mean'] > 200:
            suggestions.append("画面过亮：建议降低亮度（brightness_factor: 0.9-1.0）")
        
        if consistency['flicker_score'] < 0.8:
            suggestions.append("存在闪烁：建议增加推理步数（num_inference_steps: 40-50）")
        
        if sharpness['score'] < 0.7:
            suggestions.append("清晰度不足：建议检查分辨率设置或使用超分辨率")
        
        if motion['motion_score'] < 0.7:
            suggestions.append("运动不流畅：建议检查插帧设置或增加关键帧数")
        
        if not suggestions:
            suggestions.append("视频质量良好，无需优化")
        
        for i, suggestion in enumerate(suggestions, 1):
            report.append(f"  {i}. {suggestion}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    分析视频质量的便捷函数
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        分析结果字典
    """
    analyzer = VideoQualityAnalyzer()
    return analyzer.analyze(video_path)


if __name__ == "__main__":
    """测试视频质量分析器"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python video_quality_analyzer.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        analyzer = VideoQualityAnalyzer()
        results = analyzer.analyze(video_path)
        report = analyzer.generate_report(results)
        print(report)
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()

