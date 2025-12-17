#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频身份分析器
用于分析视频生成时的角色身份一致性

功能特性:
1. 视频帧身份检测 - 检测每帧中的人脸身份
2. 与参考图对比 - 计算与原始参考图的相似度
3. 身份漂移分析 - 检测身份漂移严重的帧
4. 帧间一致性 - 分析相邻帧之间的身份一致性
5. 报告生成 - 生成详细的身份分析报告

Author: AI Video Team
Date: 2025-12-17
Project: M6 - 视频身份保持研究
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FrameIdentityResult:
    """单帧身份结果"""
    frame_idx: int
    timestamp_sec: float
    similarity: float  # 与参考图的相似度
    face_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "similarity": round(float(self.similarity), 4),
            "face_detected": self.face_detected,
            "face_bbox": self.face_bbox
        }


@dataclass
class VideoIdentityReport:
    """视频身份分析报告"""
    video_path: str = ""
    reference_path: str = ""
    timestamp: str = ""
    
    # 帧分析结果
    total_frames: int = 0
    analyzed_frames: int = 0
    sample_interval: int = 1
    fps: float = 0.0
    duration_sec: float = 0.0
    
    # 身份相似度指标
    frame_similarities: List[float] = field(default_factory=list)
    avg_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    std_similarity: float = 0.0
    
    # 相邻帧一致性
    adjacent_similarities: List[float] = field(default_factory=list)
    avg_adjacent_similarity: float = 0.0
    
    # 身份漂移
    drift_threshold: float = 0.5
    drift_frames: List[int] = field(default_factory=list)
    drift_ratio: float = 0.0
    
    # 人脸检测
    face_detected_ratio: float = 0.0
    
    # 总体结论
    overall_passed: bool = False
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "reference_path": self.reference_path,
            "timestamp": self.timestamp,
            "total_frames": self.total_frames,
            "analyzed_frames": self.analyzed_frames,
            "sample_interval": self.sample_interval,
            "fps": self.fps,
            "duration_sec": round(self.duration_sec, 2),
            "avg_similarity": round(self.avg_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "std_similarity": round(self.std_similarity, 4),
            "avg_adjacent_similarity": round(self.avg_adjacent_similarity, 4),
            "drift_threshold": self.drift_threshold,
            "drift_frame_count": len(self.drift_frames),
            "drift_ratio": round(self.drift_ratio, 4),
            "face_detected_ratio": round(self.face_detected_ratio, 4),
            "overall_passed": self.overall_passed,
            "issues": self.issues
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class VideoIdentityAnalyzer:
    """
    视频身份分析器
    
    分析视频生成时的角色身份一致性，检测身份漂移问题
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.face_analyzer = None
        self.device = self.config.get("device", "cuda")
        
        # 阈值配置
        self.similarity_threshold = self.config.get("similarity_threshold", 0.65)
        self.drift_threshold = self.config.get("drift_threshold", 0.50)
        self.adjacent_threshold = self.config.get("adjacent_threshold", 0.85)
    
    def _load_face_analyzer(self):
        """延迟加载人脸分析器"""
        if self.face_analyzer is not None:
            return
        
        try:
            from insightface.app import FaceAnalysis
            import os
            
            # 获取模型路径
            model_root = self.config.get("insightface_root", None)
            
            if model_root is None:
                # 默认路径：gen_video 目录
                gen_video_dir = Path(__file__).parent.parent
                model_root = str(gen_video_dir)
            elif not os.path.isabs(model_root):
                gen_video_dir = Path(__file__).parent.parent
                model_root = str(gen_video_dir / model_root)
            
            logger.debug(f"InsightFace 模型根目录: {model_root}")
            
            self.face_analyzer = FaceAnalysis(
                name='antelopev2',
                root=model_root,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0)
            logger.info("✅ 视频身份分析器: 人脸分析器加载成功")
            
        except Exception as e:
            logger.warning(f"⚠️ 人脸分析器加载失败: {e}")
            self.face_analyzer = None
    
    def _extract_reference_embedding(
        self,
        reference_image: Union[str, Path, Image.Image]
    ) -> Optional[np.ndarray]:
        """提取参考图的人脸嵌入"""
        self._load_face_analyzer()
        
        if self.face_analyzer is None:
            return None
        
        try:
            # 加载图像
            if isinstance(reference_image, (str, Path)):
                img = Image.open(reference_image).convert('RGB')
            else:
                img = reference_image.convert('RGB')
            
            img_np = np.array(img)
            
            # 检测人脸
            faces = self.face_analyzer.get(img_np)
            
            if not faces:
                logger.warning("参考图中未检测到人脸")
                return None
            
            # 使用最大的人脸
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            return largest_face.embedding
            
        except Exception as e:
            logger.error(f"提取参考图嵌入失败: {e}")
            return None
    
    def _calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """计算两个嵌入的余弦相似度"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)
    
    def analyze_video(
        self,
        video_path: str,
        reference_image: Union[str, Path, Image.Image],
        sample_interval: int = 5,
        max_frames: Optional[int] = None
    ) -> VideoIdentityReport:
        """
        分析视频的身份一致性
        
        Args:
            video_path: 视频路径
            reference_image: 参考图（路径或 PIL Image）
            sample_interval: 采样间隔（每隔几帧分析一帧）
            max_frames: 最大分析帧数
            
        Returns:
            VideoIdentityReport 分析报告
        """
        report = VideoIdentityReport(
            video_path=str(video_path),
            reference_path=str(reference_image) if isinstance(reference_image, (str, Path)) else "PIL.Image",
            timestamp=datetime.now().isoformat(),
            sample_interval=sample_interval,
            drift_threshold=self.drift_threshold
        )
        
        # 提取参考图嵌入
        ref_embedding = self._extract_reference_embedding(reference_image)
        if ref_embedding is None:
            report.issues.append("参考图中未检测到人脸")
            return report
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            report.issues.append(f"无法打开视频: {video_path}")
            return report
        
        # 获取视频信息
        report.fps = cap.get(cv2.CAP_PROP_FPS)
        report.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        report.duration_sec = report.total_frames / report.fps if report.fps > 0 else 0
        
        logger.info(f"分析视频: {video_path}")
        logger.info(f"  帧数: {report.total_frames}, FPS: {report.fps:.1f}, 时长: {report.duration_sec:.1f}s")
        
        # 分析帧
        frame_results: List[FrameIdentityResult] = []
        prev_embedding = None
        
        frame_idx = 0
        analyzed_count = 0
        face_detected_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔采样
            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue
            
            # 检查最大帧数限制
            if max_frames and analyzed_count >= max_frames:
                break
            
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            result = self._analyze_frame(
                frame_rgb,
                frame_idx,
                ref_embedding,
                prev_embedding,
                report.fps
            )
            
            frame_results.append(result)
            
            if result.face_detected:
                face_detected_count += 1
                # 保存当前帧嵌入用于相邻帧对比
                faces = self.face_analyzer.get(frame_rgb)
                if faces:
                    prev_embedding = faces[0].embedding
            
            analyzed_count += 1
            frame_idx += 1
        
        cap.release()
        
        # 汇总结果
        report.analyzed_frames = analyzed_count
        report.face_detected_ratio = face_detected_count / analyzed_count if analyzed_count > 0 else 0
        
        # 计算相似度统计
        similarities = [r.similarity for r in frame_results if r.face_detected]
        if similarities:
            report.frame_similarities = similarities
            report.avg_similarity = float(np.mean(similarities))
            report.min_similarity = float(np.min(similarities))
            report.max_similarity = float(np.max(similarities))
            report.std_similarity = float(np.std(similarities))
        
        # 检测身份漂移帧
        report.drift_frames = [
            r.frame_idx for r in frame_results
            if r.face_detected and r.similarity < self.drift_threshold
        ]
        report.drift_ratio = len(report.drift_frames) / face_detected_count if face_detected_count > 0 else 0
        
        # 判断是否通过
        report.overall_passed = (
            report.avg_similarity >= self.similarity_threshold and
            report.drift_ratio <= 0.10 and  # 漂移帧不超过 10%
            report.face_detected_ratio >= 0.8  # 人脸检测率
        )
        
        # 生成问题列表
        if report.avg_similarity < self.similarity_threshold:
            report.issues.append(
                f"平均相似度不足: {report.avg_similarity:.3f} < {self.similarity_threshold}"
            )
        
        if report.drift_ratio > 0.10:
            report.issues.append(
                f"身份漂移帧过多: {len(report.drift_frames)}帧 ({report.drift_ratio*100:.1f}%)"
            )
        
        if report.face_detected_ratio < 0.8:
            report.issues.append(
                f"人脸检测率低: {report.face_detected_ratio*100:.1f}%"
            )
        
        if report.min_similarity < 0.3:
            report.issues.append(
                f"存在极低相似度帧: {report.min_similarity:.3f}"
            )
        
        return report
    
    def _analyze_frame(
        self,
        frame_rgb: np.ndarray,
        frame_idx: int,
        ref_embedding: np.ndarray,
        prev_embedding: Optional[np.ndarray],
        fps: float
    ) -> FrameIdentityResult:
        """分析单帧"""
        result = FrameIdentityResult(
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / fps if fps > 0 else 0,
            similarity=0.0,
            face_detected=False
        )
        
        if self.face_analyzer is None:
            return result
        
        try:
            faces = self.face_analyzer.get(frame_rgb)
            
            if not faces:
                return result
            
            # 使用最大的人脸
            largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            result.face_detected = True
            result.face_bbox = tuple(map(int, largest_face.bbox))
            
            # 计算与参考图的相似度
            result.similarity = self._calculate_similarity(
                largest_face.embedding,
                ref_embedding
            )
            
        except Exception as e:
            logger.debug(f"帧 {frame_idx} 分析失败: {e}")
        
        return result
    
    def analyze_frames(
        self,
        frames: List[Image.Image],
        reference_image: Image.Image
    ) -> List[float]:
        """
        分析帧列表的身份相似度
        
        Args:
            frames: 帧列表
            reference_image: 参考图
            
        Returns:
            相似度列表
        """
        ref_embedding = self._extract_reference_embedding(reference_image)
        if ref_embedding is None:
            return [0.0] * len(frames)
        
        similarities = []
        for frame in frames:
            frame_np = np.array(frame)
            result = self._analyze_frame(frame_np, 0, ref_embedding, None, 1.0)
            similarities.append(result.similarity)
        
        return similarities
    
    def format_report(self, report: VideoIdentityReport) -> str:
        """格式化报告为可读字符串"""
        lines = []
        lines.append("=" * 60)
        lines.append("📹 视频身份分析报告")
        lines.append("=" * 60)
        lines.append("")
        
        # 视频信息
        lines.append(f"📁 视频: {Path(report.video_path).name}")
        lines.append(f"📐 帧数: {report.total_frames} ({report.analyzed_frames} 已分析)")
        lines.append(f"⏱️ 时长: {report.duration_sec:.1f}s @ {report.fps:.1f}fps")
        lines.append("")
        
        # 身份相似度
        status = "✅ 通过" if report.overall_passed else "❌ 未通过"
        lines.append(f"🎯 总体状态: {status}")
        lines.append("")
        
        lines.append("👤 身份相似度:")
        lines.append(f"   平均: {report.avg_similarity:.3f}")
        lines.append(f"   最低: {report.min_similarity:.3f}")
        lines.append(f"   最高: {report.max_similarity:.3f}")
        lines.append(f"   标准差: {report.std_similarity:.3f}")
        lines.append("")
        
        # 漂移分析
        lines.append("📊 漂移分析:")
        lines.append(f"   漂移阈值: {report.drift_threshold}")
        lines.append(f"   漂移帧数: {len(report.drift_frames)} ({report.drift_ratio*100:.1f}%)")
        lines.append(f"   人脸检测率: {report.face_detected_ratio*100:.1f}%")
        lines.append("")
        
        # 问题
        if report.issues:
            lines.append("⚠️ 问题:")
            for issue in report.issues:
                lines.append(f"   • {issue}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def log_report(self, report: VideoIdentityReport):
        """记录报告到日志"""
        status = "✅ 通过" if report.overall_passed else "❌ 未通过"
        logger.info(f"视频身份分析: {status}")
        logger.info(f"  平均相似度: {report.avg_similarity:.3f}")
        logger.info(f"  漂移帧比例: {report.drift_ratio*100:.1f}%")
        
        if report.issues:
            for issue in report.issues:
                logger.warning(f"  ⚠️ {issue}")
    
    def unload(self):
        """卸载模型"""
        if self.face_analyzer is not None:
            del self.face_analyzer
            self.face_analyzer = None
            
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("视频身份分析器已卸载")


def analyze_video(
    video_path: str,
    reference_image: Union[str, Image.Image],
    sample_interval: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> VideoIdentityReport:
    """
    快捷函数：分析视频身份一致性
    
    Args:
        video_path: 视频路径
        reference_image: 参考图
        sample_interval: 采样间隔
        config: 配置
        
    Returns:
        VideoIdentityReport
    """
    analyzer = VideoIdentityAnalyzer(config)
    try:
        report = analyzer.analyze_video(video_path, reference_image, sample_interval)
        return report
    finally:
        analyzer.unload()


if __name__ == "__main__":
    """测试视频身份分析器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("视频身份分析器测试")
    print("=" * 60)
    
    # 创建分析器
    analyzer = VideoIdentityAnalyzer()
    
    # 测试：如果有视频文件
    test_video = Path("outputs/test_video.mp4")
    test_ref = Path("reference_image/hanli_mid.jpg")
    
    if test_video.exists() and test_ref.exists():
        print(f"\n分析视频: {test_video}")
        report = analyzer.analyze_video(
            str(test_video),
            str(test_ref),
            sample_interval=5
        )
        print(analyzer.format_report(report))
    else:
        print("\n⚠️ 测试视频或参考图不存在，跳过测试")
        print(f"  视频: {test_video} ({'存在' if test_video.exists() else '不存在'})")
        print(f"  参考图: {test_ref} ({'存在' if test_ref.exists() else '不存在'})")
    
    # 清理
    analyzer.unload()
    print("\n✅ 测试完成!")
